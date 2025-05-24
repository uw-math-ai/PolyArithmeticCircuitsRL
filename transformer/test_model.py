import torch
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from generator import generate_monomials_with_additive_indices
from SupervisedTransformer import CircuitBuilder, Config
from State import Game
from utils import vector_to_sympy
import numpy as np
from torch_geometric.data import Batch
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sympy_to_vector(sympy_poly, monomial_to_index, n_vars, mod=50):
    """Convert a sympy polynomial to vector representation"""
    # Expand the polynomial
    poly_expanded = sp.expand(sympy_poly)
    
    # Get the maximum index to determine vector size
    max_idx = max(monomial_to_index.values())
    vector = torch.zeros(max_idx + 1, dtype=torch.float)
    
    # Convert to dict of coefficients
    poly_dict = poly_expanded.as_coefficients_dict()
    
    for term, coeff in poly_dict.items():
        # Get the exponents for each variable
        exponents = [0] * n_vars
        
        if term == 1:  # Constant term
            pass
        elif term.is_symbol:  # Single variable
            var_idx = int(str(term)[1:])  # Extract index from x0, x1, etc.
            exponents[var_idx] = 1
        elif term.is_Pow:  # Single variable with power
            base = term.base
            if base.is_symbol:
                var_idx = int(str(base)[1:])
                exponents[var_idx] = int(term.exp)
        elif term.is_Mul:  # Product of variables
            for factor in term.as_ordered_factors():
                if factor.is_symbol:
                    var_idx = int(str(factor)[1:])
                    exponents[var_idx] += 1
                elif factor.is_Pow and factor.base.is_symbol:
                    var_idx = int(str(factor.base)[1:])
                    exponents[var_idx] += int(factor.exp)
        
        # Find the index for this monomial
        exponent_tuple = tuple(exponents)
        if exponent_tuple in monomial_to_index:
            idx = monomial_to_index[exponent_tuple]
            vector[idx] = float(coeff % mod)
    
    return vector

def build_circuit_with_model(model, target_poly_vec, target_poly_sp, config, index_to_monomial, max_steps=20):
    """Use the trained model to build a circuit for the target polynomial"""
    model.eval()
    
    # Initialize game
    game = Game(target_poly_sp, target_poly_vec.unsqueeze(0), config).to(device)
    
    steps = []
    success = False
    
    with torch.no_grad():
        while not game.is_done() and len(game.actions_taken) < max_steps:
            # Get current state
            circuit_graph, target_poly, circuit_actions, mask = game.observe()
            
            # Get model predictions
            action_logits, _ = model(circuit_graph, target_poly, circuit_actions, mask)
            
            # Find valid actions
            valid_indices = torch.where(mask[0])[0]
            valid_logits = action_logits[0, valid_indices]
            
            # Get the best action (greedy)
            best_local_idx = torch.argmax(valid_logits)
            best_action = valid_indices[best_local_idx].item()
            
            # Decode and take action
            game.take_action(best_action)
            
            # Record the step
            op, node1, node2 = game.actions_taken[-1]
            steps.append({
                'step': len(game.actions_taken) + config.n_variables,
                'operation': op,
                'node1': node1,
                'node2': node2,
                'result': game.exprs[-1]
            })
            
            # Check if we found the target
            if game.exprs[-1] - target_poly_sp == 0:
                success = True
                break
    
    return steps, success

def print_circuit(steps, n_vars):
    """Pretty print the circuit"""
    print("\nArithmetic Circuit:")
    print("-" * 50)
    
    # Print initial variables
    for i in range(n_vars):
        print(f"Node {i}: x{i}")
    print(f"Node {n_vars}: 1 (constant)")
    
    # Print operations
    for step in steps:
        node_idx = step['step']
        op_symbol = '*' if step['operation'] == 'multiply' else '+'
        print(f"Node {node_idx}: {step['operation']}({step['node1']}, {step['node2']}) = Node{step['node1']} {op_symbol} Node{step['node2']}")
        print(f"         Result: {step['result']}")
    
    print("-" * 50)

def main():
    # Load configuration
    config = Config()
    
    # Generate monomial indexing
    n = config.n_variables
    d = config.max_complexity * 2
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(n, d)
    max_vector_size = max(monomial_to_index.values()) + 1
    
    # Load the trained model
    model = CircuitBuilder(config, max_vector_size).to(device)
    model_path = f"transformer_model_n{config.n_variables}_C{config.max_complexity}_mod{config.mod}_GNN{config.num_gnn_layers}_TF{config.num_transformer_layers}.pt"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
    
    # Create symbolic variables
    vars_str = [f'x{i}' for i in range(n)]
    x0, x1, x2 = sp.symbols('x0 x1 x2')  # Define the symbols for parsing
    
    print(f"\nPolynomial Circuit Builder")
    print(f"Variables: {', '.join(vars_str)}")
    print(f"Max complexity: {config.max_complexity}")
    print(f"Working modulo: {config.mod}")
    print("\nNote: Use ^ for exponents (e.g., x0^2) or ** (e.g., x0**2)")
    
    while True:
        print("\n" + "="*50)
        poly_str = input("Enter a polynomial (or 'quit' to exit): ").strip()
        
        if poly_str.lower() == 'quit':
            break
        
        try:
            # Replace ^ with ** for Python compatibility
            poly_str_parsed = poly_str.replace('^', '**')
            
            # Parse the polynomial using sympy
            # Define local namespace with our variables
            local_dict = {'x0': x0, 'x1': x1, 'x2': x2}
            
            # Parse the expression
            poly_sp = parse_expr(poly_str_parsed, 
                               transformations=(standard_transformations + (implicit_multiplication_application,)),
                               local_dict=local_dict)
            
            poly_sp = sp.expand(poly_sp)
            
            print(f"\nTarget polynomial: {poly_sp}")
            
            # Convert to vector
            poly_vec = sympy_to_vector(poly_sp, monomial_to_index, n, config.mod).to(device)
            
            # Build circuit
            print("\nBuilding circuit...")
            steps, success = build_circuit_with_model(model, poly_vec, poly_sp, config, index_to_monomial)
            
            if success:
                print("\n✓ Successfully found circuit!")
                print_circuit(steps, n)
            else:
                print("\n✗ Could not find exact circuit within step limit")
                if steps:
                    print(f"Final expression: {steps[-1]['result']}")
                    print_circuit(steps, n)
            
        except Exception as e:
            print(f"Error parsing polynomial: {e}")
            print("Examples:")
            print("  x0^2 + 2*x0*x1 + x1^2")
            print("  x0*(x1 + x2)")
            print("  x0^3 + x1^3")

if __name__ == "__main__":
    main()