import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from enhanced_models import EnhancedMCTS, AdvancedCircuitBuilder, Config
from generator import generate_monomials_with_additive_indices, create_polynomial_vector
import argparse

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_mcts_search(model, target_poly, actions, polynomials, config, max_nodes, index_to_monomial, monomial_to_index):
    """Visualize the MCTS search process"""
    from circuit_data import CircuitDataset
    
    # Create dummy dataset for helper functions
    dummy_dataset = CircuitDataset(index_to_monomial, monomial_to_index, len(target_poly), config, size=0)
    
    # Create circuit graph
    circuit_graph = dummy_dataset.actions_to_graph(actions)
    
    # Calculate mask
    mask = dummy_dataset.get_available_actions_mask(actions, max_nodes)
    
    # Create MCTS
    mcts = EnhancedMCTS(model, config)
    
    # Convert polynomial to tensor
    target_poly_tensor = torch.tensor(target_poly, dtype=torch.float, device=device)
    
    # State for MCTS
    state = (circuit_graph, target_poly_tensor, actions, mask, polynomials)
    
    # Run MCTS and track visit counts
    print("Running MCTS visualization...")
    visit_history = []
    value_history = []
    
    # Run multiple simulations and collect data after each batch
    total_sims = 500
    batch_size = 50
    
    for i in range(0, total_sims, batch_size):
        sims = min(batch_size, total_sims - i)
        print(f"Running simulations {i+1}-{i+sims}...")
        
        # Run MCTS with current state
        mcts.root = None
        root = mcts.search(state, num_simulations=sims)
        
        # Extract visit counts for top actions
        top_actions = []
        for action, child in root.children.items():
            operation, node1, node2 = decode_action(action, max_nodes)
            top_actions.append({
                'action': action,
                'op': operation,
                'node1': node1,
                'node2': node2,
                'visits': child.visit_count,
                'value': child.value(),
                'prior': child.prior
            })
        
        # Sort by visit count
        top_actions.sort(key=lambda x: x['visits'], reverse=True)
        
        # Keep only top 5
        top_actions = top_actions[:5]
        
        # Add to history
        visit_history.append({
            'simulation': i + sims,
            'actions': top_actions
        })
        
        # Record values of top action over time
        if top_actions:
            value_history.append((i + sims, top_actions[0]['value']))
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot visit counts over time
    plt.subplot(2, 1, 1)
    for action_idx in range(5):  # Up to 5 actions
        sims = []
        visits = []
        labels = []
        
        for h in visit_history:
            if action_idx < len(h['actions']):
                sims.append(h['simulation'])
                visits.append(h['actions'][action_idx]['visits'])
                a = h['actions'][action_idx]
                label = f"{a['op']}({a['node1']}, {a['node2']})"
                labels.append(label)
        
        if sims:
            plt.plot(sims, visits, 'o-', label=labels[-1])
    
    plt.title('MCTS Visit Counts Over Time')
    plt.xlabel('Simulations')
    plt.ylabel('Visit Count')
    plt.legend()
    plt.grid(True)
    
    # Plot value of top action over time
    plt.subplot(2, 1, 2)
    plt.plot([x[0] for x in value_history], [x[1] for x in value_history], 'ro-')
    plt.title('Value of Top Action Over Time')
    plt.xlabel('Simulations')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mcts_visualization.png')
    plt.close()
    
    # Print top actions at the end
    print("\nTop actions after MCTS search:")
    for i, a in enumerate(visit_history[-1]['actions']):
        print(f"{i+1}. {a['op']}({a['node1']}, {a['node2']}): visits={a['visits']}, value={a['value']:.4f}, prior={a['prior']:.4f}")
    
    return visit_history[-1]['actions'][0]  # Return top action

def decode_action(action_idx, max_nodes):
    """Decode action index to operation and node indices"""
    # Extract operation type (0 for add, 1 for multiply)
    op_type = action_idx % 2
    operation = "multiply" if op_type == 1 else "add"
    
    # Extract pair index
    pair_idx = action_idx // 2
    
    # Calculate discriminant
    discriminant = (2 * max_nodes - 1)**2 - 8 * pair_idx
    
    # Handle negative discriminant
    if discriminant < 0:
        return operation, 0, 0
    
    # Find node indices
    node1_id = int((2 * max_nodes - 1 - np.sqrt(discriminant)) / 2)
    node1_id = max(0, min(node1_id, max_nodes - 1))
    
    # Calculate node2_id
    node2_offset = pair_idx - (node1_id * (2 * max_nodes - node1_id - 1)) // 2
    node2_id = node1_id + node2_offset
    node2_id = max(0, min(node2_id, max_nodes - 1))
    
    return operation, node1_id, node2_id

def interactive_simplification(model_path, polynomial, n_variables=3, complexity=10):
    """Run interactive polynomial simplification with visualizations"""
    # Create configuration
    config = Config()
    config.n_variables = n_variables
    config.max_complexity = complexity
    
    # Generate monomial indexing
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(
        n_variables, config.polynomial_degree
    )
    
    # Calculate maximum vector size
    max_vector_size = max(monomial_to_index.values()) + 1
    
    # Initialize model
    model = AdvancedCircuitBuilder(config, max_vector_size).to(device)
    
    # Load model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Parse polynomial
    from enhanced_models import parse_polynomial
    print(f"Parsing polynomial: {polynomial}")
    try:
        poly_vector = parse_polynomial(polynomial, index_to_monomial, monomial_to_index, n_variables)
    except Exception as e:
        print(f"Error parsing polynomial: {e}")
        return
    
    # Initialize circuit with variables and constant
    polynomials = []
    actions = []
    
    # Add variables
    for i in range(n_variables):
        actions.append(("input", None, None))
        poly = create_polynomial_vector(index_to_monomial, monomial_to_index, n_variables, var_idx=i)
        polynomials.append(poly)
    
    # Add constant
    actions.append(("constant", None, None))
    poly = create_polynomial_vector(index_to_monomial, monomial_to_index, n_variables, constant_val=1)
    polynomials.append(poly)
    
    # Interactive circuit building
    print("\nStarting interactive circuit building...")
    
    # Maximum nodes
    max_nodes = n_variables + complexity + 1
    
    # Track best circuit
    best_similarity = 0.0
    best_actions = actions.copy()
    
    # Build circuit step by step
    for step in range(min(10, complexity)):
        print(f"\n--- Step {step+1} ---")
        
        # Print current circuit
        print("Current circuit:")
        variable_names = ["x", "y", "z", "w", "v", "u"]
        for i, (op, in1, in2) in enumerate(actions):
            if op == "input" and i < len(variable_names):
                print(f"{i+1}: {variable_names[i]}")
            elif op == "input":
                print(f"{i+1}: x_{i}")
            elif op == "constant":
                print(f"{i+1}: 1")
            else:
                in1_name = variable_names[in1] if in1 < len(variable_names) else f"{in1+1}"
                in2_name = variable_names[in2] if in2 < len(variable_names) else f"{in2+1}"
                print(f"{i+1}: {op}({in1+1}, {in2+1})  # {op}({in1_name}, {in2_name})")
        
        # Visualize MCTS
        top_action = visualize_mcts_search(
            model, poly_vector, actions, polynomials, config, max_nodes,
            index_to_monomial, monomial_to_index
        )
        
        # Apply top action
        action_idx = top_action['action']
        operation, node1_idx, node2_idx = decode_action(action_idx, max_nodes)
        
        print(f"Applying action: {operation}({node1_idx}, {node2_idx})")
        
        # Apply operation
        from generator import add_polynomials_vector, multiply_polynomials_vector
        if operation == "add":
            new_poly = add_polynomials_vector(polynomials[node1_idx], polynomials[node2_idx], config.mod)
        else:  # multiply
            new_poly = multiply_polynomials_vector(polynomials[node1_idx], polynomials[node2_idx], config.mod)
        
        # Update actions and polynomials
        actions.append((operation, node1_idx, node2_idx))
        polynomials.append(new_poly)
        
        # Calculate similarity
        from enhanced_models import EnhancedMCTS
        mcts = EnhancedMCTS(model, config)
        similarity = mcts.compute_similarity(new_poly, poly_vector)
        
        print(f"Similarity after this step: {similarity:.4f}")
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_actions = actions.copy()
            print(f"New best similarity: {best_similarity:.4f}")
        
        if similarity > 0.95:
            print("Found excellent solution!")
            break
        
        # Pause to allow user to examine the visualization
        time.sleep(1)
    
    # Print final circuit
    print("\n=== Final Circuit ===")
    
    # Use best actions found
    actions = best_actions
    
    for i, (op, in1, in2) in enumerate(actions):
        if op == "input" and i < len(variable_names):
            print(f"{i+1}: {variable_names[i]}")
        elif op == "input":
            print(f"{i+1}: x_{i}")
        elif op == "constant":
            print(f"{i+1}: 1")
        else:
            in1_name = variable_names[in1] if in1 < len(variable_names) else f"{in1+1}"
            in2_name = variable_names[in2] if in2 < len(variable_names) else f"{in2+1}"
            print(f"{i+1}: {op}({in1+1}, {in2+1})  # {op}({in1_name}, {in2_name})")
    
    # Print human-readable form
    if len(actions) > n_variables + 1:
        from enhanced_models import build_readable_circuit
        try:
            readable = build_readable_circuit(actions, variable_names)
            print("\nHuman-readable form:")
            print(readable)
        except Exception as e:
            print(f"Error building readable form: {e}")
    
    print(f"\nFinal similarity: {best_similarity:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Polynomial Simplification")
    parser.add_argument("--model", default="best_rl_model_n3_C10.pt", help="Path to model file")
    parser.add_argument("--polynomial", default="x^2+2xy+y^2", help="Polynomial to simplify")
    parser.add_argument("--variables", type=int, default=3, help="Number of variables")
    parser.add_argument("--complexity", type=int, default=10, help="Maximum complexity")
    
    args = parser.parse_args()
    
    interactive_simplification(
        args.model,
        args.polynomial,
        args.variables,
        args.complexity
    )