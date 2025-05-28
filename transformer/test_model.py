import torch
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import torch.nn.functional as F
import re
import copy
import math

# --- Imports from your project files ---
# Make sure these files are in the same directory or Python path
try:
    # Try importing from the debug version first, or change to your current main file name
    from fourthGen import CircuitBuilder, Config
except ImportError:
    print("Warning: Could not import from 'SupervisedTransformer_Debug.py', trying 'SupervisedTransformer.py'")
    try:
        from SupervisedTransformer import CircuitBuilder, Config
    except ImportError:
        print("FATAL ERROR: Could not import 'CircuitBuilder' and 'Config'.")
        print("Please ensure 'SupervisedTransformer_Debug.py' or 'SupervisedTransformer.py' is accessible.")
        exit()

from generator import generate_monomials_with_additive_indices
from State import Game
from torch_geometric.data import Batch

# --- Global Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for testing: {device}")

# --- Helper Functions ---

def preprocess_poly_string(poly_str):
    """Preprocess input string to handle common exponent notations and implicit multiplication."""
    poly_str = poly_str.strip()
    # Ensure spaces around operators for safety before specific replacements
    poly_str = poly_str.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ')
    # Convert xN^M -> xN**M
    poly_str = re.sub(r'(x\d+)\^(\d+)', r'\1**\2', poly_str)
    # Convert xNM -> xN**M (e.g., x02 -> x0**2) - Be careful with this!
    poly_str = re.sub(r'(x\d+)(\d+)', r'\1**\2', poly_str)
    # Convert N*xM -> N*xM (e.g., 2x0 -> 2*x0)
    poly_str = re.sub(r'(\d+)(x\d+)', r'\1*\2', poly_str)
    # Add spaces back if lost, then strip extra spaces
    poly_str = poly_str.replace('**', ' ** ').replace('*', ' * ').replace('+', ' + ').replace('-', ' - ')
    return ' '.join(poly_str.split())

def sympy_to_vector(sympy_poly, monomial_to_index, n_vars, mod=50):
    """Convert a sympy polynomial to vector representation."""
    poly_expanded = sp.expand(sympy_poly)
    max_idx = max(monomial_to_index.values())
    vector = torch.zeros(max_idx + 1, dtype=torch.float)
    poly_dict = poly_expanded.as_coefficients_dict()
    symbols = sp.symbols([f"x{i}" for i in range(n_vars)])

    for term, coeff in poly_dict.items():
        exponents = [0] * n_vars
        if term.is_number: # Handles constant term (including 1)
            pass # exponents remain all zero
        else:
            # Use as_powers_dict for robust exponent extraction
            powers = term.as_powers_dict()
            for i, symbol in enumerate(symbols):
                exponents[i] = powers.get(symbol, 0)

        exponent_tuple = tuple(exponents)
        if exponent_tuple in monomial_to_index:
            idx = monomial_to_index[exponent_tuple]
            vector[idx] = float(coeff % mod)
        else:
             print(f"Warning: Monomial {exponent_tuple} (from {term}) not found in index.")

    return vector

def beam_search_build_circuit(model, target_poly_vec, target_poly_sp, config, index_to_monomial, beam_width=10, max_steps=15):
    """Use Beam Search with the trained model to build a circuit."""
    model.eval()
    initial_game = Game(target_poly_sp, target_poly_vec.unsqueeze(0), config).to(device)

    # Each beam: (game_state, log_probability_score, steps_list)
    beams = [(initial_game, 0.0, [])]

    with torch.no_grad():
        for step_num in range(max_steps):
            new_beams = []
            all_beams_ended = True

            for game, score, steps in beams:
                # If a beam found a solution or hit max steps, keep it but don't expand.
                if game.is_done():
                    new_beams.append((game, score, steps))
                    continue

                all_beams_ended = False
                state_tuple = game.observe()
                circuit_graph, target_poly, circuit_actions, mask = state_tuple

                # Get model predictions (use T=1.0 for greedy-like Beam search)
                action_logits, _ = model(
                    Batch.from_data_list([circuit_graph.to(device)]),
                    target_poly.to(device),
                    circuit_actions,
                    mask.to(device)
                )

                log_probs = F.log_softmax(action_logits[0], dim=0)
                valid_indices = torch.where(mask[0])[0]

                if len(valid_indices) == 0:
                    new_beams.append((game, score, steps))
                    continue

                valid_log_probs = log_probs[valid_indices]
                # Get top k valid actions
                top_k_log_probs, top_k_local_indices = torch.topk(valid_log_probs, min(beam_width, len(valid_log_probs)))

                for log_prob_val, local_idx in zip(top_k_log_probs, top_k_local_indices):
                    action_idx = valid_indices[local_idx].item()

                    new_game = copy.deepcopy(game)
                    new_game.take_action(action_idx)
                    new_score = score + log_prob_val.item() # Add log probs (higher is better)

                    op, node1, node2 = new_game.actions_taken[-1]
                    new_steps_list = steps + [{
                        'step': len(new_game.actions_taken) + config.n_variables,
                        'operation': op, 'node1': node1, 'node2': node2,
                        'result': new_game.exprs[-1]
                    }]

                    new_beams.append((new_game, new_score, new_steps_list))

            # Sort new beams by score and keep top k
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Check if any beam succeeded
            for game, score, steps in beams:
                if game.exprs and sp.expand(game.exprs[-1] - target_poly_sp) == 0:
                    print(f"Found solution after {step_num + 1} steps.")
                    return steps, True

            if all_beams_ended or not beams:
                break

    # If max steps reached, return the best beam found
    best_game, _, best_steps = beams[0]
    success = best_game.exprs and sp.expand(best_game.exprs[-1] - target_poly_sp) == 0
    return best_steps, success


def print_circuit(steps, n_vars):
    """Pretty print the circuit"""
    print("\nArithmetic Circuit:")
    print("-" * 50)

    nodes = {}
    for i in range(n_vars):
        nodes[i] = sp.symbols(f'x{i}')
        print(f"Node {i}: {nodes[i]}")
    nodes[n_vars] = sp.S.One
    print(f"Node {n_vars}: {nodes[n_vars]} (constant)")

    for step in steps:
        node_idx = step['step']
        op_symbol = '*' if step['operation'] == 'multiply' else '+'
        n1 = step['node1']
        n2 = step['node2']
        expr1 = nodes.get(n1, f"Node{n1}")
        expr2 = nodes.get(n2, f"Node{n2}")

        print(f"Node {node_idx}: {step['operation']}({n1}, {n2}) = {expr1} {op_symbol} {expr2}")
        print(f"         Result: {step['result']}")
        nodes[node_idx] = step['result']

    print("-" * 50)

# --- Main Execution ---

def main():
    # Load configuration
    config = Config()
    n = config.n_variables
    # Use max_complexity * 2 as degree (as in training)
    d = config.max_complexity * 2

    print(f"--- Running Test with Config: N={n}, C={config.max_complexity} ---")

    # Generate monomial indexing (must match training)
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(n, d)
    max_vector_size = max(monomial_to_index.values()) + 1

    # Load the trained model
    model = CircuitBuilder(config, max_vector_size).to(device)
    model_path = f"ppo_model_n{config.n_variables}_C{config.max_complexity}.pt"
    best_sup_model_path = f"best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt"

    print(f"Attempting to load PPO model: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded PPO model from {model_path}")
    except FileNotFoundError:
        print(f"PPO model not found. Trying best supervised model: {best_sup_model_path}")
        try:
             model.load_state_dict(torch.load(best_sup_model_path, map_location=device))
             print(f"Successfully loaded supervised model from {best_sup_model_path}")
        except FileNotFoundError:
             print(f"Error: No suitable model found ({model_path} or {best_sup_model_path}). Please train a model first.")
             return
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        return

    # Create symbolic variables based on config
    vars_str = [f'x{i}' for i in range(n)]
    symbols_tuple = sp.symbols(vars_str)
    # Ensure symbols_tuple is always iterable, even if N=1
    symbols_list = list(symbols_tuple) if isinstance(symbols_tuple, tuple) else [symbols_tuple]
    local_dict = {str(s): s for s in symbols_list}

    print(f"\nPolynomial Circuit Builder")
    print(f"Variables: {', '.join(vars_str)}")
    print(f"Max complexity: {config.max_complexity}")
    print("\nNote: Use ** or ^ for exponents (e.g., x0**2). Use * for multiplication.")

    while True:
        print("\n" + "="*50)
        poly_str = input("Enter a polynomial (or 'quit' to exit): ").strip()

        if poly_str.lower() == 'quit':
            break

        try:
            poly_str_processed = preprocess_poly_string(poly_str)
            print(f"Processed input: {poly_str_processed}")

            poly_sp = parse_expr(poly_str_processed,
                                 transformations=(standard_transformations + (implicit_multiplication_application,)),
                                 local_dict=local_dict)
            poly_sp = sp.expand(poly_sp)
            print(f"\nTarget polynomial (expanded): {poly_sp}")

            poly_vec = sympy_to_vector(poly_sp, monomial_to_index, n, config.mod).to(device)

            print("\nBuilding circuit with Beam Search...")
            steps, success = beam_search_build_circuit(
                model, poly_vec, poly_sp, config, index_to_monomial,
                beam_width=10, # You can adjust beam width
                max_steps=config.max_complexity + 5 # Allow a few extra steps
            )

            if success:
                print("\n✓ Successfully found circuit!")
                print_circuit(steps, n)
            else:
                print("\n✗ Could not find exact circuit within step/beam limit.")
                if steps:
                    print(f"Best expression found: {steps[-1]['result']}")
                    print_circuit(steps, n)
                else:
                    print("No circuit could be constructed.")

        except Exception as e:
            print(f"Error processing polynomial: {e}")
            print("Please check your input format. Examples for N=2:")
            print("  x0 + x1")
            print("  x0**2 + 2*x0*x1 + x1**2")
            print("  x0*(x1 + 1)")

if __name__ == "__main__":
    main()