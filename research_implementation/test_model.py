import torch
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import torch.nn.functional as F
import re
from torch_geometric.data import Batch
import copy
import heapq
import hashlib
import pickle

from generator import generate_monomials_with_additive_indices
from State import Game

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

def sympy_to_vector(sympy_poly, monomial_to_index, n_vars, mod=50, max_vector_size=None):
    """Convert a sympy polynomial to a fixed-size vector representation."""
    poly_expanded = sp.expand(sympy_poly)
    vector_length = max(monomial_to_index.values()) + 1 if max_vector_size is None else max_vector_size
    vector = torch.zeros(vector_length, dtype=torch.float)
    poly_dict = poly_expanded.as_coefficients_dict()
    symbols = sp.symbols([f"x{i}" for i in range(n_vars)])

    for term, coeff in poly_dict.items():
        exponents = [0] * n_vars
        if term.is_number:
            pass
        else:
            powers = term.as_powers_dict()
            for i, symbol in enumerate(symbols):
                exponents[i] = powers.get(symbol, 0)

        exponent_tuple = tuple(exponents)
        if exponent_tuple in monomial_to_index:
            idx = monomial_to_index[exponent_tuple]
            vector[idx] = float(coeff % mod)
        else:
            print(f"Warning: Monomial {exponent_tuple} not in index — skipped.")

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
    """Pretty print the arithmetic circuit from steps."""
    print("\nArithmetic Circuit:")
    print("-" * 50)

    nodes = {}
    # Initialize input variable nodes
    for i in range(n_vars):
        nodes[i] = sp.symbols(f'x{i}')
        print(f"Node {i}: {nodes[i]}")

    # Constant node
    const_idx = n_vars
    nodes[const_idx] = sp.S.One
    print(f"Node {const_idx}: {nodes[const_idx]} (constant)")

    # Compute and display each step
    current_idx = const_idx + 1
    for step in steps:
        op = step['operation']
        op_symbol = '*' if op == 'multiply' else '+'
        n1, n2 = step['node1'], step['node2']

        expr1 = nodes.get(n1, f"Node{n1}")
        expr2 = nodes.get(n2, f"Node{n2}")

        result_expr = step.get('result')
        if result_expr is None:
            result_expr = expr1 * expr2 if op == 'multiply' else expr1 + expr2

        print(f"Node {current_idx}: {op}(Node {n1}, Node {n2}) = {expr1} {op_symbol} {expr2}")
        print(f"         Result: {result_expr}")

        nodes[current_idx] = result_expr
        current_idx += 1

    print("-" * 50)


# Tree Search -------------------------

class TreeNode:
    def __init__(self, game, logprob=0.0, first_action=None, actions_trace=None):
        self.game = game
        self.logprob = logprob
        self.first_action = first_action
        self.actions_trace = actions_trace or []
        self.value = None

def game_key(game):
    """
    Generates a unique key including:
    - Actions taken (operation + node pairs)
    - Expressions serialized as strings
    - Current L1 distance
    """
    state_repr = {
        'actions_taken': game.actions_taken,
        'exprs': [str(expr) for expr in game.exprs],
        'current_l1_dist': game.current_l1_dist,
    }
    serialized = pickle.dumps(state_repr)
    return hashlib.sha256(serialized).hexdigest()

def hybrid_tree_search_top_w(config, model, root_game, w=5, d=5):
    """
    1. Selects top-w highest log-probability actions globally across the frontier,
    2. Expands them into new nodes,
    3. Backtracks using model value estimates from the leaves.

    Repeats up to depth `d` or until a correct solution is found.
    """
    model.eval()
    current_game = root_game
    visited = {}  # Memoization: (action_logits, value)

    for depth in range(d):
        # search frontier
        frontier = [TreeNode(current_game)]
        candidates = []

        # Collect all valid actions and log-probabilities
        for node in frontier:
            graph, target_vec, actions, mask = node.game.observe()
            key = game_key(node.game)

            if key in visited:
                action_logits, _ = visited[key]
            else:
                with torch.no_grad():
                    action_logits, value = model(
                        Batch.from_data_list([graph.to(node.game.device)]),
                        target_vec.to(node.game.device),
                        actions,
                        mask.to(node.game.device)
                    )
                visited[key] = (action_logits, value)

            log_probs = F.log_softmax(action_logits[0], dim=0)
            valid_indices = torch.where(mask[0])[0]

            for local_idx in valid_indices:
                action_idx = local_idx.item()
                log_prob_val = log_probs[action_idx].item()
                candidates.append((log_prob_val, node, action_idx))

        if not candidates:
            print("No valid actions found.")
            break

        # Select top-w candidates
        top_candidates = heapq.nlargest(w, candidates, key=lambda x: x[0])
        new_frontier = []

        for log_prob_val, parent_node, action_idx in top_candidates:
            new_game = copy.deepcopy(parent_node.game)
            new_game.take_action(action_idx)

            first_action = parent_node.first_action or action_idx
            trace = parent_node.actions_trace + [action_idx]

            new_node = TreeNode(
                game=new_game,
                logprob=parent_node.logprob + log_prob_val,
                first_action=first_action,
                actions_trace=trace
            )
            new_frontier.append(new_node)

        # Check if any node solves the target exactly
        for node in new_frontier:
            if node.game.is_done() and node.game.exprs:
                if sp.expand(node.game.exprs[-1] - root_game.target_sp) == 0:
                    return extract_steps(config, node.game), True

        # Evaluate and select best leaf by value
        best_leaf = None
        best_value = -float('inf')

        for node in new_frontier:
            graph, target_vec, actions, mask = node.game.observe()
            key = game_key(node.game)

            if key in visited:
                _, value = visited[key]
            else:
                with torch.no_grad():
                    _, value = model(
                        Batch.from_data_list([graph.to(node.game.device)]),
                        target_vec.to(node.game.device),
                        actions,
                        mask.to(node.game.device)
                    )
                visited[key] = (_, value)

            node.value = value.item()
            if node.value > best_value:
                best_leaf = node
                best_value = node.value

        if best_leaf is None:
            print("No best leaf node found.")
            break

        current_game = best_leaf.game

    return extract_steps(config, best_leaf.game), False

def extract_steps(config, game):
    """extract steps from a game state"""
    steps_structured = []
    for i, (op, n1, n2) in enumerate(game.actions_taken):
        expr = game.exprs[i]
        steps_structured.append({
            'step': i + config.n_variables,
            'operation': op,
            'node1': n1,
            'node2': n2,
            'result': expr
        })
    return steps_structured

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
    # max_vector_size = max(monomial_to_index.values()) + 1
    # Calculate max_vector_size based on n and d
    base = d + 1
    max_idx = 0
    for i in range(n):
        max_idx += d * (base ** i)
    max_vector_size = max_idx + 1

    # Load the trained model
    model = CircuitBuilder(config, max_vector_size).to(device)
    model_path = f"ppo_model_n{config.n_variables}_C{config.max_complexity}_curriculum.pt"
    # best_sup_model_path = f"best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt"

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded PPO model from {model_path}")

    #
    # print(f"Attempting to load PPO model: {model_path}")
    # try:
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     print(f"Successfully loaded PPO model from {model_path}")
    # except FileNotFoundError:
    #     print(f"PPO model not found. Trying best supervised model: {best_sup_model_path}")
    #     try:
    #          model.load_state_dict(torch.load(best_sup_model_path, map_location=device))
    #          print(f"Successfully loaded supervised model from {best_sup_model_path}")
    #     except FileNotFoundError:
    #          print(f"Error: No suitable model found ({model_path} or {best_sup_model_path}). Please train a model first.")
    #          return
    # except Exception as e:
    #     print(f"An error occurred loading the model: {e}")
    #     return

    # Create symbolic variables based on config
    vars_str = [f'x{i}' for i in range(n)]
    symbols_tuple = sp.symbols(vars_str)
    local_dict = {name: symbol for name, symbol in zip(vars_str, symbols_tuple)}

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
            # poly_str_processed = preprocess_poly_string(poly_str)
            print(f"Processed input: {poly_str}")

            poly_sp = parse_expr(poly_str,
                                 transformations=(standard_transformations + (implicit_multiplication_application,)),
                                 local_dict=local_dict)
            print(f"\nInput Target polynomial: {poly_sp}")
            poly_sp = sp.expand(poly_sp)
            print(f"\nExpanded Target polynomial: {poly_sp}")

            poly_vec = sympy_to_vector(poly_sp, monomial_to_index, n, config.mod, max_vector_size=max_vector_size).to(device)

            game = Game(poly_sp, poly_vec, config, index_to_monomial, monomial_to_index).to(device)

            print("Begin Tree Search")
            steps, success = hybrid_tree_search_top_w(config, model, game, w=10, d=15)

            # if game.exprs and sp.expand(game.exprs[-1] - poly_sp) == 0:
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
            # print("Please check your input format. Examples for N=2:")
            # print("  x0 + x1")
            # print("  x0**2 + 2*x0*x1 + x1**2")
            # print("  x0*(x1 + 1)")

if __name__ == "__main__":
    main()