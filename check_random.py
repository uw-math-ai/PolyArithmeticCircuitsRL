from src.config import Config
from src.game_board.generator import generate_random_circuit

def main():
    # 1. Initialize the default configuration
    # This sets up the environment (e.g., 2 variables, mod 5, etc.)
    config = Config()
    
    # 2. Set the desired complexity
    complexity = 10
    
    print(f"Generating a random circuit with Complexity {complexity}...")
    
    # 3. Generate the circuit
    target_poly, actions = generate_random_circuit(config, complexity)
    
    # 4. Display the results
    print("\n" + "="*40)
    print("FINAL TARGET POLYNOMIAL:")
    print("="*40)
    # The FastPoly object has a __str__ or __repr__ that will display it
    print(target_poly)
    
    print("\n" + "="*40)
    print("ACTION SEQUENCE:")
    print("="*40)
    
    # Print the base nodes to make the sequence easier to follow
    print("Base Nodes:")
    for i in range(config.n_variables):
        print(f"  Node {i}: x_{i}")
    print(f"  Node {config.n_variables}: 1 (Constant)")
    print("-" * 40)
    
    # Walk through the actions
    base_nodes_count = config.n_variables + 1
    for step_idx, action in enumerate(actions):
        op_code, i, j = action
        op_str = "ADD" if op_code == 0 else "MUL"
        new_node_idx = base_nodes_count + step_idx
        
        print(f"Step {step_idx + 1} -> Created Node {new_node_idx}: "
              f"{op_str} (Node {i}, Node {j})")

if __name__ == "__main__":
    main()