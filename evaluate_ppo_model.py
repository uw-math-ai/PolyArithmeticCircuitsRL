#!/usr/bin/env python3
"""
Evaluation script for PPO model testing on 10 random polynomials with complexity 6.
This script generates 10 random polynomial targets and evaluates how well the trained
PPO model can construct circuits to match them.
"""

import json
import sys
from pathlib import Path
import torch
import sympy
import numpy as np

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR / "src"
PPO_DIR = SRC_ROOT / "PPO RL"

for path in (CURRENT_DIR, SRC_ROOT, PPO_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from generator import generate_random_circuit, get_symbols
from State import Game
from PPO import CircuitBuilder, Config, build_compact_encoder, encode_actions_with_compact_encoder
from encoders.compact_encoder import CompactOneHotGraphEncoder
from utils import decode_action

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class EvaluationConfig(Config):
    """Configuration for evaluation, inheriting from PPO Config."""
    def __init__(self):
        super().__init__()
        self.n_variables = 3
        self.max_complexity = 5 #was 6


# def generate_target_polynomials(num_polynomials=10, n_variables=3, complexity=6):
#     """Generate random target polynomials to test on."""
#     polynomials = []
#     circuits = []
#     symbols = get_symbols(n_variables)
#     seen_keys = set()
#     
#     attempts = 0
#     max_attempts = num_polynomials * 20
#     
#     print(f"Generating {num_polynomials} random target polynomials...")
#     
#     while len(polynomials) < num_polynomials and attempts < max_attempts:
#         attempts += 1
#         actions, polys = generate_random_circuit(n_variables, complexity, mod=50)
#         
#         if not polys:
#             continue
#         
#         final_poly = polys[-1]
#         
#         # Create canonical key to avoid duplicates
#         expanded = sympy.expand(final_poly)
#         try:
#             poly_obj = sympy.Poly(expanded, symbols, domain='QQ')
#             key = sympy.srepr(poly_obj.as_expr())
#         except:
#             continue
#         
#         if key not in seen_keys:
#             polynomials.append(final_poly)
#             circuits.append(actions)
#             seen_keys.add(key)
#     
#     print(f"Generated {len(polynomials)} unique polynomials after {attempts} attempts\n")
#     return polynomials, circuits


def generate_target_polynomials(n: int, C: int, mod: int = 2, num_polynomials: int = 10):
    """
    Generate target polynomials using generate_random_circuit.
    
    Args:
        n: Number of variables
        C: Complexity level
        mod: Modulo value for coefficients
        num_polynomials: Number of unique polynomials to generate
    
    Returns:
        polynomials: List of target polynomial expressions
        circuits: List of corresponding circuit actions
    """
    polynomials = []
    circuits = []
    seen_keys = set()
    
    print(f"Generating {num_polynomials} random target polynomials with n={n}, C={C}, mod={mod}...")
    
    attempts = 0
    max_attempts = num_polynomials * 50  # Allow many attempts to find unique polynomials
    symbols = get_symbols(n)
    
    while len(polynomials) < num_polynomials and attempts < max_attempts:
        attempts += 1
        
        actions, polys = generate_random_circuit(n, C, mod=mod)
        
        if not polys:
            continue
        
        final_poly = polys[-1]
        
        # Create canonical key to avoid duplicates
        try:
            expanded = sympy.expand(final_poly)
            poly_obj = sympy.Poly(expanded, symbols, domain='QQ')
            key = sympy.srepr(poly_obj.as_expr())
        except:
            continue
        
        if key not in seen_keys:
            polynomials.append(final_poly)
            circuits.append(actions)
            seen_keys.add(key)
    
    print(f"Generated {len(polynomials)} unique polynomial(s) after {attempts} attempts\n")
    return polynomials, circuits


def evaluate_on_polynomial(model, target_poly, reference_circuit, config, max_steps=10, temperature=1.0):
    """
    Evaluate the model on a single target polynomial.
    
    Args:
        model: The trained PPO model
        target_poly: The target polynomial (sympy expression)
        reference_circuit: The reference circuit actions that generate the target
        config: Configuration object
        max_steps: Maximum steps to take
        temperature: Sampling temperature
    
    Returns:
        success (bool): Whether the model found the target polynomial
        steps (int): Number of steps taken
        final_poly (sympy.Expr): The final polynomial generated
        actions (list): List of actions taken
    """
    
    # Encode the target polynomial using the reference circuit
    target_encoding = encode_actions_with_compact_encoder(reference_circuit, config)
    target_encoding = target_encoding.unsqueeze(0)  # Add batch dimension
    
    # Create the game/environment
    game = Game(target_poly, target_encoding, config)
    
    actions_taken = []
    
    # Run the game until it's done or max_steps reached
    for step in range(max_steps):
        # Get current state
        state = game.observe()
        
        # Get action from model
        action, log_prob, entropy, value = model.get_action_and_value(state, temperature=temperature)
        
        if action is None:
            print(f"  Warning: Model returned None action at step {step}")
            break
        
        actions_taken.append(action)
        
        # Take the action
        game.take_action(action)
        
        # Check if done
        if game.is_done():
            break
    
    # Check success
    success = False
    final_poly = None
    if game.polynomials:
        final_poly = game.polynomials[-1]
        if sympy.expand(final_poly - target_poly) == 0:
            success = True
    
    return success, len(actions_taken), final_poly, actions_taken


def main():
    """Main evaluation function."""
    
    # Load configuration
    config = EvaluationConfig()
    print(f"Configuration:")
    print(f"  Variables: {config.n_variables}")
    print(f"  Max Complexity: {config.max_complexity}")
    print(f"  Mod: {config.mod}\n")
    
    # Generate target polynomials
    target_polynomials, reference_circuits = generate_target_polynomials(
        n=config.n_variables,
        C=config.max_complexity,
        mod=config.mod,
        num_polynomials=1000
    )
    
    # Load the trained model
    model_path = Path("/home/ec2-user/DESKTOP/Naomi/PolyArithmeticCircuitsRL/src/PPO RL/Trained Model/best_supervised_model_n3_C5.pt")
        #/home/ec2-user/DESKTOP/Naomi/PolyArithmeticCircuitsRL/src/PPO RL/Trained Model/ppo_model_n3_C5_curriculum.pt")
        #"/home/ec2-user/DESKTOP/Naomi/PolyArithmeticCircuitsRL/ppo_model_n3_C6_curriculum.pt")
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    state_encoding_size = build_compact_encoder(config).size
    model = CircuitBuilder(config, state_encoding_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    
    # Evaluate on each polynomial
    results = []
    successes = 0
    total_steps = 0
    
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for i, (target_poly, ref_circuit) in enumerate(zip(target_polynomials, reference_circuits)):
        print(f"\nTest {i+1}/10:")
        print(f"  Target polynomial: {target_poly}")
        print(f"  Reference circuit length: {len(ref_circuit)}")
        
        # Evaluate
        success, steps, final_poly, actions = evaluate_on_polynomial(
            model, target_poly, ref_circuit, config, max_steps=config.max_complexity + 5, temperature=1.0
        )
        
        print(f"  Success: {'✓ YES' if success else '✗ NO'}")
        print(f"  Steps taken: {steps}")
        if final_poly:
            print(f"  Final polynomial: {final_poly}")
        
        results.append({
            "test_id": i + 1,
            "target_polynomial": str(target_poly),
            "success": success,
            "steps": steps,
            "final_polynomial": str(final_poly) if final_poly else None,
            "reference_circuit_length": len(ref_circuit)
        })
        
        if success:
            successes += 1
        total_steps += steps
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(target_polynomials)}")
    print(f"Successful: {successes}/{len(target_polynomials)} ({100*successes/len(target_polynomials):.1f}%)")
    print(f"Average steps: {total_steps/len(target_polynomials):.1f}")
    print()
    
    # Save results to JSON
    output_file = CURRENT_DIR / "evaluation_results_C6.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "n_variables": config.n_variables,
                "max_complexity": config.max_complexity,
                "mod": config.mod
            },
            "summary": {
                "total_tests": len(target_polynomials),
                "successes": successes,
                "success_rate": successes / len(target_polynomials),
                "average_steps": total_steps / len(target_polynomials)
            },
            "results": results
        }, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}\n")
    
    return successes == len(target_polynomials)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
