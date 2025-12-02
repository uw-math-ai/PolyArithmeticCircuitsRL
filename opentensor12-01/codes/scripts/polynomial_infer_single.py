#!/usr/bin/env python3
"""
Inference script for single polynomial using curriculum-trained model.
Tests the model on a single polynomial (provided or randomly generated).
Shows the generated circuit and resulting polynomial.
"""

import argparse
import sys
import os
import torch
import numpy as np
import sympy as sp
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from codes.net.polynomial_net import PolynomialNet
from codes.mcts.polynomial_mcts import PolynomialMCTS
from codes.env.polynomial_environment import PolynomialEnvironment
from codes.scripts.polynomial_train_curriculum import sample_polynomial_by_complexity
from polynomial_env.actions import decode_action


def decode_action_readable(action_idx: int, max_nodes: int) -> tuple:
    """Convert action index to human-readable operation."""
    operation, node1_id, node2_id = decode_action(action_idx, max_nodes)
    return operation, node1_id, node2_id


def show_circuit_trace(env: PolynomialEnvironment, actions: list, max_nodes: int):
    """Show the circuit construction step by step."""
    print("\n" + "="*70)
    print("Circuit Construction Trace")
    print("="*70)
    
    # Reset and show initial state
    env.reset()
    print(f"\nInitial polynomials:")
    for i, poly in enumerate(env.poly_env.polynomials):
        print(f"  [{i}] = {poly}")
    
    # Execute each action and show the result
    for step_num, action in enumerate(actions):
        operation, node1_id, node2_id = decode_action_readable(action, max_nodes)
        
        # Get the polynomials before the operation
        poly1 = env.poly_env.polynomials[node1_id]
        poly2 = env.poly_env.polynomials[node2_id]
        
        # Apply action
        env.step(action)
        new_idx = len(env.poly_env.polynomials) - 1
        new_poly = env.poly_env.polynomials[new_idx]
        
        # Show operation
        op_symbol = "*" if operation == "multiply" else "+"
        print(f"\nStep {step_num + 1}: {operation.upper()} [{node1_id}] {op_symbol} [{node2_id}]")
        print(f"  {poly1} {op_symbol} {poly2}")
        print(f"  [{new_idx}] = {new_poly}")
        
        if env.is_terminate():
            if env.accumulate_reward > 0:
                print(f"\n✓ SUCCESS! Built target polynomial in {step_num + 1} steps")
                print(f"  Final reward: {env.accumulate_reward:.2f}")
            else:
                print(f"\n✗ FAILED. Reward: {env.accumulate_reward:.2f}")
            break
    
    print("="*70)


def infer_single_polynomial(
    model_path: str,
    target_poly_str: str = None,
    n_variables: int = 3,
    max_degree: int = 3,
    max_nodes: int = 10,
    complexity: int = None,
    mcts_simulations: int = 500,
    device: str = "cuda",
    show_trace: bool = True
):
    """
    Run inference on a single polynomial.
    
    Args:
        model_path: Path to trained model checkpoint
        target_poly_str: Polynomial string (e.g., "x0*x1 + x2") or None to generate random
        n_variables: Number of variables
        max_degree: Maximum polynomial degree
        max_nodes: Maximum nodes in circuit
        complexity: Complexity level for random generation (1-6)
        mcts_simulations: Number of MCTS simulations
        device: Device to use
        show_trace: Whether to show circuit construction trace
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse or generate target polynomial
    if target_poly_str:
        print(f"\n{'='*70}")
        print(f"Testing on provided polynomial: {target_poly_str}")
        print(f"{'='*70}")
        target_poly = sp.sympify(target_poly_str)
    else:
        if complexity is None:
            complexity = np.random.randint(1, 7)  # Random complexity 1-6
        
        target_poly = sample_polynomial_by_complexity(n_variables, max_degree, complexity)
        print(f"\n{'='*70}")
        print(f"Generated random polynomial (complexity {complexity}):")
        print(f"  {target_poly}")
        print(f"{'='*70}")
    
    # Expand and verify polynomial
    target_poly = sp.expand(target_poly)
    print(f"\nExpanded target: {target_poly}")
    
    # Check degree
    symbols = sp.symbols(f"x0:{n_variables}")
    try:
        poly_obj = target_poly.as_poly(*symbols)
        if poly_obj:
            degree = poly_obj.total_degree()
            print(f"Polynomial degree: {degree}")
            if degree > max_degree:
                print(f"WARNING: Degree {degree} exceeds max_degree {max_degree}!")
    except:
        pass
    
    # Setup environment
    env_cfg = {
        'n_variables': n_variables,
        'max_degree': max_degree,
        'max_nodes': max_nodes,
        'T': 1,
        'step_penalty': -0.1,
        'success_reward': 10.0,
        'failure_penalty': -5.0,
    }
    
    env = PolynomialEnvironment(target_poly_expr=target_poly, **env_cfg)
    
    # Setup network
    action_dim = (max_nodes * (max_nodes + 1)) // 2 * 2
    hidden_dim = 256
    s_size = max_degree + 1
    
    net = PolynomialNet(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        T=1,
        s_size=s_size,
        device=str(device)
    ).to(device)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        net.load_state_dict(checkpoint)
    net.eval()
    
    # Setup MCTS
    mcts = PolynomialMCTS(
        net=net,
        simulations=mcts_simulations,
        c_puct=1.5,
        device=device
    )
    
    # Run inference
    print(f"\nRunning MCTS with {mcts_simulations} simulations...")
    env.reset()
    actions_taken = []
    
    step_count = 0
    max_steps = 50
    
    while not env.is_terminate() and step_count < max_steps:
        tensors, scalars, mask = env.get_network_input()
        action, pi = mcts.run(env)
        actions_taken.append(action)
        env.step(action)
        step_count += 1
    
    # Show results
    print(f"\n{'='*70}")
    print("Inference Results")
    print(f"{'='*70}")
    print(f"Target polynomial: {target_poly}")
    print(f"Steps taken: {step_count}")
    print(f"Terminated: {env.is_terminate()}")
    print(f"Final reward: {env.accumulate_reward:.2f}")
    
    if env.accumulate_reward > 0:
        print(f"\n✓ SUCCESS! Model built the target polynomial!")
    else:
        print(f"\n✗ FAILED. Model could not build the target polynomial.")
        if step_count >= max_steps:
            print(f"   Reason: Reached maximum steps ({max_steps})")
    
    # Show final polynomial built
    if len(env.poly_env.polynomials) > n_variables:
        final_poly = env.poly_env.polynomials[-1]
        print(f"\nFinal polynomial built: {final_poly}")
        print(f"Target was:             {target_poly}")
        
        # Check if they match
        diff = sp.expand(final_poly - target_poly)
        if diff == 0:
            print("\n✓ Polynomials match exactly!")
        else:
            print(f"\n✗ Difference: {diff}")
    
    # Show circuit trace if requested
    if show_trace and len(actions_taken) > 0:
        show_circuit_trace(env, actions_taken, max_nodes)
    
    return env.accumulate_reward > 0


def main():
    parser = argparse.ArgumentParser(
        description="Test curriculum-trained model on a single polynomial"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="codes/scripts/runs/polynomial_net_complexity.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--polynomial",
        type=str,
        default=None,
        help="Target polynomial string (e.g., 'x0*x1 + x2'). If not provided, generates random."
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=None,
        help="Complexity level (1-6) for random generation. Random if not specified."
    )
    parser.add_argument(
        "--n_variables",
        type=int,
        default=3,
        help="Number of variables"
    )
    parser.add_argument(
        "--max_degree",
        type=int,
        default=3,
        help="Maximum polynomial degree"
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=10,
        help="Maximum nodes in circuit"
    )
    parser.add_argument(
        "--mcts_simulations",
        type=int,
        default=500,
        help="Number of MCTS simulations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--no_trace",
        action="store_true",
        help="Don't show circuit construction trace"
    )
    
    args = parser.parse_args()
    
    success = infer_single_polynomial(
        model_path=args.model_path,
        target_poly_str=args.polynomial,
        n_variables=args.n_variables,
        max_degree=args.max_degree,
        max_nodes=args.max_nodes,
        complexity=args.complexity,
        mcts_simulations=args.mcts_simulations,
        device=args.device,
        show_trace=not args.no_trace
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
