"""
Inference script for curriculum-trained polynomial circuit model.

Tests the model on polynomials of varying complexity (C1-C6).
"""
import argparse
import sys
import sympy as sp
import torch
import numpy as np

from codes.env.polynomial_environment import PolynomialEnvironment
from codes.mcts.polynomial_mcts import PolynomialMCTS
from codes.net.polynomial_net import PolynomialNet
from codes.scripts.polynomial_train_curriculum import sample_polynomial_by_complexity


def build_circuit_mcts(env: PolynomialEnvironment, net: PolynomialNet, mcts_simulations: int = 128):
    """
    MCTS-based policy: use tree search to find actions.
    Returns the sequence of actions and final reward.
    """
    net.eval()
    net.set_mode("infer")
    
    mcts = PolynomialMCTS(
        net=net,
        simulations=mcts_simulations,
        c_puct=1.0,
        device=net.device,
        virtual_loss=1.0
    )
    
    actions_taken = []
    
    while not env.is_terminate():
        action, policy = mcts.run(env)
        actions_taken.append(action)
        env.step(action)
    
    return actions_taken, env.accumulate_reward, env.poly_env.is_success()


def build_circuit_greedy(env: PolynomialEnvironment, net: PolynomialNet, max_steps: int = 20):
    """
    Greedy policy: at each step, pick the highest probability valid action.
    Returns the sequence of actions and final reward.
    """
    net.eval()
    net.set_mode("infer")
    
    actions_taken = []
    step = 0
    
    while not env.is_terminate() and step < max_steps:
        tensors, scalars, mask = env.get_network_input()
        
        # Convert to tensors
        tensors_t = torch.from_numpy(tensors).unsqueeze(0).float().to(net.device)
        scalars_t = torch.from_numpy(scalars).unsqueeze(0).float().to(net.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).bool().to(net.device)
        
        with torch.no_grad():
            action_idx, probs, value = net(tensors_t, scalars_t, mask_t)
            action_idx = action_idx.item()
        
        actions_taken.append(action_idx)
        env.step(action_idx)
        step += 1
    
    return actions_taken, env.accumulate_reward, env.poly_env.is_success()


def test_complexity_level(net, device, complexity, n_tests=10, n_variables=3, max_degree=3, 
                          max_nodes=10, mcts_simulations=256, use_mcts=True):
    """Test model on polynomials of a specific complexity level."""
    successes = 0
    total_steps = 0
    results = []
    
    for i in range(n_tests):
        # Generate random polynomial at this complexity
        target = sample_polynomial_by_complexity(n_variables, max_degree, complexity)
        target_expanded = sp.expand(target)
        
        # Create environment
        env_cfg = {
            'n_variables': n_variables,
            'max_degree': max_degree,
            'max_nodes': max_nodes,
            'T': 1,
            'step_penalty': -0.1,
            'success_reward': 10.0,
            'failure_penalty': -5.0,
        }
        env = PolynomialEnvironment(target_poly_expr=target_expanded, **env_cfg)
        
        # Build circuit
        if use_mcts:
            actions, reward, success = build_circuit_mcts(env, net, mcts_simulations)
        else:
            actions, reward, success = build_circuit_greedy(env, net, max_steps=20)
        
        if success:
            successes += 1
        total_steps += len(actions)
        
        results.append({
            'target': str(target_expanded),
            'success': success,
            'steps': len(actions),
            'reward': reward
        })
    
    success_rate = (successes / n_tests) * 100
    avg_steps = total_steps / n_tests
    
    return success_rate, avg_steps, results


def main(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    
    # Initialize network
    env_cfg = {
        'n_variables': args.n_variables,
        'max_degree': args.max_degree,
        'max_nodes': args.max_nodes,
        'T': 1,
    }
    
    # Calculate action space size (same as PolynomialEnvironment.max_actions)
    action_dim = (args.max_nodes * (args.max_nodes + 1)) // 2 * 2
    s_size = args.max_degree + 1
    
    net = PolynomialNet(
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        T=1,
        s_size=s_size,
        device=device
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()
    print("✓ Model loaded successfully")
    
    # Test on each complexity level
    print(f"\n{'='*70}")
    print(f"Testing Curriculum-Trained Model")
    print(f"{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Method: {'MCTS' if args.use_mcts else 'Greedy'} ({'256 simulations' if args.use_mcts else ''})")
    print(f"Tests per complexity: {args.n_tests}")
    print(f"Variables: {args.n_variables} (x0 to x{args.n_variables-1})")
    print(f"{'='*70}\n")
    
    overall_results = []
    
    for complexity in range(args.complexity_start, args.complexity_end + 1):
        print(f"Testing Complexity {complexity} ({complexity} operations required):")
        
        success_rate, avg_steps, results = test_complexity_level(
            net, device, complexity, 
            n_tests=args.n_tests,
            n_variables=args.n_variables,
            max_degree=args.max_degree,
            max_nodes=args.max_nodes,
            mcts_simulations=args.mcts_simulations,
            use_mcts=args.use_mcts
        )
        
        print(f"  Success Rate: {success_rate:.1f}% ({int(success_rate * args.n_tests / 100)}/{args.n_tests})")
        print(f"  Average Steps: {avg_steps:.2f}")
        
        # Show some examples
        if args.show_examples:
            print(f"  Examples:")
            for j, result in enumerate(results[:3]):
                status = "✓" if result['success'] else "✗"
                print(f"    {status} {result['target']} - {result['steps']} steps, reward={result['reward']:.1f}")
        
        print()
        
        overall_results.append({
            'complexity': complexity,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'results': results
        })
    
    # Summary
    print(f"{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    total_tests = len(overall_results) * args.n_tests
    total_successes = sum(int(r['success_rate'] * args.n_tests / 100) for r in overall_results)
    overall_success_rate = (total_successes / total_tests) * 100
    
    print(f"Overall Success Rate: {overall_success_rate:.1f}% ({total_successes}/{total_tests})")
    print()
    print("By Complexity:")
    for r in overall_results:
        print(f"  C{r['complexity']}: {r['success_rate']:5.1f}% success, {r['avg_steps']:4.2f} avg steps")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for curriculum-trained polynomial model")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, 
                       default="codes/scripts/runs/polynomial_net_complexity.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    
    # Environment parameters
    parser.add_argument("--n_variables", type=int, default=3, help="Number of variables")
    parser.add_argument("--max_degree", type=int, default=3, help="Maximum polynomial degree")
    parser.add_argument("--max_nodes", type=int, default=10, help="Maximum circuit nodes")
    
    # Testing parameters
    parser.add_argument("--complexity_start", type=int, default=1, help="Starting complexity")
    parser.add_argument("--complexity_end", type=int, default=6, help="Ending complexity")
    parser.add_argument("--n_tests", type=int, default=20, help="Number of tests per complexity")
    parser.add_argument("--mcts_simulations", type=int, default=256, help="MCTS simulations")
    parser.add_argument("--use_mcts", action="store_true", default=True, help="Use MCTS (vs greedy)")
    parser.add_argument("--greedy", dest='use_mcts', action="store_false", help="Use greedy policy")
    parser.add_argument("--show_examples", action="store_true", help="Show example results")
    
    args = parser.parse_args()
    main(args)
