"""
Polynomial circuit inference script.

Loads a trained MCTS model and attempts to build circuits for input polynomials.
"""
import argparse
import sympy as sp
import torch

from codes.env.polynomial_environment import PolynomialEnvironment
from codes.mcts.polynomial_mcts import PolynomialMCTS
from codes.net.polynomial_net import PolynomialNet
from codes.utils.polynomial_random import random_polynomial
import random


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


def decode_and_print_circuit(env: PolynomialEnvironment, actions: list):
    """
    Decode actions into human-readable circuit steps.
    """
    from polynomial_env.actions import decode_action
    
    print("\nCircuit construction steps:")
    for i, action_idx in enumerate(actions):
        op, idx_i, idx_j = decode_action(action_idx, env.config.max_nodes)
        op_name = "ADD" if op == "add" else "MUL"
        node_result = len(env.poly_env.polynomials) - (len(actions) - i)
        print(f"  Step {i+1}: {op_name}(node_{idx_i}, node_{idx_j}) -> node_{node_result}")
    
    print(f"\nFinal circuit has {len(env.poly_env.polynomials)} nodes")
    print(f"Output polynomial: {env.poly_env.polynomials[-1]}")


def sample_polynomials_via_circuits(
    n_samples: int,
    n_variables: int,
    max_degree: int,
    max_nodes: int,
    max_steps: int = 4,
):
    """Generate unique polynomials using random add/mul circuits within a few steps."""
    samples = []
    seen = set()
    base_nodes = n_variables + 1  # vars + constant 1
    symbols = [sp.Symbol(f"x{i}") for i in range(n_variables)]
    constant_one = sp.Integer(1)

    max_steps = max(0, min(max_steps, max_nodes - base_nodes))

    while len(samples) < n_samples:
        nodes = symbols + [constant_one]
        steps = random.randint(1, max_steps) if max_steps > 0 else 0

        for _ in range(steps):
            i = random.randrange(len(nodes))
            j = random.randrange(len(nodes))
            op = random.choice(["add", "multiply"])

            if op == "add":
                new_poly = sp.expand(sp.Add(nodes[i], nodes[j]))
            else:
                new_poly = sp.expand(sp.Mul(nodes[i], nodes[j]))

            poly_terms = sp.Poly(new_poly, *symbols)
            if any(any(exp > max_degree for exp in monom) for monom, _ in poly_terms.terms()):
                continue  # skip this operation and try another

            nodes.append(new_poly)
            if len(nodes) >= max_nodes:
                break

        candidate = sp.expand(nodes[-1])
        key = sp.simplify(candidate)
        key_str = str(key)
        if key_str not in seen:
            seen.add(key_str)
            samples.append(candidate)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Polynomial circuit inference")
    parser.add_argument("--model_path", type=str, default="src/OpenTensor/codes/scripts/runs/polynomial_net_parallel.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--polynomial", type=str, default=None,
                        help="Polynomial expression (e.g., 'x0**2 + x0*x1 + 1'). If not provided, generates random polynomial.")
    parser.add_argument("--n_variables", type=int, default=2,
                        help="Number of variables")
    parser.add_argument("--max_degree", type=int, default=3,
                        help="Maximum degree")
    parser.add_argument("--max_nodes", type=int, default=8,
                        help="Maximum circuit nodes")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension of network")
    parser.add_argument("--method", type=str, default="mcts", choices=["greedy", "mcts"],
                        help="Inference method: greedy or mcts")
    parser.add_argument("--mcts_simulations", type=int, default=128,
                        help="Number of MCTS simulations per step")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    parser.add_argument("--num_tests", type=int, default=1,
                        help="Number of random polynomials to test (only used if --polynomial is not provided)")
    
    args = parser.parse_args()
    
    # Load model once
    print("=" * 80)
    print("POLYNOMIAL CIRCUIT INFERENCE")
    print("=" * 80)
    print(f"Variables: {args.n_variables}, Max degree: {args.max_degree}, Max nodes: {args.max_nodes}")
    print(f"Method: {args.method.upper()}")
    print(f"Device: {args.device}")
    print(f"\nLoading model from: {args.model_path}")
    
    # Create a dummy environment to get action_dim
    dummy_poly = sp.Symbol('x0')
    dummy_env = PolynomialEnvironment(
        target_poly_expr=dummy_poly,
        n_variables=args.n_variables,
        max_degree=args.max_degree,
        max_nodes=args.max_nodes,
        T=1,
    )
    action_dim = dummy_env.max_actions
    s_size = args.max_degree + 1
    
    net = PolynomialNet(
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        T=1,
        s_size=s_size,
        device=args.device
    ).to(args.device)
    
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=True)
    net.load_state_dict(checkpoint)
    net.eval()
    print("Model loaded successfully!\n")
    
    # Determine test polynomials
    if args.polynomial:
        # Single polynomial test
        symbols_dict = {f"x{i}": sp.Symbol(f"x{i}") for i in range(args.n_variables)}
        poly_expr = sp.sympify(args.polynomial, locals=symbols_dict)
        poly_expr = sp.expand(poly_expr)
        test_polynomials = [poly_expr]
    else:
        test_polynomials = sample_polynomials_via_circuits(
            n_samples=args.num_tests,
            n_variables=args.n_variables,
            max_degree=args.max_degree,
            max_nodes=args.max_nodes,
            max_steps=4,
        )
    
    # Run tests
    successes = 0
    for test_idx, poly_expr in enumerate(test_polynomials, 1):
        if len(test_polynomials) > 1:
            print("=" * 80)
            print(f"TEST {test_idx}/{len(test_polynomials)}")
            print("=" * 80)
        
        print(f"\nTarget polynomial: {poly_expr}")
        
        # Create environment
        env = PolynomialEnvironment(
            target_poly_expr=poly_expr,
            n_variables=args.n_variables,
            max_degree=args.max_degree,
            max_nodes=args.max_nodes,
            T=1,
        )
        
        # Run inference
        print(f"Building circuit using {args.method} method...")
        print("-" * 80)
        
        if args.method == "greedy":
            actions, reward, success = build_circuit_greedy(env, net, max_steps=args.max_nodes)
        else:  # mcts
            actions, reward, success = build_circuit_mcts(env, net, mcts_simulations=args.mcts_simulations)
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Success: {'✓ YES' if success else '✗ NO'}")
        print(f"Total reward: {reward:.2f}")
        print(f"Number of steps: {len(actions)}")
        
        if success:
            successes += 1
            built_poly = env.poly_env.polynomials[-1]
            print(f"Built polynomial: {built_poly}")
            print(f"Residual: 0")
            if len(test_polynomials) == 1:
                print(f"Actions taken: {actions}")
                decode_and_print_circuit(env, actions)
                print("\n✓ Successfully built circuit for target polynomial!")
        else:
            built_poly = env.poly_env.polynomials[-1]
            residual = sp.expand(env.poly_env.target_poly_expr - built_poly)
            print(f"Built polynomial: {built_poly}")
            print(f"Residual: {residual}")
            if len(test_polynomials) == 1:
                print(f"Actions taken: {actions}")
                print("\n✗ Failed to build exact circuit")
        
        if len(test_polynomials) > 1:
            print("")
    
    # Summary for batch tests
    if len(test_polynomials) > 1:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total tests: {len(test_polynomials)}")
        print(f"Successes: {successes}")
        print(f"Failures: {len(test_polynomials) - successes}")
        print(f"Success rate: {100 * successes / len(test_polynomials):.1f}%")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
