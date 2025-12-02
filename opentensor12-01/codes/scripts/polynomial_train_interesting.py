"""
Polynomial circuit training with curriculum learning, focusing on 'interesting' algebraic patterns.

Strategy:
- Start from existing checkpoint trained on random polynomials
- Mix random construction with specific algebraic patterns (factorizable forms)
- Patterns include: Square of Sums, Distributive Laws, Product of Sums
- Continue adaptive curriculum
"""
import argparse
import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import tempfile
from pathlib import Path
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from codes.env.polynomial_environment import PolynomialEnvironment
from codes.mcts.polynomial_mcts import PolynomialMCTS
from codes.net.polynomial_net import PolynomialNet
from codes.utils.polynomial_random import random_polynomial
from polynomial_env.actions import encode_action


class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensors, scalars, mask, pi, reward = self.samples[idx]
        return (
            torch.from_numpy(tensors).float(),
            torch.from_numpy(scalars).float(),
            torch.from_numpy(mask).bool(),
            torch.from_numpy(pi).float(),
            torch.tensor(reward, dtype=torch.float32),
        )


def sample_interesting_polynomial(n_variables: int, max_degree: int, complexity: int):
    """
    Generate 'interesting' polynomials based on algebraic identities.
    These often have efficient circuit representations (factored forms) 
    but complex expanded forms.
    """
    symbols = sp.symbols(f"x0:{n_variables}")
    
    # Helper to get random variables
    def get_vars(n=2):
        return np.random.choice(symbols, size=n, replace=True)

    choices = []
    
    # Complexity 2: 2 operations
    # (A+B)**2 -> A**2 + 2AB + B**2 (Naive: ~5 ops, Efficient: 2 ops)
    # A*(B+C)  -> AB + AC           (Naive: 3 ops, Efficient: 2 ops)
    if complexity == 2:
        # (A+B)**2
        if max_degree >= 2:
            a, b = get_vars(2)
            choices.append((a + b)**2)
        
        # A*(B+C)
        a, b, c = get_vars(3)
        choices.append(a * (b + c))

    # Complexity 3: 3 operations
    # (A+B+C)**2 -> (Naive: many, Efficient: 3 ops: A+B, +C, **2)
    # (A+B)*(C+D) -> AC+AD+BC+BD (Naive: ~7 ops, Efficient: 3 ops)
    # A*(B+C)**2
    elif complexity == 3:
        # (A+B+C)**2
        if max_degree >= 2:
            a, b, c = get_vars(3)
            choices.append((a + b + c)**2)
            
        # (A+B)*(C+D)
        if max_degree >= 2:
            a, b, c, d = get_vars(4)
            choices.append((a + b) * (c + d))
            
        # A*(B+C)**2
        if max_degree >= 3:
            a, b, c = get_vars(3)
            choices.append(a * (b + c)**2)

    # Complexity 4: 4 operations
    # (A+B+C+D)**2
    # (A+B)*(C+D+E)
    # (A+B)**2 + (C+D)**2
    elif complexity == 4:
        # (A+B+C+D)**2
        if max_degree >= 2:
            a, b, c, d = get_vars(4)
            choices.append((a + b + c + d)**2)
            
        # (A+B)*(C+D+E)
        if max_degree >= 2:
            a, b, c, d, e = get_vars(5)
            choices.append((a + b) * (c + d + e))
            
        # (A+B)**2 + (C+D)**2
        if max_degree >= 2:
            a, b, c, d = get_vars(4)
            choices.append((a + b)**2 + (c + d)**2)

    # Complexity 5+: More complex combinations
    elif complexity >= 5:
        # (A+B+C)*(D+E+F)
        if max_degree >= 2:
            a, b, c, d, e, f = get_vars(6)
            choices.append((a + b + c) * (d + e + f))
            
        # (A+B)**3 (if degree allows)
        if max_degree >= 3:
            a, b = get_vars(2)
            choices.append((a + b)**3)
            
        # (A+B)**2 * (C+D)
        if max_degree >= 3:
            a, b, c, d = get_vars(4)
            choices.append((a + b)**2 * (c + d))

    # If we have choices, pick one
    if choices:
        poly = np.random.choice(choices)
        return sp.expand(poly)
    
    # Fallback to random construction if no patterns match criteria
    return None


def sample_polynomial_mixed(n_variables: int, max_degree: int, complexity: int, max_attempts: int = 10):
    """
    Sample polynomial using a mix of random construction and interesting patterns.
    """
    # 50% chance to try an interesting pattern first
    if np.random.random() < 0.5:
        poly = sample_interesting_polynomial(n_variables, max_degree, complexity)
        if poly is not None:
            # Verify degree
            try:
                symbols = sp.symbols(f"x0:{n_variables}")
                poly_obj = poly.as_poly(*symbols)
                if poly_obj and poly_obj.total_degree() <= max_degree:
                    return poly
            except:
                pass

    # Fallback to random construction (from original script)
    symbols = sp.symbols(f"x0:{n_variables}")
    
    for attempt in range(max_attempts):
        available = list(symbols)
        for i in range(n_variables):
            available.append(symbols[i]**2)
        
        current_poly = None
        for op_idx in range(complexity):
            op_type = np.random.choice(['add', 'multiply'], p=[0.6, 0.4])
            
            if op_type == 'add':
                if len(available) >= 2:
                    idx1, idx2 = np.random.choice(len(available), size=2, replace=False)
                    poly1, poly2 = available[idx1], available[idx2]
                    coeff = np.random.choice([1, 1, 1, 2])
                    new_poly = sp.expand(coeff * poly1 + poly2)
                    current_poly = new_poly
                    available.append(new_poly)
                else:
                    current_poly = available[0] if available else symbols[0]
            else:
                if len(available) >= 2:
                    idx1, idx2 = np.random.choice(len(available), size=2, replace=False)
                    poly1, poly2 = available[idx1], available[idx2]
                    new_poly = sp.expand(poly1 * poly2)
                    try:
                        poly_obj = new_poly.as_poly(*symbols)
                        if poly_obj and poly_obj.total_degree() <= max_degree:
                            current_poly = new_poly
                            available.append(new_poly)
                        else:
                            new_poly = sp.expand(poly1 + poly2)
                            current_poly = new_poly
                            available.append(new_poly)
                    except:
                        new_poly = sp.expand(poly1 + poly2)
                        current_poly = new_poly
                        available.append(new_poly)
                else:
                    current_poly = available[0] if available else symbols[0]
        
        result = current_poly if current_poly is not None else symbols[0]
        result = sp.expand(result)
        
        if result != 0 and not result.is_number:
            try:
                poly_obj = result.as_poly(*symbols)
                if poly_obj and poly_obj.total_degree() <= max_degree:
                    return result
            except:
                pass
    
    return sp.expand(symbols[0] + symbols[1 % n_variables])


def generate_episode_complexity(env_cfg, net, mcts, device, complexity: int, seen_polynomials=None):
    """
    Run one self-play episode with target polynomial of given complexity.
    """
    max_attempts = 20
    target = None
    for attempt in range(max_attempts):
        # Use the mixed sampling strategy
        target = sample_polynomial_mixed(
            env_cfg["n_variables"], 
            env_cfg["max_degree"],
            complexity=complexity
        )
        
        target_str = str(sp.expand(target))
        if seen_polynomials is None or target_str not in seen_polynomials:
            if seen_polynomials is not None:
                seen_polynomials.add(target_str)
            break
    
    if target is None:
        # Fallback if something went wrong
        target = sp.Symbol("x0") + sp.Symbol("x1")

    env = PolynomialEnvironment(target_poly_expr=target, **env_cfg)
    traj = []
    while not env.is_terminate():
        tensors, scalars, mask = env.get_network_input()
        action, pi = mcts.run(env)
        env.step(action)
        traj.append((tensors, scalars, mask, pi))
    reward = env.accumulate_reward
    samples = []
    for tensors, scalars, mask, pi in traj:
        samples.append((tensors, scalars, mask, pi, reward))
    return samples, target, complexity


def generate_episode_worker_file(args_tuple):
    """Worker function that saves results to file instead of returning via pipe."""
    (env_cfg, net_state_path, mcts_simulations, c_puct, virtual_loss, device_str, 
     action_dim, hidden_dim, T, s_size, episode_id, output_dir, complexity) = args_tuple
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    net = PolynomialNet(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        T=T,
        s_size=s_size,
        device=device,
    )
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        net.load_state_dict(torch.load(net_state_path, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    
    mcts = PolynomialMCTS(
        net=net, 
        simulations=mcts_simulations, 
        c_puct=c_puct, 
        device=device,
        virtual_loss=virtual_loss
    )
    
    ep_samples, target, actual_complexity = generate_episode_complexity(env_cfg, net, mcts, device, complexity)
    
    output_file = os.path.join(output_dir, f"episode_{episode_id}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump((ep_samples, target, actual_complexity), f)
    
    return output_file


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    wandb.init(
        project="polynomial-mcts-training",
        entity="zengrf-university-of-washington",
        name=f"mcts_interesting_v{args.n_variables}_d{args.max_degree}",
        config={
            "n_variables": args.n_variables,
            "max_degree": args.max_degree,
            "max_nodes": args.max_nodes,
            "mcts_simulations": args.mcts_simulations,
            "episodes_per_epoch": args.episodes_per_epoch,
            "epochs": args.epochs,
            "lr": args.lr,
            "curriculum_type": "interesting_patterns",
            "complexity_start": args.complexity_start,
            "complexity_end": args.complexity_end,
        }
    )

    env_cfg = dict(
        n_variables=args.n_variables,
        max_degree=args.max_degree,
        max_nodes=args.max_nodes,
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        failure_penalty=args.failure_penalty,
    )
    dummy_env = PolynomialEnvironment(
        target_poly_expr=sp.expand((sp.Symbol("x0") + sp.Symbol("x1")) ** 2), **env_cfg
    )
    action_dim = dummy_env.max_actions

    net = PolynomialNet(
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        T=1,
        s_size=dummy_env.S_size,
        device=device,
    )
    
    # Load from checkpoint
    if args.checkpoint_path and args.checkpoint_path != "None" and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        net.load_state_dict(checkpoint)
        print(f"✓ Checkpoint loaded successfully")
    else:
        print("WARNING: Starting from scratch! (Provide --checkpoint_path to continue training)")
    
    net.to(device)
    mcts = PolynomialMCTS(
        net=net, 
        simulations=args.mcts_simulations, 
        c_puct=args.c_puct, 
        device=device,
        virtual_loss=args.virtual_loss
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    num_workers = args.num_workers if args.num_workers > 0 else 1
    use_parallel = num_workers > 1
    temp_dir = tempfile.mkdtemp(prefix="mcts_episodes_interesting_")
    
    if use_parallel:
        print(f"Using {num_workers} workers for parallel episode generation")
    
    current_complexity = args.complexity_start
    max_complexity = args.complexity_end
    complexity_success_window = []
    window_size = 5
    
    print(f"\nInteresting Patterns Curriculum:")
    print(f"  Starting complexity: {current_complexity}")
    print(f"  Max complexity: {max_complexity}")
    print(f"  Mixing 50% random / 50% interesting patterns")

    for epoch in range(args.epochs):
        samples = []
        episode_rewards = []
        episode_lengths = []
        episode_complexities = []
        success_count = 0
        seen_polynomials = set()
        
        print(f"\nEpoch {epoch+1}/{args.epochs} - Complexity: {current_complexity}")
        
        if use_parallel:
            net_state_path = os.path.join(temp_dir, f"net_state_epoch{epoch}.pt")
            torch.save(net.state_dict(), net_state_path)
            
            worker_args = [
                (env_cfg, net_state_path, args.mcts_simulations, args.c_puct, args.virtual_loss, str(device), 
                 action_dim, args.hidden_dim, 1, dummy_env.S_size, epoch * args.episodes_per_epoch + i, temp_dir, current_complexity)
                for i in range(args.episodes_per_epoch)
            ]
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                result_files = list(executor.map(generate_episode_worker_file, worker_args))
            
            for ep_idx, result_file in enumerate(result_files):
                with open(result_file, 'rb') as f:
                    ep_samples, target, complexity = pickle.load(f)
                samples.extend(ep_samples)
                if ep_samples:
                    reward = ep_samples[0][-1]
                    episode_rewards.append(reward)
                    episode_lengths.append(len(ep_samples))
                    episode_complexities.append(complexity)
                    if reward > 0:
                        success_count += 1
                os.remove(result_file)
                
                if (ep_idx + 1) % 10 == 0:
                    print(f"    Generated {ep_idx + 1}/{len(result_files)} episodes")
            
            os.remove(net_state_path)
        else:
            for ep_idx in range(args.episodes_per_epoch):
                ep_samples, target, complexity = generate_episode_complexity(env_cfg, net, mcts, device, current_complexity, seen_polynomials)
                samples.extend(ep_samples)
                if ep_samples:
                    reward = ep_samples[0][-1]
                    episode_rewards.append(reward)
                    episode_lengths.append(len(ep_samples))
                    episode_complexities.append(complexity)
                    if reward > 0:
                        success_count += 1
                if (ep_idx + 1) % 10 == 0:
                    print(f"    Generated {ep_idx + 1}/{args.episodes_per_epoch} episodes")
        
        if len(samples) == 0:
            continue
        
        dataset = TrajectoryDataset(samples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        net.set_mode("train")
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        for batch in loader:
            tensors, scalars, mask, pi, returns = [b.to(device) for b in batch]
            valid = mask.any(dim=1)
            if valid.sum() == 0: continue
            tensors, scalars, mask, pi, returns = tensors[valid], scalars[valid], mask[valid], pi[valid], returns[valid]

            optimizer.zero_grad()
            logits, values = net(tensors=tensors, scalars=scalars, mask=mask)

            policy_loss = F.cross_entropy(logits, pi.argmax(dim=1))
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + args.value_coef * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_pl = total_policy_loss / max(num_batches, 1)
        avg_vl = total_value_loss / max(num_batches, 1)
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        success_rate = success_count / len(episode_rewards) if episode_rewards else 0.0

        complexity_success_window.append(success_rate)
        if len(complexity_success_window) > window_size:
            complexity_success_window.pop(0)
        
        if len(complexity_success_window) >= window_size:
            window_avg_success = np.mean(complexity_success_window)
            if window_avg_success >= 0.70 and current_complexity < max_complexity:
                current_complexity += 1
                complexity_success_window = []
                print(f"\n  ✓ CURRICULUM ADVANCED! New complexity: {current_complexity}")

        print(
            f"  Epoch {epoch+1}: loss={avg_pl+avg_vl:.4f}, "
            f"reward={avg_reward:.2f}, success={success_rate:.2%}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "policy_loss": avg_pl,
            "value_loss": avg_vl,
            "avg_episode_reward": avg_reward,
            "current_complexity": current_complexity,
            "success_rate": success_rate,
        })
        
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                os.path.dirname(args.save_path), 
                f"polynomial_net_interesting_epoch{epoch+1}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(net.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(net.state_dict(), args.save_path)
    wandb.save(args.save_path)
    print(f"\nTraining finished. Saved to {args.save_path}")
    
    import shutil
    shutil.rmtree(temp_dir)
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Training with interesting algebraic patterns")
    p.add_argument("--n_variables", type=int, default=3)
    p.add_argument("--max_degree", type=int, default=3)
    p.add_argument("--max_nodes", type=int, default=10)
    p.add_argument("--step_penalty", type=float, default=-0.1)
    p.add_argument("--success_reward", type=float, default=10.0)
    p.add_argument("--failure_penalty", type=float, default=-5.0)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--mcts_simulations", type=int, default=256)
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--episodes_per_epoch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=32)
    p.add_argument("--virtual_loss", type=float, default=1.0)
    p.add_argument("--checkpoint_path", type=str, default="src/OpenTensor/codes/scripts/runs/polynomial_net_complexity.pth")
    p.add_argument("--save_path", type=str, default="src/OpenTensor/codes/scripts/runs/polynomial_net_interesting.pth")
    p.add_argument("--checkpoint_freq", type=int, default=20)
    p.add_argument("--complexity_start", type=int, default=2)
    p.add_argument("--complexity_end", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
