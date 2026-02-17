"""
Polynomial circuit training with curriculum learning for higher complexity polynomials.

Strategy:
- Start from existing checkpoint trained on simpler polynomials
- Gradually increase complexity: more terms, higher max_nodes
- Use adaptive MCTS simulations based on complexity
- Save periodic checkpoints
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


def sample_polynomial_by_complexity(n_variables: int, max_degree: int, complexity: int, max_attempts: int = 10):
    """
    Generate a polynomial by performing 'complexity' random operations.
    
    This uses a random construction approach that builds variety exponentially:
    - Start with base polynomials (variables and their squares)
    - Apply random add/multiply operations 'complexity' times
    - Each operation choice creates different paths, giving massive variety at higher complexity
    
    For complexity C, we have roughly O(n^C) different possible polynomials.
    """
    symbols = sp.symbols(f"x0:{n_variables}")
    
    # Random construction approach for maximum variety
    for attempt in range(max_attempts):
        # Start with a pool of available polynomials
        available = list(symbols)  # [x0, x1, x2, ...]
        
        # Also include squares as base building blocks
        for i in range(n_variables):
            available.append(symbols[i]**2)
        
        # Perform 'complexity' random operations
        current_poly = None
        for op_idx in range(complexity):
            # Choose operation type: add (60%) or multiply (40%)
            # Bias toward addition to avoid degree explosion
            op_type = np.random.choice(['add', 'multiply'], p=[0.6, 0.4])
            
            if op_type == 'add':
                # Add two random polynomials from available pool
                if len(available) >= 2:
                    idx1, idx2 = np.random.choice(len(available), size=2, replace=False)
                    poly1, poly2 = available[idx1], available[idx2]
                    # Optionally multiply by random small coefficient
                    coeff = np.random.choice([1, 1, 1, 2])  # Mostly 1, sometimes 2
                    new_poly = sp.expand(coeff * poly1 + poly2)
                    current_poly = new_poly
                    available.append(new_poly)
                else:
                    # Not enough polynomials, just pick one
                    current_poly = available[0] if available else symbols[0]
                    
            else:  # multiply
                # Multiply two random polynomials
                if len(available) >= 2:
                    idx1, idx2 = np.random.choice(len(available), size=2, replace=False)
                    poly1, poly2 = available[idx1], available[idx2]
                    new_poly = sp.expand(poly1 * poly2)
                    
                    # Check if degree is within bounds
                    try:
                        poly_obj = new_poly.as_poly(*symbols)
                        if poly_obj and poly_obj.total_degree() <= max_degree:
                            current_poly = new_poly
                            available.append(new_poly)
                        else:
                            # Degree too high, fall back to addition
                            new_poly = sp.expand(poly1 + poly2)
                            current_poly = new_poly
                            available.append(new_poly)
                    except:
                        # Error in poly construction, use addition
                        new_poly = sp.expand(poly1 + poly2)
                        current_poly = new_poly
                        available.append(new_poly)
                else:
                    current_poly = available[0] if available else symbols[0]
        
        # Final result
        result = current_poly if current_poly is not None else symbols[0]
        result = sp.expand(result)
        
        # Verify it's not trivial (not zero, not just a constant)
        if result != 0 and not result.is_number:
            # Verify degree is within bounds
            try:
                poly_obj = result.as_poly(*symbols)
                if poly_obj and poly_obj.total_degree() <= max_degree:
                    return result
            except:
                pass
    
    # Fallback: return a simple polynomial if all attempts failed
    return sp.expand(symbols[0] + symbols[1 % n_variables])


def generate_episode_complexity(env_cfg, net, mcts, device, complexity: int, seen_polynomials=None):
    """
    Run one self-play episode with target polynomial of given complexity.
    Avoids generating duplicate polynomials if seen_polynomials is provided.
    """
    max_attempts = 20
    for attempt in range(max_attempts):
        target = sample_polynomial_by_complexity(
            env_cfg["n_variables"], 
            env_cfg["max_degree"],
            complexity=complexity
        )
        
        # Check if we've seen this polynomial before
        target_str = str(sp.expand(target))
        if seen_polynomials is None or target_str not in seen_polynomials:
            if seen_polynomials is not None:
                seen_polynomials.add(target_str)
            break
    
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
    
    # Recreate network and MCTS in worker process
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    net = PolynomialNet(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        T=T,
        s_size=s_size,
        device=device,
    )
    
    # Load network state from file (suppress warnings)
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
    
    # Generate episode with specific complexity
    ep_samples, target, actual_complexity = generate_episode_complexity(env_cfg, net, mcts, device, complexity)
    
    # Save to file instead of returning
    output_file = os.path.join(output_dir, f"episode_{episode_id}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump((ep_samples, target, actual_complexity), f)
    
    # Return just the filename (small data)
    return output_file


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Initialize wandb
    wandb.init(
        project="polynomial-mcts-training",
        entity="zengrf-university-of-washington",
        name=f"mcts_complexity_v{args.n_variables}_d{args.max_degree}_adaptive",
        config={
            "n_variables": args.n_variables,
            "max_degree": args.max_degree,
            "max_nodes": args.max_nodes,
            "step_penalty": args.step_penalty,
            "success_reward": args.success_reward,
            "failure_penalty": args.failure_penalty,
            "hidden_dim": args.hidden_dim,
            "mcts_simulations": args.mcts_simulations,
            "episodes_per_epoch": args.episodes_per_epoch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "virtual_loss": args.virtual_loss,
            "checkpoint_path": args.checkpoint_path,
            "curriculum_type": "adaptive_complexity",
            "no_constant_term": True,
            "complexity_start": args.complexity_start,
            "complexity_end": args.complexity_end,
            "adaptive_threshold": 0.70,
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
    
    # Load from checkpoint if provided
    if args.checkpoint_path and args.checkpoint_path != "None" and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        net.load_state_dict(checkpoint)
        print(f"✓ Checkpoint loaded successfully")
    else:
        print("Starting training from scratch (no checkpoint)")
    
    net.to(device)
    mcts = PolynomialMCTS(
        net=net, 
        simulations=args.mcts_simulations, 
        c_puct=args.c_puct, 
        device=device,
        virtual_loss=args.virtual_loss
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # Determine number of workers
    num_workers = args.num_workers if args.num_workers > 0 else 1
    use_parallel = num_workers > 1
    
    # Create temp directory for file-based communication
    temp_dir = tempfile.mkdtemp(prefix="mcts_episodes_curriculum_")
    
    if use_parallel:
        print(f"Using {num_workers} workers for parallel episode generation (file-based)")
        print(f"Temp directory: {temp_dir}")
    
    # Adaptive Curriculum: start with simple circuits, increase when success rate > 70%
    current_complexity = args.complexity_start
    max_complexity = args.complexity_end
    complexity_success_window = []  # Track recent success rates
    window_size = 5  # Number of epochs to consider for advancement
    
    print(f"\nAdaptive Complexity-Based Curriculum:")
    print(f"  Starting complexity: {current_complexity} operations")
    print(f"  Maximum complexity: {max_complexity} operations")
    print(f"  Advancement threshold: 70% success rate over {window_size} epochs")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Variables: {args.n_variables} (x0 to x{args.n_variables-1})")

    for epoch in range(args.epochs):
        samples = []
        episode_rewards = []
        episode_lengths = []
        episode_complexities = []
        success_count = 0
        seen_polynomials = set()  # Track polynomials in this epoch to avoid duplicates
        
        print(f"\nEpoch {epoch+1}/{args.epochs} - Current complexity: {current_complexity} operations")
        print(f"  Generating {args.episodes_per_epoch} episodes...")
        
        if use_parallel:
            # Save network state to file
            net_state_path = os.path.join(temp_dir, f"net_state_epoch{epoch}.pt")
            torch.save(net.state_dict(), net_state_path)
            
            # Prepare worker arguments with current complexity level
            worker_args = [
                (env_cfg, net_state_path, args.mcts_simulations, args.c_puct, args.virtual_loss, str(device), 
                 action_dim, args.hidden_dim, 1, dummy_env.S_size, epoch * args.episodes_per_epoch + i, temp_dir, current_complexity)
                for i in range(args.episodes_per_epoch)
            ]
            
            # Use ProcessPoolExecutor for better resource management
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                result_files = list(executor.map(generate_episode_worker_file, worker_args))
            
            # Load results from files
            for ep_idx, result_file in enumerate(result_files):
                with open(result_file, 'rb') as f:
                    ep_samples, target, complexity = pickle.load(f)
                samples.extend(ep_samples)
                if ep_samples:
                    reward = ep_samples[0][-1]
                    episode_rewards.append(reward)
                    episode_lengths.append(len(ep_samples))
                    episode_complexities.append(complexity)
                    if reward > 0:  # Success
                        success_count += 1
                
                # Clean up result file
                os.remove(result_file)
                
                if (ep_idx + 1) % 10 == 0 or ep_idx + 1 == len(result_files):
                    print(f"    Generated {ep_idx + 1}/{len(result_files)} episodes")
            
            # Clean up network state file
            os.remove(net_state_path)
        else:
            # Sequential episode generation
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
            print(f"  No samples collected, skipping training")
            continue
        
        dataset = TrajectoryDataset(samples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        print(f"  Training on {len(dataset)} samples...")
        net.set_mode("train")
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        for batch in loader:
            tensors, scalars, mask, pi, returns = [b.to(device) for b in batch]
            valid = mask.any(dim=1)
            if valid.sum() == 0:
                continue
            tensors, scalars, mask, pi, returns = (
                tensors[valid],
                scalars[valid],
                mask[valid],
                pi[valid],
                returns[valid],
            )

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
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_complexity = np.mean(episode_complexities) if episode_complexities else 0.0
        success_rate = success_count / len(episode_rewards) if episode_rewards else 0.0

        # Track success rate for curriculum advancement
        complexity_success_window.append(success_rate)
        if len(complexity_success_window) > window_size:
            complexity_success_window.pop(0)
        
        # Check if we should advance to next complexity level
        if len(complexity_success_window) >= window_size:
            window_avg_success = np.mean(complexity_success_window)
            if window_avg_success >= 0.70 and current_complexity < max_complexity:
                current_complexity += 1
                complexity_success_window = []  # Reset window
                print(f"\n  ✓ CURRICULUM ADVANCED! New complexity: {current_complexity} operations")
                print(f"    (Achieved {window_avg_success:.1%} success rate over {window_size} epochs)")

        print(
            f"  Epoch {epoch+1}: policy_loss={avg_pl:.4f}, value_loss={avg_vl:.4f}, "
            f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}, "
            f"complexity={current_complexity}, success_rate={success_rate:.2%}, samples={len(dataset)}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "policy_loss": avg_pl,
            "value_loss": avg_vl,
            "total_loss": avg_pl + args.value_coef * avg_vl,
            "avg_episode_reward": avg_reward,
            "avg_episode_length": avg_length,
            "current_complexity": current_complexity,
            "avg_complexity": avg_complexity,
            "success_rate": success_rate,
            "window_avg_success": np.mean(complexity_success_window) if complexity_success_window else 0.0,
            "num_samples": len(dataset),
        })
        
        # Save periodic checkpoints
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                os.path.dirname(args.save_path), 
                f"polynomial_net_complexity{current_complexity}_epoch{epoch+1}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(net.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

    # Save the final model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(net.state_dict(), args.save_path)
    wandb.save(args.save_path)
    print(f"\nTraining finished. Final model saved to {args.save_path}")
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)
    
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Complexity-based curriculum learning for polynomial circuits")
    p.add_argument("--n_variables", type=int, default=3, help="Number of variables (default 3: x0, x1, x2)")
    p.add_argument("--max_degree", type=int, default=3, help="Maximum degree of polynomials")
    p.add_argument("--max_nodes", type=int, default=10, help="Maximum circuit nodes (base + constructed)")
    p.add_argument("--step_penalty", type=float, default=-0.1, help="Penalty for each step")
    p.add_argument("--success_reward", type=float, default=10.0, help="Reward for successfully building target")
    p.add_argument("--failure_penalty", type=float, default=-5.0, help="Penalty for failure")
    p.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of network")
    p.add_argument("--mcts_simulations", type=int, default=256, help="Number of MCTS simulations per step")
    p.add_argument("--c_puct", type=float, default=1.5, help="Exploration constant for MCTS")
    p.add_argument("--episodes_per_epoch", type=int, default=64, help="Episodes per training epoch")
    p.add_argument("--epochs", type=int, default=500, help="Total training epochs")
    p.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    p.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers")
    p.add_argument("--virtual_loss", type=float, default=1.0, help="Virtual loss for parallel MCTS")
    p.add_argument("--checkpoint_path", type=str, default=None, 
                   help="Path to load checkpoint from (None to train from scratch)")
    p.add_argument("--save_path", type=str, default="src/OpenTensor/codes/scripts/runs/polynomial_net_complexity.pth")
    p.add_argument("--checkpoint_freq", type=int, default=50, help="Save checkpoint every N epochs")
    p.add_argument("--complexity_start", type=int, default=1, help="Starting circuit complexity (number of operations)")
    p.add_argument("--complexity_end", type=int, default=6, help="Maximum circuit complexity to reach")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
