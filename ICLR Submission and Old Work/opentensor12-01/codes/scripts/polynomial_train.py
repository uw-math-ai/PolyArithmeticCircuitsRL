"""
Polynomial circuit training loop using MCTS targets (AlphaZero-style) on GPU.

Strategy:
- Sample random target polynomials.
- For each target, run an episode with PolynomialMCTS to get per-step policies (visit counts).
- Train PolynomialNet to match MCTS policies (cross-entropy) and predict returns (MSE).
"""
import argparse
import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb
import torch.multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from codes.env.polynomial_environment import PolynomialEnvironment
from codes.mcts.polynomial_mcts import PolynomialMCTS
from codes.net.polynomial_net import PolynomialNet
from codes.utils.polynomial_random import random_polynomial
from polynomial_env.actions import encode_action

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


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


def generate_episode(env_cfg, net, mcts, device):
    """Run one self-play episode and return list of (state, mask, pi, reward)."""
    # Sample target polynomial
    target = random_polynomial(env_cfg["n_variables"], env_cfg["max_degree"])
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
    return samples, target


def generate_episode_worker_simple(args_tuple):
    """Simplified worker function - recreates network and loads weights."""
    env_cfg, net_state_dict, mcts_simulations, c_puct, virtual_loss, device_str, action_dim, hidden_dim, T, s_size, seed = args_tuple
    
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Use CUDA in workers (spawn method allows this)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    net = PolynomialNet(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        T=T,
        s_size=s_size,
        device=device,
    )
    # Load weights from main process
    net.load_state_dict(net_state_dict)
    net.to(device)
    net.eval()
    
    mcts = PolynomialMCTS(
        net=net, 
        simulations=mcts_simulations, 
        c_puct=c_puct, 
        device=device,
        virtual_loss=virtual_loss
    )
    
    return generate_episode(env_cfg, net, mcts, device)


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize wandb
    wandb.init(
        project="polynomial-mcts-training",
        entity="zengrf-university-of-washington",
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
            "learning_rate": args.lr,
            "value_coef": args.value_coef,
        },
        name=f"mcts_n{args.n_variables}_d{args.max_degree}_nodes{args.max_nodes}"
    )
    
    env_cfg = dict(
        n_variables=args.n_variables,
        max_degree=args.max_degree,
        max_nodes=args.max_nodes,
        T=1,
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
    net.to(device)
    mcts = PolynomialMCTS(
        net=net, 
        simulations=args.mcts_simulations, 
        c_puct=1.0, 
        device=device,
        virtual_loss=args.virtual_loss
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # Determine number of workers for parallel episode generation
    # Don't limit by CPU count since GPU has plenty of memory and workers are mostly I/O bound
    num_workers = args.num_workers if args.num_workers > 0 else 1
    use_parallel = num_workers > 1
    
    if use_parallel:
        print(f"Using {num_workers} workers for parallel episode generation")
        if num_workers > mp.cpu_count():
            print(f"  Note: Using {num_workers} workers with {mp.cpu_count()} CPU cores (GPU-bound workload)")

    for epoch in range(args.epochs):
        samples = []
        episode_rewards = []
        episode_lengths = []
        
        print(f"Epoch {epoch+1}/{args.epochs} - Generating {args.episodes_per_epoch} episodes...")
        
        if use_parallel:
            # Parallel episode generation with single-task-per-worker to avoid pipe buffer overflow
            net_state_dict = {k: v.cpu() for k, v in net.state_dict().items()}  # Move to CPU for serialization
            
            worker_args = [
                (env_cfg, net_state_dict, args.mcts_simulations, 1.0, args.virtual_loss, str(device), 
                 action_dim, args.hidden_dim, 1, dummy_env.S_size, epoch * args.episodes_per_epoch + i)
                for i in range(args.episodes_per_epoch)
            ]
            
            # Use maxtasksperchild=1 and process one result at a time to avoid large pipe buffers
            with mp.Pool(processes=min(num_workers, 8), maxtasksperchild=1) as pool:
                for ep_idx, (ep_samples, target) in enumerate(pool.imap_unordered(generate_episode_worker_simple, worker_args)):
                    samples.extend(ep_samples)
                    if ep_samples:
                        episode_rewards.append(ep_samples[0][-1])
                        episode_lengths.append(len(ep_samples))
                    if (ep_idx + 1) % 10 == 0 or ep_idx + 1 == len(worker_args):
                        print(f"  Generated {ep_idx + 1}/{len(worker_args)} episodes")
        else:
            # Sequential episode generation (original)
            for ep_idx in range(args.episodes_per_epoch):
                ep_samples, target = generate_episode(env_cfg, net, mcts, device)
                samples.extend(ep_samples)
                if ep_samples:
                    episode_rewards.append(ep_samples[0][-1])  # reward is last element
                    episode_lengths.append(len(ep_samples))
                if (ep_idx + 1) % 10 == 0:
                    print(f"  Generated {ep_idx + 1}/{args.episodes_per_epoch} episodes")
        
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
            logits, values = net(tensors, scalars=scalars, mask=mask)
            # Align pi with valid actions and renormalize.
            pi = pi * mask.float()
            pi_sum = pi.sum(dim=-1, keepdim=True)
            pi = torch.where(pi_sum > 0, pi / pi_sum, torch.zeros_like(pi))

            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
            policy_loss = -(pi * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + args.value_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_policy_loss += policy_loss.item() * tensors.size(0)
            total_value_loss += value_loss.item() * tensors.size(0)
            num_batches += 1

        avg_pl = total_policy_loss / len(dataset) if len(dataset) > 0 else 0.0
        avg_vl = total_value_loss / len(dataset) if len(dataset) > 0 else 0.0
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        print(f"Epoch {epoch+1}: policy_loss={avg_pl:.4f}, value_loss={avg_vl:.4f}, "
              f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}, samples={len(dataset)}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "policy_loss": avg_pl,
            "value_loss": avg_vl,
            "total_loss": avg_pl + args.value_coef * avg_vl,
            "avg_episode_reward": avg_reward,
            "avg_episode_length": avg_length,
            "num_samples": len(dataset),
        })

    # Save the model
    torch.save(net.state_dict(), args.save_path)
    wandb.save(args.save_path)
    print(f"Training finished. Model saved to {args.save_path}")
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_variables", type=int, default=2)
    p.add_argument("--max_degree", type=int, default=3)
    p.add_argument("--max_nodes", type=int, default=8)
    p.add_argument("--step_penalty", type=float, default=-0.1)
    p.add_argument("--success_reward", type=float, default=10.0)
    p.add_argument("--failure_penalty", type=float, default=-5.0)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--mcts_simulations", type=int, default=128, help="Number of MCTS simulations per step (reduced from 256 for speed)")
    p.add_argument("--episodes_per_epoch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for episode generation (0 or 1 for sequential)")
    p.add_argument("--virtual_loss", type=float, default=1.0, help="Virtual loss penalty for parallel MCTS simulations")
    p.add_argument("--save_path", type=str, default="polynomial_net.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
