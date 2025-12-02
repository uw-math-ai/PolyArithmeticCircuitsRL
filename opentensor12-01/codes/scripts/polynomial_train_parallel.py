"""
Polynomial circuit training loop with TRUE parallel processing using file-based communication.

This version uses file I/O instead of pipes to avoid BrokenPipeError with many workers.
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


def generate_episode(env_cfg, net, mcts, device):
    """Run one self-play episode and return list of (state, mask, pi, reward)."""
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


def generate_episode_worker_file(args_tuple):
    """Worker function that saves results to file instead of returning via pipe."""
    (env_cfg, net_state_path, mcts_simulations, c_puct, virtual_loss, device_str, 
     action_dim, hidden_dim, T, s_size, episode_id, output_dir) = args_tuple
    
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
    
    # Generate episode
    ep_samples, target = generate_episode(env_cfg, net, mcts, device)
    
    # Save to file instead of returning
    output_file = os.path.join(output_dir, f"episode_{episode_id}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump((ep_samples, target), f)
    
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
        name=f"mcts_n{args.n_variables}_d{args.max_degree}_nodes{args.max_nodes}_parallel",
        config={
            "n_variables": args.n_variables,
            "max_degree": args.max_degree,
            "max_nodes": args.max_nodes,
            "hidden_dim": args.hidden_dim,
            "mcts_simulations": args.mcts_simulations,
            "episodes_per_epoch": args.episodes_per_epoch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "virtual_loss": args.virtual_loss,
            "parallel_mode": "file_based",
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
    net.to(device)
    mcts = PolynomialMCTS(
        net=net, 
        simulations=args.mcts_simulations, 
        c_puct=1.0, 
        device=device,
        virtual_loss=args.virtual_loss
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # Determine number of workers
    num_workers = args.num_workers if args.num_workers > 0 else 1
    use_parallel = num_workers > 1
    
    # Create temp directory for file-based communication
    temp_dir = tempfile.mkdtemp(prefix="mcts_episodes_")
    
    if use_parallel:
        print(f"Using {num_workers} workers for TRUE parallel episode generation (file-based)")
        print(f"Temp directory: {temp_dir}")

    for epoch in range(args.epochs):
        samples = []
        episode_rewards = []
        episode_lengths = []
        
        print(f"Epoch {epoch+1}/{args.epochs} - Generating {args.episodes_per_epoch} episodes...")
        
        if use_parallel:
            # Save network state to file
            net_state_path = os.path.join(temp_dir, f"net_state_epoch{epoch}.pt")
            torch.save(net.state_dict(), net_state_path)
            
            # Prepare worker arguments
            worker_args = [
                (env_cfg, net_state_path, args.mcts_simulations, 1.0, args.virtual_loss, str(device), 
                 action_dim, args.hidden_dim, 1, dummy_env.S_size, epoch * args.episodes_per_epoch + i, temp_dir)
                for i in range(args.episodes_per_epoch)
            ]
            
            # Use ProcessPoolExecutor for better resource management
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                result_files = list(executor.map(generate_episode_worker_file, worker_args))
            
            # Load results from files
            for ep_idx, result_file in enumerate(result_files):
                with open(result_file, 'rb') as f:
                    ep_samples, target = pickle.load(f)
                samples.extend(ep_samples)
                if ep_samples:
                    episode_rewards.append(ep_samples[0][-1])
                    episode_lengths.append(len(ep_samples))
                
                # Clean up result file
                os.remove(result_file)
                
                if (ep_idx + 1) % 10 == 0 or ep_idx + 1 == len(result_files):
                    print(f"  Generated {ep_idx + 1}/{len(result_files)} episodes")
            
            # Clean up network state file
            os.remove(net_state_path)
        else:
            # Sequential episode generation
            for ep_idx in range(args.episodes_per_epoch):
                ep_samples, target = generate_episode(env_cfg, net, mcts, device)
                samples.extend(ep_samples)
                if ep_samples:
                    episode_rewards.append(ep_samples[0][-1])
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

            optimizer.zero_grad()
            logits, values = net(tensors=tensors, scalars=scalars, mask=mask)

            policy_loss = F.cross_entropy(logits, pi.argmax(dim=1))
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + args.value_coef * value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_pl = total_policy_loss / max(num_batches, 1)
        avg_vl = total_value_loss / max(num_batches, 1)
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0

        print(
            f"Epoch {epoch+1}: policy_loss={avg_pl:.4f}, value_loss={avg_vl:.4f}, "
            f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}, samples={len(dataset)}"
        )

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
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(net.state_dict(), args.save_path)
    wandb.save(args.save_path)
    print(f"Training finished. Model saved to {args.save_path}")
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)
    
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
    p.add_argument("--mcts_simulations", type=int, default=128, help="Number of MCTS simulations per step")
    p.add_argument("--episodes_per_epoch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers (no limit with file-based approach)")
    p.add_argument("--virtual_loss", type=float, default=1.0, help="Virtual loss penalty for parallel MCTS simulations")
    p.add_argument("--save_path", type=str, default="src/OpenTensor/codes/scripts/runs/polynomial_net_parallel.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
