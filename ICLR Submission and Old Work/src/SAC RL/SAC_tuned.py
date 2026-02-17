import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
PPO_DIR = SRC_ROOT / "PPO RL"
for path in (CURRENT_DIR, SRC_ROOT, PROJECT_ROOT, PPO_DIR):
	path_str = str(path)
	if path_str not in sys.path:
		sys.path.insert(0, path_str)

import torch
import wandb

import SAC as sac
from encoders.compact_encoder import CompactOneHotGraphEncoder


def _recompute_compact_size(config: sac.Config) -> None:
	config.compact_size = CompactOneHotGraphEncoder(
		N=config.max_complexity,
		P=config.mod,
		D=config.n_variables,
	).size


class TunedConfig(sac.Config):
	def __init__(self):
		super().__init__()
		# Reward shaping (OpenTensor-style)
		self.step_penalty = -0.1
		self.success_reward = 10.0
		self.failure_penalty = -5.0

		# Curriculum tuning
		self.complexity_threshold = 0.65
		self.complexity_window = 400

		# Data / update balance
		self.steps_per_iter = 3072
		self.updates_per_iter = 512
		self.batch_size = 256

		# Logging / prefill
		self.show_progress_bars = True
		self.use_synthetic_dataset = False

		# Target sampling and MCTS guidance
		self.training_target_mode = "mixed"
		self.mcts_policy_mix = 0.5
		self.mcts_simulations = 96
		self.mcts_ce_coef = 0.6

		# Keep derived values in sync
		self.max_degree = self.max_complexity * 2
		_recompute_compact_size(self)


def main():
	if torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	config = TunedConfig()
	print(f"Compact encoder size: {config.compact_size}")

	if config.use_wandb:
		wandb.init(
			project=config.wandb_project,
			name=config.wandb_run_name,
			config={
				"n_variables": config.n_variables,
				"max_complexity": config.max_complexity,
				"max_degree": config.max_degree,
				"hidden_dim": config.hidden_dim,
				"embedding_dim": config.embedding_dim,
				"num_gnn_layers": config.num_gnn_layers,
				"num_transformer_layers": config.num_transformer_layers,
				"transformer_heads": config.transformer_heads,
				"transformer_dropout": config.transformer_dropout,
				"mod": config.mod,
				"gamma": config.gamma,
				"alpha": config.alpha,
				"tau": config.tau,
				"learning_rate": config.learning_rate,
				"batch_size": config.batch_size,
				"buffer_size": config.buffer_size,
				"min_buffer_size": config.min_buffer_size,
				"steps_per_iter": config.steps_per_iter,
				"updates_per_iter": config.updates_per_iter,
				"action_temperature": config.action_temperature,
				"step_penalty": config.step_penalty,
				"success_reward": config.success_reward,
				"failure_penalty": config.failure_penalty,
				"use_mcts": config.use_mcts,
				"mcts_simulations": config.mcts_simulations,
				"mcts_exploration": config.mcts_exploration,
				"mcts_policy_mix": config.mcts_policy_mix,
				"mcts_policy_temperature": config.mcts_policy_temperature,
				"mcts_ce_coef": config.mcts_ce_coef,
				"complexity_threshold": config.complexity_threshold,
				"complexity_window": config.complexity_window,
				"training_target_mode": config.training_target_mode,
				"use_synthetic_dataset": config.use_synthetic_dataset,
				"synthetic_samples": config.synthetic_samples,
				"synthetic_complexity_min": config.synthetic_complexity_min,
				"synthetic_complexity_max": config.synthetic_complexity_max,
			},
		)

	model = sac.SACCircuitBuilder(config, config.compact_size).to(sac.device)
	target_model = sac.SACCircuitBuilder(config, config.compact_size).to(sac.device)
	checkpoint_path = PROJECT_ROOT / "sac_copy_n3c8.pt"
	if checkpoint_path.exists():
		state_dict = torch.load(checkpoint_path, map_location=sac.device)
		model.load_state_dict(state_dict)
		print(f"Loaded checkpoint: {checkpoint_path}")
	target_model.load_state_dict(model.state_dict())

	optimizer = torch.optim.Adam(
		model.parameters(), lr=config.learning_rate, eps=config.rl_eps
	)

	buffer = sac.ReplayBuffer(config.buffer_size)
	planner = sac.MCTSPlanner(config) if config.use_mcts else None

	if config.use_synthetic_dataset:
		synthetic_transitions = sac.generate_synthetic_transitions(
			config,
			samples_n=config.synthetic_samples,
			complexity_min=config.synthetic_complexity_min,
			complexity_max=config.synthetic_complexity_max,
		)
		for transition in synthetic_transitions:
			buffer.add(*transition)
		print(f"Prefilled replay buffer with {len(synthetic_transitions)} synthetic steps")

	current_complexity = 1
	recent_successes = []
	target_pool_by_complexity = {}

	last_iteration = 0
	last_success_rate = 0.0
	last_metrics = {}
	best_success_rate = -1.0
	best_iteration = 0
	interrupted = False

	try:
		for iteration in range(1, 10001):
			if config.training_target_mode == "pool":
				pool = target_pool_by_complexity.get(current_complexity)
				if pool is None:
					pool = sac.build_target_pool(
						config,
						current_complexity,
						sac.load_interesting_circuit_data(config),
					)
					target_pool_by_complexity[current_complexity] = pool
				target_pool = pool
			else:
				target_pool = None

			success_rate, circuit_examples, iteration_successes = sac.collect_experience(
				model,
				config,
				planner,
				buffer,
				current_complexity,
				target_pool=target_pool,
			)

			last_iteration = iteration
			last_success_rate = success_rate
			if success_rate > best_success_rate:
				best_success_rate = success_rate
				best_iteration = iteration

			recent_successes.extend(iteration_successes)
			recent_successes = recent_successes[-config.complexity_window:]
			if len(recent_successes) >= config.complexity_window // 2:
				recent_success_rate = sum(recent_successes) / len(recent_successes)
				if (
					recent_success_rate > config.complexity_threshold
					and current_complexity < config.max_complexity
				):
					current_complexity += 1
					recent_successes = []
					print(
						f"*** Complexity Increased to {current_complexity} (SR: {recent_success_rate:.2f}) ***"
					)

			metrics = sac.sac_update(model, target_model, optimizer, buffer, config)
			last_metrics = metrics
			print(
				f"Iter {iteration}: SR {success_rate:.1f}%, "
				f"Q {metrics.get('q_loss', 0.0):.4f}, "
				f"Pi {metrics.get('policy_loss', 0.0):.4f}, "
				f"CE {metrics.get('ce_loss', 0.0):.4f}, "
				f"Buffer {len(buffer)}"
			)
			if config.use_wandb:
				wandb.log(
					{
						"success_rate": success_rate,
						"q_loss": metrics.get("q_loss", 0.0),
						"policy_loss": metrics.get("policy_loss", 0.0),
						"ce_loss": metrics.get("ce_loss", 0.0),
						"buffer_size": len(buffer),
						"complexity": current_complexity,
					},
					step=iteration,
				)
			for i, ex in enumerate(circuit_examples):
				print(
					f"  Ex {i + 1}: {'Success' if ex['success'] else 'Fail'} "
					f"(R: {ex['reward']:.2f}, Steps: {ex['steps']}) Target: {ex['target']}"
				)

			if iteration % 50 == 0:
				path = f"sac_tuned_model_n{config.n_variables}_C{config.max_complexity}.pt"
				torch.save(model.state_dict(), path)
				print(f"  Model saved to {path}")
	except KeyboardInterrupt:
		interrupted = True
		interrupt_path = (
			f"sac_tuned_model_n{config.n_variables}_C{config.max_complexity}_interrupt.pt"
		)
		torch.save(model.state_dict(), interrupt_path)
		print(f"\nTraining interrupted. Model saved to {interrupt_path}")

	if last_iteration == 0:
		print("\n=== Training Summary ===")
		print("No iterations completed.")
		if interrupted:
			print("Run status: interrupted")
		return

	print("\n=== Training Summary ===")
	print(f"Total iterations: {last_iteration}")
	print(f"Final complexity: {current_complexity}")
	if best_success_rate >= 0:
		print(f"Best SR: {best_success_rate:.1f}% (Iter {best_iteration})")
	print(f"Final SR: {last_success_rate:.1f}%")
	print(
		"Final losses: "
		f"Q {last_metrics.get('q_loss', 0.0):.4f}, "
		f"Pi {last_metrics.get('policy_loss', 0.0):.4f}, "
		f"CE {last_metrics.get('ce_loss', 0.0):.4f}"
	)
	print(f"Buffer size: {len(buffer)}")
	if interrupted:
		print("Run status: interrupted")


if __name__ == "__main__":
	main()
