import pytest
import os
import tempfile

from poly_circuit_rl.config import Config
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv
from poly_circuit_rl.rl.agent import DQNAgent
from poly_circuit_rl.rl.trainer import evaluate, train, _presample_eval_targets


@pytest.mark.slow
def test_sparse_reward_one_op_one_var_learns():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            n_vars=1,
            max_ops=1,
            L=4,
            max_nodes=4,
            m=8,
            total_steps=1200,
            eval_every=100,
            eval_episodes=20,
            curriculum_levels=(1,),
            batch_size=32,
            buffer_size=5000,
            learning_starts=20,
            train_freq=1,
            eps_decay_steps=300,
            reward_mode="sparse",
            expert_demo_count=0,
            use_mcts=False,
            dropout=0.0,
            factor_library_enabled=False,
            auto_interesting=False,
            log_dir=tmpdir,
        )
        _ = train(config)

        best_path = os.path.join(tmpdir, "best_lvl0.pt")
        assert os.path.exists(best_path), "training did not produce a best checkpoint"

        best_agent = DQNAgent(config)
        best_agent.load(best_path)
        eval_env = PolyCircuitEnv(config)
        eval_targets = _presample_eval_targets(config, [1], interesting_sampler=None)[0]
        final_sr = evaluate(
            eval_env,
            best_agent,
            max_ops=1,
            num_episodes=len(eval_targets),
            level_idx=0,
            eval_targets=eval_targets,
        )["success_rate"]
    assert final_sr > 0.5, f"agent failed to learn 1-op task: SR={final_sr}"
