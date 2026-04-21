import unittest

import numpy as np
import torch

from poly_circuit_rl.config import Config
from poly_circuit_rl.rl.mcts import MCTS, MCTSNode


class LookupNet(torch.nn.Module):
    def __init__(self, q_by_state):
        super().__init__()
        self.q_by_state = {
            int(state): torch.tensor(values, dtype=torch.float32)
            for state, values in q_by_state.items()
        }

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rows = []
        for state in obs[:, 0].tolist():
            rows.append(self.q_by_state[int(round(float(state)))])
        return torch.stack(rows, dim=0)


class FakeAgent:
    def __init__(self, q_by_state):
        self.device = torch.device("cpu")
        self.q_network = LookupNet(q_by_state)


class FakeEnv:
    def __init__(self, transitions, action_masks):
        self.transitions = transitions
        self.action_masks = action_masks
        self.state = 0
        self._simulation = False

    def get_state(self):
        return {"state": self.state}

    def set_state(self, state):
        self.state = state["state"]
        self._simulation = True

    def step(self, action):
        reward, next_state, terminated = self.transitions[(self.state, action)]
        self.state = next_state
        return (
            {
                "obs": np.array([float(next_state)], dtype=np.float32),
                "action_mask": self.action_masks[next_state].copy(),
            },
            reward,
            terminated,
            False,
            {},
        )


class TestMCTS(unittest.TestCase):
    def _config(self, **overrides):
        base = Config(
            gamma=0.5,
            mcts_simulations=8,
            mcts_c_puct=0.0,
            mcts_temperature=0.0,
        )
        return Config(**{**base.__dict__, **overrides})

    def test_nonterminal_backup_includes_immediate_reward(self):
        env = FakeEnv(
            transitions={(0, 0): (-0.4, 1, False)},
            action_masks={0: np.array([1, 0], dtype=np.int8), 1: np.array([1, 0], dtype=np.int8)},
        )
        agent = FakeAgent({0: [0.0, 0.0], 1: [0.6, 0.2]})
        mcts = MCTS(agent, env, self._config())

        root = MCTSNode(env.get_state(), np.array([0.0], dtype=np.float32), env.action_masks[0].copy())
        mcts._init_node(root)
        child, leaf_value = mcts._expand_one(root)
        mcts._backup(child, leaf_value)

        self.assertAlmostEqual(child.q_value, -0.4 + 0.5 * 0.6, places=6)

    def test_terminal_backup_uses_immediate_reward_only(self):
        env = FakeEnv(
            transitions={(0, 0): (1.25, 1, True)},
            action_masks={0: np.array([1, 0], dtype=np.int8), 1: np.array([0, 0], dtype=np.int8)},
        )
        agent = FakeAgent({0: [0.0, 0.0], 1: [0.0, 0.0]})
        mcts = MCTS(agent, env, self._config())

        root = MCTSNode(env.get_state(), np.array([0.0], dtype=np.float32), env.action_masks[0].copy())
        mcts._init_node(root)
        child, leaf_value = mcts._expand_one(root)
        mcts._backup(child, leaf_value)

        self.assertAlmostEqual(child.q_value, 1.25, places=6)

    def test_select_child_prefers_higher_action_return(self):
        env = FakeEnv(
            transitions={
                (0, 0): (-0.5, 1, False),
                (0, 1): (-0.05, 2, False),
            },
            action_masks={
                0: np.array([1, 1], dtype=np.int8),
                1: np.array([1, 0], dtype=np.int8),
                2: np.array([1, 0], dtype=np.int8),
            },
        )
        agent = FakeAgent({
            0: [0.0, 0.0],
            1: [0.5, 0.0],
            2: [0.2, 0.0],
        })
        mcts = MCTS(agent, env, self._config())

        root = MCTSNode(env.get_state(), np.array([0.0], dtype=np.float32), env.action_masks[0].copy())
        mcts._init_node(root)

        first_child, first_value = mcts._expand_one(root)
        mcts._backup(first_child, first_value)
        second_child, second_value = mcts._expand_one(root)
        mcts._backup(second_child, second_value)

        best_child = mcts._select_child(root)
        self.assertIs(best_child, root.children[1])
        self.assertGreater(root.children[1].q_value, root.children[0].q_value)

    def test_temperature_sampling_is_seed_reproducible(self):
        cfg = self._config(mcts_simulations=4, mcts_c_puct=1.0, mcts_temperature=1.0, seed=777)
        transitions = {
            (0, 0): (0.0, 1, False),
            (0, 1): (0.0, 2, False),
            (1, 0): (0.0, 1, True),
            (2, 0): (0.0, 2, True),
        }
        masks = {
            0: np.array([1, 1], dtype=np.int8),
            1: np.array([1, 0], dtype=np.int8),
            2: np.array([1, 0], dtype=np.int8),
        }
        q = {0: [0.5, 0.5], 1: [0.0, 0.0], 2: [0.0, 0.0]}

        env1 = FakeEnv(transitions=transitions, action_masks=masks)
        env2 = FakeEnv(transitions=transitions, action_masks=masks)
        mcts1 = MCTS(FakeAgent(q), env1, cfg)
        mcts2 = MCTS(FakeAgent(q), env2, cfg)

        action1 = mcts1.search(np.array([0.0], dtype=np.float32), masks[0].copy())
        action2 = mcts2.search(np.array([0.0], dtype=np.float32), masks[0].copy())
        self.assertEqual(action1, action2)

    def test_search_restores_simulation_flag_on_exception(self):
        class RaisingEnv(FakeEnv):
            def step(self, action):
                raise RuntimeError("boom")

        env = RaisingEnv(
            transitions={(0, 0): (0.0, 0, False)},
            action_masks={0: np.array([1, 0], dtype=np.int8)},
        )
        mcts = MCTS(FakeAgent({0: [0.0, 0.0]}), env, self._config())

        with self.assertRaises(RuntimeError):
            mcts.search(np.array([0.0], dtype=np.float32), env.action_masks[0].copy())
        self.assertFalse(env._simulation)


if __name__ == "__main__":
    unittest.main()
