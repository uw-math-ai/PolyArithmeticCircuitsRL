import math
import unittest
import random
import numpy as np
from fractions import Fraction

from poly_circuit_rl.core.builder import CircuitBuilder
from poly_circuit_rl.core.fingerprints import sample_eval_points, eval_poly_points
from poly_circuit_rl.core.poly import make_var, add

from poly_circuit_rl.config import Config
from poly_circuit_rl.env.obs import (
    encode_obs, extract_goal, replace_goal, encode_node, get_num_real_nodes,
    _TYPE_OFFSET, _OP_OFFSET, _PARENT_OFFSET, _POS_OFFSET,
    _leaf_offset, _eval_offset,
)


class TestObsEncoder(unittest.TestCase):

    def setUp(self):
        self.config = Config(n_vars=2, m=8, L=8)
        self.rng = random.Random(42)
        self.eval_points = sample_eval_points(self.rng, 2, 8, -3, 3)
        self.builder = CircuitBuilder(2, eval_points=self.eval_points)

    def test_encode_obs_shape(self):
        obs = encode_obs(
            self.builder.nodes,
            tuple(eval_poly_points(make_var(2, 0), self.eval_points)),
            self.eval_points, steps_left=2, max_ops=4, config=self.config,
        )
        self.assertEqual(obs.shape, (self.config.obs_dim,))
        self.assertEqual(obs.dtype, np.float32)

    def test_obs_dim_matches_config(self):
        # d_node_raw = 3 + 2 + 2 + 1 + 3 + 8 = 19; obs_dim = 8*19 + 8 + 1 = 161
        self.assertEqual(self.config.d_node_raw, 19)
        self.assertEqual(self.config.obs_dim, 8 * 19 + 8 + 1)

    def test_zero_padding(self):
        obs = encode_obs(
            self.builder.nodes,
            tuple(eval_poly_points(make_var(2, 0), self.eval_points)),
            self.eval_points, steps_left=2, max_ops=4, config=self.config,
        )
        d = self.config.d_node_raw
        num_real = len(self.builder.nodes)  # 3: x0, x1, const_1
        for i in range(num_real, self.config.L):
            self.assertAlmostEqual(obs[i * d + _TYPE_OFFSET + 2], 1.0)

    def test_parent_indices_for_ops(self):
        self.builder.add_add(0, 1)
        node = self.builder.nodes[-1]
        feat = encode_node(node, 3, self.eval_points, self.config)
        self.assertEqual(feat[_PARENT_OFFSET], 0.0)
        self.assertEqual(feat[_PARENT_OFFSET + 1], 1.0)

    def test_parent_indices_sentinel(self):
        node = self.builder.nodes[0]  # VAR
        feat = encode_node(node, 0, self.eval_points, self.config)
        self.assertEqual(feat[_PARENT_OFFSET], float(self.config.L))
        self.assertEqual(feat[_PARENT_OFFSET + 1], float(self.config.L))

    def test_position_index(self):
        for i, node in enumerate(self.builder.nodes):
            feat = encode_node(node, i, self.eval_points, self.config)
            self.assertEqual(feat[_POS_OFFSET], float(i))

    def test_steps_left_normalized(self):
        obs = encode_obs(
            self.builder.nodes,
            tuple(eval_poly_points(make_var(2, 0), self.eval_points)),
            self.eval_points, steps_left=2, max_ops=4, config=self.config,
        )
        self.assertAlmostEqual(obs[-1], 0.5)

    def test_eval_vector_correctness(self):
        # Eval values are tanh-normalised before storage; compare against normalised expected.
        node = self.builder.nodes[0]  # x0
        feat = encode_node(node, 0, self.eval_points, self.config)
        eo = _eval_offset(self.config)
        scale = self.config.eval_norm_scale
        expected = eval_poly_points(node.poly, self.eval_points)
        for k in range(self.config.m):
            expected_norm = math.tanh(float(expected[k]) / scale)
            self.assertAlmostEqual(feat[eo + k], expected_norm, places=5)

    def test_leaf_id_correctness(self):
        lo = _leaf_offset(self.config)
        # x0
        feat0 = encode_node(self.builder.nodes[0], 0, self.eval_points, self.config)
        self.assertEqual(feat0[lo], 1.0)
        self.assertEqual(feat0[lo + 1], 0.0)
        # x1
        feat1 = encode_node(self.builder.nodes[1], 1, self.eval_points, self.config)
        self.assertEqual(feat1[lo], 0.0)
        self.assertEqual(feat1[lo + 1], 1.0)
        # const_1
        feat2 = encode_node(self.builder.nodes[2], 2, self.eval_points, self.config)
        self.assertEqual(feat2[lo + 2], 1.0)

    def test_goal_extract_replace_roundtrip(self):
        target = tuple(eval_poly_points(make_var(2, 0), self.eval_points))
        obs = encode_obs(
            self.builder.nodes, target, self.eval_points,
            steps_left=2, max_ops=4, config=self.config,
        )
        goal = extract_goal(obs, self.config)
        self.assertEqual(goal.shape, (self.config.m,))
        # Target is tanh-normalised in obs; compare against normalised expected values.
        scale = self.config.eval_norm_scale
        expected_norm = [math.tanh(float(e) / scale) for e in target]
        np.testing.assert_allclose(goal, expected_norm, rtol=1e-5)

        new_goal = np.ones(self.config.m, dtype=np.float32) * 99.0
        new_obs = replace_goal(obs, new_goal, self.config)
        np.testing.assert_allclose(extract_goal(new_obs, self.config), new_goal)

    def test_get_num_real_nodes(self):
        target = tuple(eval_poly_points(make_var(2, 0), self.eval_points))
        obs = encode_obs(
            self.builder.nodes, target, self.eval_points,
            steps_left=2, max_ops=4, config=self.config,
        )
        self.assertEqual(get_num_real_nodes(obs, self.config), len(self.builder.nodes))


if __name__ == "__main__":
    unittest.main()
