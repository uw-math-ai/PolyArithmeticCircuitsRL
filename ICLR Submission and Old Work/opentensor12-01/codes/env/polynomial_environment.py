import copy
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from polynomial_env.tensor_env import PolynomialEnvConfig, PolynomialTensorEnv
from polynomial_env.actions import encode_action, decode_action


class PolynomialEnvironment:
    """
    Adapter so the OpenTensor pipeline can operate on polynomial circuits.

    State:
        cur_state: residual tensor padded to [S_size, S_size, S_size] for compatibility.
        hist_actions: last T-1 actions encoded as indices (kept as a fixed-length list).

    Actions:
        Discrete action indices encoding (op, i, j) with encode_action from polynomial_env.actions.
    """

    def __init__(
        self,
        target_poly_expr: sp.Expr,
        n_variables: int = 2,
        max_degree: int = 3,
        max_nodes: int = 6,
        T: int = 1,
        step_penalty: float = -0.1,
        success_reward: float = 10.0,
        failure_penalty: float = -5.0,
        tolerance: float = 1e-6,
    ):
        self.config = PolynomialEnvConfig(
            n_variables=n_variables,
            max_degree=max_degree,
            max_nodes=max_nodes,
            step_penalty=step_penalty,
            success_reward=success_reward,
            failure_penalty=failure_penalty,
            tolerance=tolerance,
        )
        self.S_size = max_degree + 1  # matches tensor axis length
        self.T = max(1, T)
        self.poly_env = PolynomialTensorEnv(target_poly_expr=target_poly_expr, config=self.config)
        self.max_actions = (max_nodes * (max_nodes + 1)) // 2 * 2
        self.reset()

    # ------------------------------------------------------------------ core API
    def reset(self):
        self.poly_env.reset()
        self.cur_state = self._residual_to_state(self.poly_env.observe()["residual_tensor"])
        self.accumulate_reward = 0.0
        self.step_ct = 0
        self.hist_actions: List[int] = [-2 for _ in range(self.T - 1)]  # -2 denotes padding

    def step(self, action_idx: int) -> bool:
        """Apply a gate action. Returns True if terminated."""
        action_idx = int(action_idx)
        obs, reward, done, info = self.poly_env.step(action_idx)
        self.accumulate_reward += reward
        self.step_ct += 1
        self.cur_state = self._residual_to_state(obs["residual_tensor"])
        if self.T > 1:
            self.hist_actions.append(action_idx)
            self.hist_actions = self.hist_actions[-(self.T - 1) :]
        return done

    def is_terminate(self) -> bool:
        return self.poly_env.is_done()

    def get_network_input(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
            tensors: [T, S, S, S] float32
            scalars: [3] float32 (step counts for compatibility)
            action_mask: [max_actions] bool
        """
        tensors = np.zeros((self.T, self.S_size, self.S_size, self.S_size), dtype=np.float32)
        tensors[0] = self.cur_state
        scalars = np.array([self.step_ct, self.step_ct, self.step_ct], dtype=np.float32)

        action_mask = obs_to_mask(self.poly_env)
        return tensors, scalars, action_mask

    # ------------------------------------------------------------------ helpers
    def _residual_to_state(self, residual_tensor: np.ndarray) -> np.ndarray:
        """
        Flatten an n-dimensional residual tensor into a 3D cube [S,S,S].
        For n_variables=2: residual is 2D, pad to 3D
        For n_variables=3: residual is already 3D, return as-is (with padding if needed)
        For n_variables>3: flatten and reshape
        """
        # If already the right shape, return
        if residual_tensor.shape == (self.S_size, self.S_size, self.S_size):
            return residual_tensor.astype(np.float32)
        
        # Flatten and reshape to 3D cube
        flat = residual_tensor.flatten()
        target_size = self.S_size ** 3
        
        # Pad or truncate to fit
        if len(flat) < target_size:
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(flat)] = flat
            flat = padded
        elif len(flat) > target_size:
            flat = flat[:target_size]
        
        return flat.reshape((self.S_size, self.S_size, self.S_size))

    def clone(self) -> "PolynomialEnvironment":
        """Deep copy for MCTS branching."""
        new_env = copy.deepcopy(self)
        return new_env


# ---------------------------------------------------------------------- utils
def obs_to_mask(poly_env: PolynomialTensorEnv) -> np.ndarray:
    """Fetch the available_actions_mask from the wrapped PolynomialTensorEnv."""
    obs = poly_env.observe()
    return obs["available_actions_mask"].astype(bool)
