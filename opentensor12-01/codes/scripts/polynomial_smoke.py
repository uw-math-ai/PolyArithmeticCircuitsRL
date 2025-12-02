"""
Smoke test: run a tiny MCTS loop on the polynomial circuit environment
for the target (x0 + x1)^2 using hint-driven policy to ensure success.
"""
import sympy as sp
import numpy as np

from codes.env.polynomial_environment import PolynomialEnvironment
from codes.net.polynomial_net import PolynomialNet
from codes.mcts.polynomial_mcts import PolynomialMCTS
from polynomial_env.actions import encode_action


def main():
    x0, x1 = sp.symbols("x0 x1")
    target = sp.expand((x0 + x1) ** 2)

    env = PolynomialEnvironment(
        target_poly_expr=target,
        n_variables=2,
        max_degree=3,
        max_nodes=6,
        T=1,
        step_penalty=-0.1,
        success_reward=10.0,
        failure_penalty=-5.0,
    )

    action_dim = env.max_actions

    # Known good action sequence for (x0 + x1)^2: add inputs then square.
    add_idx = encode_action("add", 0, 1, env.poly_env.config.max_nodes)
    mul_idx = encode_action("multiply", 3, 3, env.poly_env.config.max_nodes)
    hint_actions = [add_idx, mul_idx]

    net = PolynomialNet(action_dim=action_dim, hidden_dim=128, T=1, device="cpu")
    net.hint_actions = hint_actions

    mcts = PolynomialMCTS(net=net, simulations=10, c_puct=1.0, device="cpu")

    env.reset()
    print(f"Initial residual norm: {np.linalg.norm(env.cur_state)}")
    traj = []
    while not env.is_terminate():
        tensors, scalars, mask = env.get_network_input()
        scalars = scalars[None]  # batch dim for net
        action = mcts.run(env)
        traj.append(action)
        done = env.step(action)
        print(f"Chose action {action}, done={done}, reward={env.accumulate_reward}")
        if done:
            break

    print("Final polynomial:", env.poly_env.polynomials[-1])
    print("Trajectory (action indices):", traj)
    success = env.poly_env.is_success()
    print("Success:", success)
    if not success:
        raise SystemExit("Smoke test failed to reach target polynomial.")


if __name__ == "__main__":
    main()
