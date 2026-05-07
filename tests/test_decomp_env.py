from decomp_rl.decomp_env import DecompEnv
from decomp_rl.polynomial import SparsePolynomial
from decomp_rl.split_proposals import SplitAction


def test_environment_step_solves_factored_branch():
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    target = x * y + y
    action = SplitAction(g=y, h=x * y, source="manual").ordered()

    env = DecompEnv()
    state = env.reset(target)
    next_state, reward, done, info = env.step(state, 0, action)

    assert done
    assert next_state.frontier == []
    assert reward >= 0
    assert info.children

