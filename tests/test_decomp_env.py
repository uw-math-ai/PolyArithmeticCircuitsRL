from decomp_rl.decomp_env import DecompEnv
from decomp_rl.config import DecompEnvConfig
from decomp_rl.polynomial import SparsePolynomial
from decomp_rl.split_proposals import SplitAction


def test_environment_step_solves_factored_branch():
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    target = x * y + y
    action = SplitAction(g=y, h=x * y, source="manual").ordered()

    env = DecompEnv(config=DecompEnvConfig(factor_initial_target=False))
    state = env.reset(target)
    next_state, reward, done, info = env.step(state, 0, action)

    assert done
    assert next_state.frontier == []
    assert reward >= 0
    assert info.children


def test_environment_reset_factors_initial_target_before_agent_action():
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    target = x * y * (x + y)

    env = DecompEnv()
    state = env.reset(target)

    assert state.acc_cost == 2
    assert state.frontier == [x + y]
    assert len(state.history) == 1
    assert state.history[0].action_kind == "factor"


def test_environment_reset_factors_f3_perfect_square_before_agent_action():
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    target = x * x + (x * y).scale(2) + y * y

    env = DecompEnv()
    state = env.reset(target)

    assert state.acc_cost == 1
    assert state.frontier == [x + y]
    assert len(state.history) == 1
    assert state.history[0].action_kind == "factor"
