"""Top-down polynomial decomposition algorithms.

This subpackage implements an alternative training paradigm for the
arithmetic-circuit synthesis problem. Instead of building circuits bottom-up
from base variables (x_0, x_1, ..., 1) toward the target polynomial, the
agent operates *top-down*: starting from the target, it recursively breaks
the polynomial into smaller sub-polynomials (additively or multiplicatively)
until every leaf of the decomposition tree is a base node.

Two trainers are provided, mirroring the two algorithms used in the
bottom-up setting:

  * ``PPOMCTSBreakdownTrainer``   in ``ppo_mcts_breakdown``
  * ``SACBreakdownTrainer``       in ``sac_breakdown``

The shared decomposition environment lives in ``breakdown_env`` and reuses
the existing ``FactorLibrary`` (over Z, reduced mod p) for guided
multiplicative splits and ``FastPoly`` for fast polynomial arithmetic.

Imports inside this package are kept lazy (this ``__init__`` does not import
torch or the trainers) so that the existing PPO/SAC training loops are not
affected by the addition of this code.
"""
