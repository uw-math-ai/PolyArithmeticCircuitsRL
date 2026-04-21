# PolyArithmeticCircuitsRL

Arithmetic-circuit synthesis with DQN + HER over exact polynomial targets.

## Module Map

- Canonical training entrypoint: `scripts/train.py`
- Canonical evaluation entrypoint: `scripts/evaluate.py`
- Core package: `poly_circuit_rl/`
- SymPy conversion helpers used by symbolic baselines:
  - `poly_circuit_rl/env/factor.py` (imported by `poly_circuit_rl/baselines/symbolic_utils.py`)
