from pathlib import Path

import pytest

from decomp_rl.config import FactorizerConfig
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.polynomial import SparsePolynomial


def test_factorization_reconstructs_polynomial():
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    poly = x * y + y

    factorizer = FiniteFieldFactorizer()
    factorization = factorizer.factor(poly)

    assert factorization.unit == 1
    assert factorizer.reconstruct(poly) == poly
    factor_keys = {factor.to_key() for factor, _ in factorization.factors}
    assert y.to_key() in factor_keys


def test_sage_backend_if_available():
    cas_python = Path(__file__).resolve().parents[1] / ".cas_env" / "bin" / "python"
    if not cas_python.exists():
        pytest.skip("Sage CAS environment is not available")

    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    poly = x * y + y

    factorizer = FiniteFieldFactorizer(
        FactorizerConfig(backend_name="sage", cas_python_path=str(cas_python))
    )
    factorization = factorizer.factor(poly)
    factorizer.close()

    assert factorization.backend == "sage"
    assert factorizer.reconstruct(poly) == poly
