"""Finite-field polynomial factorization with caching."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import FactorizerConfig
from .polynomial import SparsePolynomial


@dataclass(frozen=True)
class FactorizationResult:
    unit: int
    factors: tuple[tuple[SparsePolynomial, int], ...]
    backend: str = "sympy"

    def iter_unique_factors(self) -> tuple[SparsePolynomial, ...]:
        seen: dict[str, SparsePolynomial] = {}
        for factor, _ in self.factors:
            seen.setdefault(factor.to_key(), factor)
        return tuple(seen.values())


class FiniteFieldFactorizer:
    """Wrap CAS-backed factorization and normalize results for reuse."""

    def __init__(
        self,
        config: FactorizerConfig | None = None,
        library=None,  # FactorizableLibrary | None
    ) -> None:
        self.config = config or FactorizerConfig()
        self.library = library
        self._cache: dict[str, FactorizationResult] = {}
        self._sage_worker: _SageFactorWorker | None = None
        self.cache_requests = 0
        self.cache_hits = 0
        atexit.register(self.close)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def factor(self, poly: SparsePolynomial) -> FactorizationResult:
        key = poly.to_key()
        self.cache_requests += 1
        if self.config.cache_enabled and key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        result = self._factor_uncached(poly)
        if self.config.cache_enabled:
            self._cache[key] = result
        if self.library is not None and not poly.is_zero and not poly.is_constant:
            try:
                from .baseline_cost import BaselineCostModel
                direct = BaselineCostModel().direct_construction_cost(poly)
                self.library.maybe_add_from_factorization(poly, result, direct)
            except Exception:
                pass
        return result

    def _factor_uncached(self, poly: SparsePolynomial) -> FactorizationResult:
        if poly.is_zero:
            return FactorizationResult(unit=0, factors=(), backend=self._resolve_backend_name())
        if poly.is_constant:
            constant = poly.terms[0][0] if poly.terms else 0
            return FactorizationResult(
                unit=constant % poly.p,
                factors=(),
                backend=self._resolve_backend_name(),
            )

        backend_name = self._resolve_backend_name()
        if backend_name == "sage":
            return self._factor_via_sage(poly)
        return self._factor_via_sympy(poly)

    def _resolve_backend_name(self) -> str:
        if self.config.backend_name != "auto":
            return self.config.backend_name
        return "sage" if self._sage_python_available() else "sympy"

    def _sage_python_available(self) -> bool:
        if not self.config.cas_python_path:
            return False
        return Path(self.config.cas_python_path).exists()

    def _factor_via_sage(self, poly: SparsePolynomial) -> FactorizationResult:
        worker = self._ensure_sage_worker()
        result = worker.factor(poly)
        if self._reconstruct_from_result(poly, result) != poly:
            raise ValueError("Sage backend returned factors that do not reconstruct the polynomial")
        return result

    def _ensure_sage_worker(self) -> "_SageFactorWorker":
        if not self._sage_python_available():
            raise RuntimeError(
                "Sage backend requested but no CAS Python executable was found. "
                "Expected .cas_env/bin/python or an explicit FactorizerConfig.cas_python_path."
            )
        if self._sage_worker is None:
            self._sage_worker = _SageFactorWorker(
                python_executable=self.config.cas_python_path or "",
                startup_timeout_sec=self.config.helper_startup_timeout_sec,
            )
        return self._sage_worker

    def _factor_via_sympy(self, poly: SparsePolynomial) -> FactorizationResult:
        try:
            from sympy import Poly, symbols
        except ImportError as exc:
            raise RuntimeError(
                "SymPy is required for finite-field factorization. Install the project dependencies."
            ) from exc

        symbols_tuple = symbols(poly.variables)
        sympy_poly = Poly(poly.to_sympy_expr(), *symbols_tuple, modulus=poly.p)
        backend = "sympy"
        try:
            unit, raw_factors = sympy_poly.factor_list()
        except NotImplementedError:
            from sympy import factor_list

            unit, raw_factors = factor_list(sympy_poly.as_expr())
            backend = f"{self.config.backend_name}-integer-fallback"
        unit = int(unit) % poly.p
        factors: list[tuple[SparsePolynomial, int]] = []
        for factor_poly, exponent in raw_factors:
            factor_expr = factor_poly.as_expr() if hasattr(factor_poly, "as_expr") else factor_poly
            sparse_factor = SparsePolynomial.from_sympy_expr(factor_expr, prime=poly.p, variables=poly.variables)
            monic_factor, extracted_unit = sparse_factor.make_monic()
            unit = (unit * extracted_unit) % poly.p
            factors.append((monic_factor, int(exponent)))

        if unit == 0:
            raise ValueError("Non-zero polynomial unexpectedly factored with zero unit")
        result = FactorizationResult(unit=unit, factors=tuple(factors), backend=backend)
        if self._reconstruct_from_result(poly, result) != poly:
            raise ValueError("Factorization backend returned factors that do not reconstruct the polynomial")
        return result

    def reconstruct(self, poly: SparsePolynomial) -> SparsePolynomial:
        return self._reconstruct_from_result(poly, self.factor(poly))

    def _reconstruct_from_result(
        self,
        reference: SparsePolynomial,
        factorization: FactorizationResult,
    ) -> SparsePolynomial:
        result = SparsePolynomial.one(reference.p, reference.variables)
        if factorization.unit == 0:
            return SparsePolynomial.zero(reference.p, reference.variables)
        result = result.scale(factorization.unit)
        for factor, exponent in factorization.factors:
            result = result * factor.pow(exponent)
        return result

    def clear(self) -> None:
        self._cache.clear()
        self.reset_stats()

    def reset_stats(self) -> None:
        self.cache_requests = 0
        self.cache_hits = 0

    def close(self) -> None:
        if self._sage_worker is not None:
            self._sage_worker.close()
            self._sage_worker = None


class _SageFactorWorker:
    """Persistent Sage subprocess speaking a small JSON lines protocol."""

    def __init__(self, python_executable: str, startup_timeout_sec: float = 30.0) -> None:
        self.python_executable = python_executable
        self.startup_timeout_sec = startup_timeout_sec
        self.helper_path = Path(__file__).with_name("sage_factor_worker.py")
        self.process = self._start()

    def _start(self) -> subprocess.Popen[str]:
        env = os.environ.copy()
        python_dir = str(Path(self.python_executable).resolve().parent)
        env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")
        repo_root = Path(__file__).resolve().parents[2]
        sage_home = repo_root / ".sage_home"
        xdg_cache_home = sage_home / ".cache"
        sage_home.mkdir(parents=True, exist_ok=True)
        xdg_cache_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(sage_home)
        env["XDG_CACHE_HOME"] = str(xdg_cache_home)
        process = subprocess.Popen(
            [self.python_executable, "-u", str(self.helper_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        ready_line = process.stdout.readline() if process.stdout is not None else ""
        if not ready_line:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"Failed to start Sage factor worker: {stderr.strip()}")
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"Sage factor worker failed to initialize: {ready}")
        return process

    def factor(self, poly: SparsePolynomial) -> FactorizationResult:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Sage worker pipes are unavailable")
        request = {
            "prime": poly.p,
            "variables": list(poly.variables),
            "terms": [[coeff, list(exponent)] for coeff, exponent in poly.terms],
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError(f"Sage worker exited unexpectedly: {stderr.strip()}")
        response = json.loads(response_line)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Unknown Sage factorization error"))
        factors = tuple(
            (
                SparsePolynomial.from_terms(
                    tuple(
                        (int(coeff), tuple(int(v) for v in exponent))
                        for coeff, exponent in factor_payload["terms"]
                    ),
                    prime=poly.p,
                    variables=poly.variables,
                ),
                int(factor_payload["exponent"]),
            )
            for factor_payload in response["factors"]
        )
        return FactorizationResult(
            unit=int(response["unit"]) % poly.p,
            factors=factors,
            backend=str(response.get("backend", "sage")),
        )

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            self.process.wait(timeout=2)
