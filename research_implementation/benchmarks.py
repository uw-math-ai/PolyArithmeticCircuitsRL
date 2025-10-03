"""
Structured benchmark suite for polynomial circuit construction.

This module provides generators for well-known polynomial families
that serve as standard benchmarks in algebraic complexity theory:
- Elementary symmetric polynomials
- Determinants of small matrices
- Power sum polynomials
- Chebyshev polynomials
- Other structured polynomial families

These benchmarks allow comparison against known optimal constructions
and provide standardized test cases for circuit synthesis algorithms.
"""

import sympy as sp
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from itertools import combinations, product

from generator import generate_monomials_with_additive_indices


class PolynomialBenchmarks:
    """
    Collection of structured polynomial benchmark generators.

    Each method generates both the symbolic representation (SymPy)
    and vector representation for use with the circuit construction system.
    """

    def __init__(self, config, index_to_monomial, monomial_to_index):
        """
        Initialize benchmark generator.

        Args:
            config: Configuration object with n_variables, max_complexity, mod
            index_to_monomial: Mapping from index to monomial tuple
            monomial_to_index: Mapping from monomial tuple to index
        """
        self.config = config
        self.index_to_monomial = index_to_monomial
        self.monomial_to_index = monomial_to_index
        self.symbols = sp.symbols([f"x{i}" for i in range(config.n_variables)])

    def sympy_to_vector(self, poly_sp: sp.Expr) -> torch.Tensor:
        """
        Convert SymPy polynomial to vector representation.

        Args:
            poly_sp: SymPy polynomial expression

        Returns:
            Vector representation of the polynomial
        """
        poly_expanded = sp.expand(poly_sp)
        vector_length = len(self.index_to_monomial)
        vector = torch.zeros(vector_length, dtype=torch.float)

        poly_dict = poly_expanded.as_coefficients_dict()

        for term, coeff in poly_dict.items():
            exponents = [0] * self.config.n_variables

            if term.is_number:
                # Constant term
                pass
            else:
                powers = term.as_powers_dict()
                for i, symbol in enumerate(self.symbols):
                    exponents[i] = powers.get(symbol, 0)

            exponent_tuple = tuple(exponents)
            if exponent_tuple in self.monomial_to_index:
                idx = self.monomial_to_index[exponent_tuple]
                vector[idx] = float(coeff % self.config.mod)
            else:
                print(f"Warning: Monomial {exponent_tuple} not in index mapping")

        return vector

    def elementary_symmetric(self, k: int) -> Tuple[sp.Expr, torch.Tensor]:
        """
        Generate k-th elementary symmetric polynomial.

        e_k(x_0, ..., x_{n-1}) = sum of all products of k distinct variables

        Examples:
        - e_1(x_0, x_1, x_2) = x_0 + x_1 + x_2
        - e_2(x_0, x_1, x_2) = x_0*x_1 + x_0*x_2 + x_1*x_2
        - e_3(x_0, x_1, x_2) = x_0*x_1*x_2

        Args:
            k: Degree of elementary symmetric polynomial (1 <= k <= n_variables)

        Returns:
            Tuple of (SymPy expression, vector representation)
        """
        if k < 1 or k > self.config.n_variables:
            raise ValueError(f"k must be between 1 and {self.config.n_variables}")

        # Generate all combinations of k variables
        poly_sp = sp.sympify(0)
        for var_combination in combinations(self.symbols, k):
            # Product of variables in this combination
            term = sp.sympify(1)
            for var in var_combination:
                term *= var
            poly_sp += term

        poly_vec = self.sympy_to_vector(poly_sp)
        return poly_sp, poly_vec

    def power_sum(self, k: int) -> tuple[sp.Expr, torch.Tensor]:
        """
        Generate k-th power sum polynomial.

        p_k(x_0, ..., x_{n-1}) = x_0^k + x_1^k + ... + x_{n-1}^k

        Args:
            k: Power degree (k >= 1)

        Returns:
            Tuple of (SymPy expression, vector representation)
        """
        if k < 1:
            raise ValueError("Power k must be >= 1")

        poly_sp = sp.Expr(sum(var**k for var in self.symbols))
        poly_vec = self.sympy_to_vector(poly_sp)
        return poly_sp, poly_vec

    def determinant_2x2(self) -> Tuple[sp.Expr, torch.Tensor]:
        """
        Generate determinant of 2x2 matrix with variables.

        det([[x_0, x_1], [x_2, x_3]]) = x_0*x_3 - x_1*x_2

        Requires n_variables >= 4.

        Returns:
            Tuple of (SymPy expression, vector representation)
        """
        if self.config.n_variables < 4:
            raise ValueError("Determinant 2x2 requires at least 4 variables")

        x0, x1, x2, x3 = self.symbols[:4]
        poly_sp = x0 * x3 - x1 * x2
        poly_vec = self.sympy_to_vector(poly_sp)
        return poly_sp, poly_vec

    def determinant_3x3(self) -> Tuple[sp.Expr, torch.Tensor]:
        """
        Generate determinant of 3x3 matrix with variables.

        det([[x_0, x_1, x_2],
             [x_3, x_4, x_5],
             [x_6, x_7, x_8]]) =
        x_0*(x_4*x_8 - x_5*x_7) - x_1*(x_3*x_8 - x_5*x_6) + x_2*(x_3*x_7 - x_4*x_6)

        Requires n_variables >= 9.

        Returns:
            Tuple of (SymPy expression, vector representation)
        """
        if self.config.n_variables < 9:
            raise ValueError("Determinant 3x3 requires at least 9 variables")

        # Create 3x3 matrix
        matrix = sp.Matrix(3, 3, self.symbols[:9])
        poly_sp = matrix.det()
        poly_vec = self.sympy_to_vector(poly_sp)
        return poly_sp, poly_vec

    def chebyshev_first_kind(self, n: int) -> Tuple[sp.Expr, torch.Tensor]:
        """
        Generate Chebyshev polynomial of the first kind T_n(x).

        Uses the recurrence relation:
        T_0(x) = 1
        T_1(x) = x
        T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)

        Args:
            n: Degree of Chebyshev polynomial

        Returns:
            Tuple of (SymPy expression, vector representation)
        """
        if n < 0:
            raise ValueError("Chebyshev degree must be >= 0")
        if self.config.n_variables < 1:
            raise ValueError("Need at least 1 variable for Chebyshev polynomial")

        x = self.symbols[0]

        if n == 0:
            poly_sp = sp.sympify(1)
        elif n == 1:
            poly_sp = x
        else:
            # Use recurrence relation
            T_prev_prev = sp.sympify(1)  # T_0
            T_prev = x  # T_1

            for i in range(2, n + 1):
                T_curr = sp.expand(2 * x * T_prev - T_prev_prev)
                T_prev_prev = T_prev
                T_prev = T_curr

            poly_sp = T_prev

        poly_vec = self.sympy_to_vector(poly_sp)
        return poly_sp, poly_vec

    def vandermonde_determinant(self, degree: int = 2) -> Tuple[sp.Expr, torch.Tensor]:
        """
        Generate Vandermonde determinant polynomial.

        For variables x_0, x_1, ..., x_{n-1}, creates the Vandermonde matrix
        and computes its determinant.

        Args:
            degree: Maximum degree in Vandermonde matrix

        Returns:
            Tuple of (SymPy expression, vector representation)
        """
        if degree < 1:
            raise ValueError("Vandermonde degree must be >= 1")

        n = self.config.n_variables
        if n < 2:
            raise ValueError("Need at least 2 variables for Vandermonde determinant")

        # Create Vandermonde matrix: V[i,j] = x_i^j
        matrix_entries = []
        for i in range(n):
            row = []
            for j in range(min(n, degree + 1)):
                row.append(self.symbols[i] ** j)
            matrix_entries.append(row)

        # Truncate to square matrix
        size = min(n, degree + 1)
        matrix = sp.Matrix(
            size, size, [matrix_entries[i][j] for i in range(size) for j in range(size)]
        )

        poly_sp = matrix.det()
        poly_vec = self.sympy_to_vector(poly_sp)
        return poly_sp, poly_vec

    def monomial_product(self, exponents: List[int]) -> Tuple[sp.Expr, torch.Tensor]:
        """
        Generate product of monomials: x_0^e_0 * x_1^e_1 * ... * x_{k-1}^e_{k-1}

        Args:
            exponents: List of exponents for each variable

        Returns:
            Tuple of (SymPy expression, vector representation)
        """
        if len(exponents) > self.config.n_variables:
            raise ValueError("Too many exponents for available variables")

        poly_sp = sp.sympify(1)
        for i, exp in enumerate(exponents):
            if exp > 0:
                poly_sp *= self.symbols[i] ** exp

        poly_vec = self.sympy_to_vector(poly_sp)
        return poly_sp, poly_vec

    def random_sparse_polynomial(
        self, num_terms: int = 3, max_degree: int = 2
    ) -> Tuple[sp.Expr, torch.Tensor]:
        """
        Generate a random sparse polynomial with specified number of terms.

        Args:
            num_terms: Number of monomial terms
            max_degree: Maximum total degree of any term

        Returns:
            Tuple of (SymPy expression, vector representation)
        """
        poly_sp = sp.sympify(0)

        for _ in range(num_terms):
            # Generate random exponents
            total_degree = np.random.randint(0, max_degree + 1)
            exponents = [0] * self.config.n_variables

            if total_degree > 0:
                # Distribute degree among variables
                for _ in range(total_degree):
                    var_idx = np.random.randint(0, self.config.n_variables)
                    exponents[var_idx] += 1

            # Generate random coefficient
            coeff = np.random.randint(1, self.config.mod)

            # Create term
            term = sp.sympify(coeff)
            for i, exp in enumerate(exponents):
                if exp > 0:
                    term *= self.symbols[i] ** exp

            poly_sp += term

        poly_sp = sp.expand(poly_sp)
        poly_vec = self.sympy_to_vector(poly_sp)
        return poly_sp, poly_vec

    def get_all_benchmarks(self) -> List[Tuple[str, sp.Expr, torch.Tensor]]:
        """
        Get all available benchmarks that are feasible with current configuration.

        Returns:
            List of (name, sympy_expression, vector) tuples
        """
        benchmarks = []

        # Elementary symmetric polynomials
        for k in range(1, min(self.config.n_variables + 1, 4)):  # Up to e_3
            try:
                poly_sp, poly_vec = self.elementary_symmetric(k)
                benchmarks.append((f"elementary_symmetric_{k}", poly_sp, poly_vec))
            except:
                pass

        # Power sums
        for k in range(1, 4):  # p_1, p_2, p_3
            try:
                poly_sp, poly_vec = self.power_sum(k)
                benchmarks.append((f"power_sum_{k}", poly_sp, poly_vec))
            except:
                pass

        # Determinants
        if self.config.n_variables >= 4:
            try:
                poly_sp, poly_vec = self.determinant_2x2()
                benchmarks.append(("determinant_2x2", poly_sp, poly_vec))
            except:
                pass

        if self.config.n_variables >= 9:
            try:
                poly_sp, poly_vec = self.determinant_3x3()
                benchmarks.append(("determinant_3x3", poly_sp, poly_vec))
            except:
                pass

        # Chebyshev polynomials
        for n in range(2, 5):  # T_2, T_3, T_4
            try:
                poly_sp, poly_vec = self.chebyshev_first_kind(n)
                benchmarks.append((f"chebyshev_T{n}", poly_sp, poly_vec))
            except:
                pass

        # Vandermonde determinant
        if self.config.n_variables >= 2:
            try:
                poly_sp, poly_vec = self.vandermonde_determinant(2)
                benchmarks.append(("vandermonde_det", poly_sp, poly_vec))
            except:
                pass

        # Some specific monomials
        try:
            poly_sp, poly_vec = self.monomial_product([2, 1])  # x_0^2 * x_1
            benchmarks.append(("monomial_x0_2_x1", poly_sp, poly_vec))
        except:
            pass

        try:
            poly_sp, poly_vec = self.monomial_product([1, 1, 1])  # x_0 * x_1 * x_2
            benchmarks.append(("monomial_x0_x1_x2", poly_sp, poly_vec))
        except:
            pass

        return benchmarks


def generate_benchmark_dataset(
    config, size: int = 100
) -> List[Tuple[str, sp.Expr, torch.Tensor]]:
    """
    Generate a dataset of benchmark polynomials.

    Args:
        config: Configuration object
        size: Number of benchmark instances to generate

    Returns:
        List of (name, sympy_expression, vector) tuples
    """
    # Generate monomial mappings
    n, d = config.n_variables, config.max_complexity * 2
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(
        n, d
    )

    # Create benchmark generator
    benchmarks = PolynomialBenchmarks(config, index_to_monomial, monomial_to_index)

    # Get all structured benchmarks
    dataset = benchmarks.get_all_benchmarks()

    # Add random sparse polynomials to reach desired size
    while len(dataset) < size:
        try:
            poly_sp, poly_vec = benchmarks.random_sparse_polynomial()
            name = f"random_sparse_{len(dataset)}"
            dataset.append((name, poly_sp, poly_vec))
        except:
            break

    return dataset[:size]


def print_benchmark_info(config):
    """Print information about available benchmarks."""
    # Generate monomial mappings
    n, d = config.n_variables, config.max_complexity * 2
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(
        n, d
    )

    # Create benchmark generator
    benchmarks = PolynomialBenchmarks(config, index_to_monomial, monomial_to_index)

    print(
        f"Benchmark Suite for n_variables={config.n_variables}, max_complexity={config.max_complexity}"
    )
    print("=" * 70)

    available_benchmarks = benchmarks.get_all_benchmarks()

    for name, poly_sp, poly_vec in available_benchmarks:
        print(f"{name:25}: {poly_sp}")

    print(f"\nTotal available benchmarks: {len(available_benchmarks)}")


if __name__ == "__main__":
    # Test with sample configuration
    class TestConfig:
        def __init__(self):
            self.n_variables = 3
            self.max_complexity = 5
            self.mod = 50

    config = TestConfig()
    print_benchmark_info(config)
