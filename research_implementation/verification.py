"""
Comprehensive verification pipeline for polynomial circuit correctness.

This module provides multiple verification methods to ensure that
generated arithmetic circuits exactly compute the target polynomial:
1. Symbolic verification using SymPy
2. Randomized modular evaluation
3. Floating-point verification with multiple precision levels
4. Structural validation of circuit properties

The verification pipeline is designed to catch all possible errors
and provide high confidence in circuit correctness.
"""

import sympy as sp
import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import hashlib
import time

from State import Game
from generator import create_polynomial_vector


@dataclass
class VerificationResult:
    """Result of circuit verification."""
    is_correct: bool
    verification_time: float
    symbolic_check: Optional[bool] = None
    modular_checks: Optional[List[bool]] = None
    floating_point_checks: Optional[List[bool]] = None
    structural_checks: Optional[Dict[str, bool]] = None
    error_message: Optional[str] = None


class CircuitVerifier:
    """
    Comprehensive circuit verification system.
    
    Uses multiple independent verification methods to ensure
    circuit correctness with high confidence.
    """
    
    def __init__(self, config, index_to_monomial, monomial_to_index):
        """
        Initialize verifier.
        
        Args:
            config: Configuration object
            index_to_monomial: Mapping from index to monomial tuple
            monomial_to_index: Mapping from monomial tuple to index
        """
        self.config = config
        self.index_to_monomial = index_to_monomial
        self.monomial_to_index = monomial_to_index
        self.symbols = sp.symbols([f"x{i}" for i in range(config.n_variables)])
        
        # Verification parameters
        self.num_modular_tests = 10  # Number of different moduli to test
        self.num_random_evaluations = 20  # Number of random point evaluations
        self.moduli_range = (101, 997)  # Range of prime moduli to use
        self.floating_point_tolerance = 1e-10
        
    def verify_circuit(self, game: Game, target_poly_sp: sp.Expr, 
                      target_poly_vec: torch.Tensor) -> VerificationResult:
        """
        Comprehensive verification of circuit correctness.
        
        Args:
            game: Game state containing the constructed circuit
            target_poly_sp: Target polynomial (SymPy expression)
            target_poly_vec: Target polynomial (vector representation)
            
        Returns:
            VerificationResult with detailed verification information
        """
        start_time = time.time()
        
        try:
            # Check if circuit exists
            if not game.exprs:
                return VerificationResult(
                    is_correct=False,
                    verification_time=time.time() - start_time,
                    error_message="No circuit constructed"
                )
            
            circuit_expr = game.exprs[-1]
            
            # 1. Symbolic verification
            symbolic_result = self._verify_symbolic(circuit_expr, target_poly_sp)
            
            # 2. Modular verification
            modular_results = self._verify_modular(circuit_expr, target_poly_sp)
            
            # 3. Floating-point verification
            fp_results = self._verify_floating_point(circuit_expr, target_poly_sp)
            
            # 4. Structural verification
            structural_results = self._verify_structural(game, target_poly_vec)
            
            # 5. Vector representation verification
            vector_result = self._verify_vector_representation(game, target_poly_vec)
            
            # Combine all results
            all_checks_passed = (
                symbolic_result and
                all(modular_results) and
                all(fp_results) and
                all(structural_results.values()) and
                vector_result
            )
            
            verification_time = time.time() - start_time
            
            return VerificationResult(
                is_correct=all_checks_passed,
                verification_time=verification_time,
                symbolic_check=symbolic_result,
                modular_checks=modular_results,
                floating_point_checks=fp_results,
                structural_checks=structural_results
            )
            
        except Exception as e:
            return VerificationResult(
                is_correct=False,
                verification_time=time.time() - start_time,
                error_message=f"Verification failed with error: {str(e)}"
            )
    
    def _verify_symbolic(self, circuit_expr: sp.Expr, target_expr: sp.Expr) -> bool:
        """
        Verify circuit using symbolic computation.
        
        Args:
            circuit_expr: Circuit output expression
            target_expr: Target polynomial expression
            
        Returns:
            True if expressions are symbolically equal
        """
        try:
            # Expand both expressions and check equality
            circuit_expanded = sp.expand(circuit_expr)
            target_expanded = sp.expand(target_expr)
            
            # Check if difference is zero
            difference = sp.expand(circuit_expanded - target_expanded)
            return difference == 0
            
        except Exception as e:
            print(f"Symbolic verification failed: {e}")
            return False
    
    def _verify_modular(self, circuit_expr: sp.Expr, target_expr: sp.Expr) -> List[bool]:
        """
        Verify circuit using modular arithmetic.
        
        Tests equality modulo several different primes by evaluating
        both expressions at random points.
        
        Args:
            circuit_expr: Circuit output expression
            target_expr: Target polynomial expression
            
        Returns:
            List of boolean results for each modular test
        """
        results = []
        
        # Generate list of prime moduli
        primes = self._generate_primes(self.moduli_range[0], self.moduli_range[1])
        test_primes = random.sample(primes, min(len(primes), self.num_modular_tests))
        
        for prime in test_primes:
            try:
                # Test multiple random evaluations for this prime
                prime_passed = True
                
                for _ in range(5):  # 5 random points per prime
                    # Generate random values modulo prime
                    values = {symbol: random.randint(0, prime - 1) 
                            for symbol in self.symbols}
                    
                    # Evaluate both expressions
                    circuit_val = int(circuit_expr.subs(values)) % prime
                    target_val = int(target_expr.subs(values)) % prime
                    
                    if circuit_val != target_val:
                        prime_passed = False
                        break
                
                results.append(prime_passed)
                
            except Exception as e:
                print(f"Modular verification failed for prime {prime}: {e}")
                results.append(False)
        
        return results
    
    def _verify_floating_point(self, circuit_expr: sp.Expr, target_expr: sp.Expr) -> List[bool]:
        """
        Verify circuit using floating-point evaluation.
        
        Args:
            circuit_expr: Circuit output expression
            target_expr: Target polynomial expression
            
        Returns:
            List of boolean results for each floating-point test
        """
        results = []
        
        for _ in range(self.num_random_evaluations):
            try:
                # Generate random floating-point values
                values = {symbol: random.uniform(-10.0, 10.0) 
                         for symbol in self.symbols}
                
                # Evaluate both expressions
                circuit_val = float(circuit_expr.subs(values))
                target_val = float(target_expr.subs(values))
                
                # Check if values are close within tolerance
                diff = abs(circuit_val - target_val)
                max_val = max(abs(circuit_val), abs(target_val), 1.0)
                relative_error = diff / max_val
                
                results.append(relative_error < self.floating_point_tolerance)
                
            except Exception as e:
                print(f"Floating-point verification failed: {e}")
                results.append(False)
        
        return results
    
    def _verify_structural(self, game: Game, target_poly_vec: torch.Tensor) -> Dict[str, bool]:
        """
        Verify structural properties of the circuit.
        
        Args:
            game: Game state containing circuit
            target_poly_vec: Target polynomial vector
            
        Returns:
            Dictionary of structural check results
        """
        checks = {}
        
        try:
            # Check circuit complexity bounds
            checks['complexity_bound'] = len(game.actions_taken) <= self.config.max_complexity
            
            # Check that circuit uses only valid operations
            valid_ops = {'add', 'multiply'}
            checks['valid_operations'] = all(op in valid_ops for op, _, _ in game.actions_taken)
            
            # Check action validity (node references)
            checks['valid_node_refs'] = self._check_valid_node_references(game)
            
            # Check circuit produces expected vector length
            if game.poly_vectors:
                circuit_vec_len = len(game.poly_vectors[-1])
                target_vec_len = len(target_poly_vec)
                checks['vector_length'] = circuit_vec_len == target_vec_len
            else:
                checks['vector_length'] = False
            
            # Check for cycles (shouldn't happen in our DAG construction)
            checks['acyclic'] = self._check_acyclic(game)
            
        except Exception as e:
            print(f"Structural verification failed: {e}")
            # Mark all checks as failed if exception occurs
            checks = {key: False for key in checks}
        
        return checks
    
    def _verify_vector_representation(self, game: Game, target_poly_vec: torch.Tensor) -> bool:
        """
        Verify that the vector representation matches the target.
        
        Args:
            game: Game state containing circuit
            target_poly_vec: Target polynomial vector
            
        Returns:
            True if vector representations match
        """
        try:
            if not game.poly_vectors:
                return False
            
            circuit_vec = torch.tensor(game.poly_vectors[-1], dtype=torch.float)
            target_vec = target_poly_vec.float()
            
            # Ensure same length
            max_len = max(len(circuit_vec), len(target_vec))
            if len(circuit_vec) < max_len:
                circuit_vec = torch.cat([circuit_vec, torch.zeros(max_len - len(circuit_vec))])
            if len(target_vec) < max_len:
                target_vec = torch.cat([target_vec, torch.zeros(max_len - len(target_vec))])
            
            # Check exact equality
            return torch.equal(circuit_vec, target_vec)
            
        except Exception as e:
            print(f"Vector verification failed: {e}")
            return False
    
    def _check_valid_node_references(self, game: Game) -> bool:
        """Check that all node references in actions are valid."""
        try:
            num_base_nodes = self.config.n_variables + 1
            
            for i, (op, node1, node2) in enumerate(game.actions_taken):
                max_valid_node = num_base_nodes + i  # Nodes created so far
                
                if node1 >= max_valid_node or node2 >= max_valid_node:
                    return False
                if node1 < 0 or node2 < 0:
                    return False
            
            return True
        except:
            return False
    
    def _check_acyclic(self, game: Game) -> bool:
        """Check that the circuit forms a DAG (no cycles)."""
        try:
            # In our construction, cycles shouldn't be possible since
            # we only reference previously created nodes
            # This is a sanity check
            
            num_base_nodes = self.config.n_variables + 1
            
            for i, (op, node1, node2) in enumerate(game.actions_taken):
                current_node = num_base_nodes + i
                
                # Check that we only reference earlier nodes
                if node1 >= current_node or node2 >= current_node:
                    return False
            
            return True
        except:
            return False
    
    def _generate_primes(self, min_val: int, max_val: int) -> List[int]:
        """Generate list of prime numbers in given range."""
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = []
        for n in range(min_val, max_val + 1):
            if is_prime(n):
                primes.append(n)
        
        return primes
    
    def quick_verify(self, game: Game, target_poly_sp: sp.Expr) -> bool:
        """
        Quick verification using only symbolic check.
        
        Args:
            game: Game state containing circuit
            target_poly_sp: Target polynomial expression
            
        Returns:
            True if circuit is correct (symbolic check only)
        """
        if not game.exprs:
            return False
        
        try:
            return self._verify_symbolic(game.exprs[-1], target_poly_sp)
        except:
            return False


def verify_circuit_comprehensive(game: Game, target_poly_sp: sp.Expr, 
                                target_poly_vec: torch.Tensor,
                                config, index_to_monomial, monomial_to_index) -> VerificationResult:
    """
    Convenience function for comprehensive circuit verification.
    
    Args:
        game: Game state containing constructed circuit
        target_poly_sp: Target polynomial (SymPy)
        target_poly_vec: Target polynomial (vector)
        config: Configuration object
        index_to_monomial: Monomial index mapping
        monomial_to_index: Monomial to index mapping
        
    Returns:
        VerificationResult with detailed verification information
    """
    verifier = CircuitVerifier(config, index_to_monomial, monomial_to_index)
    return verifier.verify_circuit(game, target_poly_sp, target_poly_vec)


def verify_circuit_quick(game: Game, target_poly_sp: sp.Expr,
                        config, index_to_monomial, monomial_to_index) -> bool:
    """
    Convenience function for quick circuit verification.
    
    Args:
        game: Game state containing constructed circuit
        target_poly_sp: Target polynomial (SymPy)
        config: Configuration object
        index_to_monomial: Monomial index mapping
        monomial_to_index: Monomial to index mapping
        
    Returns:
        True if circuit is correct (symbolic check only)
    """
    verifier = CircuitVerifier(config, index_to_monomial, monomial_to_index)
    return verifier.quick_verify(game, target_poly_sp)


if __name__ == "__main__":
    # Test verification system
    print("Circuit verification system implemented.")
    print("Features:")
    print("- Symbolic verification with SymPy")
    print("- Randomized modular evaluation")
    print("- Floating-point verification")
    print("- Structural validation")
    print("- Vector representation verification")
    print("- Comprehensive and quick verification modes")