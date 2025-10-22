"""
Integration module for MCTS with transformer-based circuit construction.

This module bridges the MCTS implementation with the existing transformer
model and provides enhanced training and evaluation capabilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Optional
import time
import copy
from dataclasses import dataclass

from fourthGen import CircuitBuilder, Config
from mcts import MCTS, MCTSNode, mcts_self_play_game
from benchmarks import PolynomialBenchmarks, generate_benchmark_dataset
from verification import CircuitVerifier, VerificationResult
from State import Game
from generator import generate_monomials_with_additive_indices
from utils import vector_to_sympy


@dataclass
class MCTSConfig:
    """Configuration for MCTS-enhanced system."""

    mcts_simulations: int = 800
    c_puct: float = 1.0
    temperature: float = 1.0
    training_temperature: float = 1.0
    evaluation_temperature: float = 0.1


@dataclass
class SearchResult:
    """Result of MCTS search."""

    success: bool
    circuit_found: bool
    num_actions: int
    search_time: float
    verification_result: Optional[VerificationResult] = None
    final_expression: Optional[sp.Expr] = None
    mcts_simulations_used: int = 0


class MCTSCircuitSolver:
    """
    Enhanced circuit solver using MCTS with transformer guidance.

    Combines the neural network policy and value functions with
    Monte Carlo Tree Search for improved circuit construction.
    """

    def __init__(
        self, model: CircuitBuilder, config: Config, mcts_config: MCTSConfig = None
    ):
        """
        Initialize MCTS-enhanced solver.

        Args:
            model: Trained CircuitBuilder model
            config: System configuration
            mcts_config: MCTS-specific configuration
        """
        self.model = model
        self.config = config
        self.mcts_config = mcts_config or MCTSConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate monomial mappings
        n, d = config.n_variables, config.max_complexity * 2
        self.index_to_monomial, self.monomial_to_index, _ = (
            generate_monomials_with_additive_indices(n, d)
        )

        # Initialize verifier
        self.verifier = CircuitVerifier(
            config, self.index_to_monomial, self.monomial_to_index
        )

        # Initialize benchmark generator
        self.benchmarks = PolynomialBenchmarks(
            config, self.index_to_monomial, self.monomial_to_index
        )

    def solve_polynomial(
        self,
        target_poly_sp: sp.Expr,
        target_poly_vec: torch.Tensor,
        max_time: float = 60.0,
    ) -> SearchResult:
        """
        Solve a single polynomial using MCTS-guided search.

        Args:
            target_poly_sp: Target polynomial (SymPy)
            target_poly_vec: Target polynomial (vector)
            max_time: Maximum search time in seconds

        Returns:
            SearchResult with solution details
        """
        start_time = time.time()

        # Initialize game
        game = Game(
            target_poly_sp,
            target_poly_vec.unsqueeze(0),
            self.config,
            self.index_to_monomial,
            self.monomial_to_index,
        )

        # Initialize MCTS
        mcts = MCTS(
            self.model,
            self.config,
            c_puct=self.mcts_config.c_puct,
            num_simulations=self.mcts_config.mcts_simulations,
        )

        simulations_used = 0

        while not game.is_done() and (time.time() - start_time) < max_time:
            # Run MCTS search
            root = mcts.search(game)
            simulations_used += self.mcts_config.mcts_simulations

            # Select action
            action = mcts.select_action(root, self.mcts_config.evaluation_temperature)

            # Take action
            game.take_action(action)

        search_time = time.time() - start_time

        # Verify result
        verification_result = None
        circuit_found = False
        final_expression = None

        if game.exprs:
            final_expression = game.exprs[-1]
            verification_result = self.verifier.verify_circuit(
                game, target_poly_sp, target_poly_vec
            )
            circuit_found = verification_result.is_correct

        success = (
            circuit_found and len(game.actions_taken) <= self.config.max_complexity
        )

        return SearchResult(
            success=success,
            circuit_found=circuit_found,
            num_actions=len(game.actions_taken),
            search_time=search_time,
            verification_result=verification_result,
            final_expression=final_expression,
            mcts_simulations_used=simulations_used,
        )

    def evaluate_on_benchmarks(
        self, benchmark_names: List[str] = None, max_time_per_problem: float = 30.0
    ) -> Dict[str, SearchResult]:
        """
        Evaluate MCTS solver on benchmark problems.

        Args:
            benchmark_names: List of benchmark names to test (None for all)
            max_time_per_problem: Maximum time per problem in seconds

        Returns:
            Dictionary mapping benchmark names to results
        """
        all_benchmarks = self.benchmarks.get_all_benchmarks()

        if benchmark_names is not None:
            benchmarks = [
                (name, poly_sp, poly_vec)
                for name, poly_sp, poly_vec in all_benchmarks
                if name in benchmark_names
            ]
        else:
            benchmarks = all_benchmarks

        results = {}

        print(f"Evaluating MCTS solver on {len(benchmarks)} benchmarks...")
        print("=" * 60)

        for name, poly_sp, poly_vec in benchmarks:
            print(f"Testing {name}: {poly_sp}")

            result = self.solve_polynomial(poly_sp, poly_vec, max_time_per_problem)
            results[name] = result

            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            print(
                f"  {status} | Actions: {result.num_actions} | Time: {result.search_time:.2f}s"
            )

            if result.verification_result and not result.verification_result.is_correct:
                print(
                    f"  Verification failed: {result.verification_result.error_message}"
                )

            print()

        # Print summary
        successful = sum(1 for r in results.values() if r.success)
        total = len(results)
        avg_time = np.mean([r.search_time for r in results.values()])
        avg_actions = np.mean(
            [r.num_actions for r in results.values() if r.circuit_found]
        )

        print(f"Summary: {successful}/{total} solved ({100 * successful / total:.1f}%)")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Average actions (successful): {avg_actions:.1f}")

        return results

    def generate_training_data(self, num_games: int = 100) -> List[Tuple]:
        """
        Generate self-play training data using MCTS.

        Args:
            num_games: Number of self-play games to generate

        Returns:
            List of training examples (states, action_probs, values)
        """
        print(f"Generating {num_games} MCTS self-play games...")

        training_data = []
        successful_games = 0

        for i in range(num_games):
            if i % 10 == 0:
                print(f"Game {i}/{num_games} (successful: {successful_games})")

            # Generate random polynomial
            try:
                poly_sp, poly_vec = self.benchmarks.random_sparse_polynomial()

                states, action_probs, game_result = mcts_self_play_game(
                    self.model,
                    self.config,
                    poly_sp,
                    poly_vec,
                    self.index_to_monomial,
                    self.monomial_to_index,
                    mcts_simulations=self.mcts_config.mcts_simulations,
                    temperature=self.mcts_config.training_temperature,
                )

                # Store training examples
                for state, action_prob in zip(states, action_probs):
                    training_data.append((state, action_prob, game_result))

                if game_result > 0:  # Successful game
                    successful_games += 1

            except Exception as e:
                print(f"Game {i} failed: {e}")
                continue

        print(
            f"Generated {len(training_data)} training examples from {successful_games} successful games"
        )
        return training_data

    def interactive_solver(self):
        """
        Interactive mode for testing polynomial solving.
        """
        print("MCTS-Enhanced Circuit Solver")
        print("=" * 40)
        print("Enter polynomials using variables x0, x1, x2, ...")
        print("Examples: x0 + x1, x0*x1 + x0*x2, x0**2 + x1**2")
        print("Type 'quit' to exit, 'benchmarks' to see available benchmarks")
        print()

        while True:
            try:
                user_input = input("Enter polynomial: ").strip()

                if user_input.lower() == "quit":
                    break

                if user_input.lower() == "benchmarks":
                    benchmarks = self.benchmarks.get_all_benchmarks()
                    print("\nAvailable benchmarks:")
                    for name, poly_sp, _ in benchmarks:
                        print(f"  {name}: {poly_sp}")
                    print()
                    continue

                # Parse polynomial
                symbols_dict = {
                    f"x{i}": self.benchmarks.symbols[i]
                    for i in range(self.config.n_variables)
                }

                # Try to parse as benchmark name first
                benchmarks = self.benchmarks.get_all_benchmarks()
                benchmark_dict = {
                    name: (poly_sp, poly_vec) for name, poly_sp, poly_vec in benchmarks
                }

                if user_input in benchmark_dict:
                    poly_sp, poly_vec = benchmark_dict[user_input]
                    print(f"Using benchmark: {poly_sp}")
                else:
                    # Parse as polynomial expression
                    poly_sp = sp.sympify(user_input, locals=symbols_dict)
                    poly_vec = self.benchmarks.sympy_to_vector(poly_sp)

                print(f"Target: {poly_sp}")
                print("Searching...")

                # Solve using MCTS
                result = self.solve_polynomial(poly_sp, poly_vec)

                print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
                print(f"Actions taken: {result.num_actions}")
                print(f"Search time: {result.search_time:.2f}s")
                print(f"MCTS simulations: {result.mcts_simulations_used}")

                if result.final_expression:
                    print(f"Circuit output: {result.final_expression}")

                if result.verification_result:
                    vr = result.verification_result
                    print(f"Verification time: {vr.verification_time:.3f}s")
                    print(f"Symbolic check: {vr.symbolic_check}")
                    if vr.modular_checks:
                        modular_success = sum(vr.modular_checks)
                        print(
                            f"Modular checks: {modular_success}/{len(vr.modular_checks)}"
                        )
                    if vr.floating_point_checks:
                        fp_success = sum(vr.floating_point_checks)
                        print(
                            f"Floating-point checks: {fp_success}/{len(vr.floating_point_checks)}"
                        )

                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print()


def load_model_and_create_solver(
    model_path: str, config: Config = None, mcts_config: MCTSConfig = None
) -> MCTSCircuitSolver:
    """
    Load trained model and create MCTS solver.

    Args:
        model_path: Path to trained model checkpoint
        config: System configuration (None to use default)
        mcts_config: MCTS configuration (None to use default)

    Returns:
        Configured MCTSCircuitSolver
    """
    if config is None:
        config = Config()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate monomial mappings for model initialization
    n, d = config.n_variables, config.max_complexity * 2
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(
        n, d
    )

    # Initialize model
    model = CircuitBuilder(config, len(index_to_monomial))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create solver
    solver = MCTSCircuitSolver(model, config, mcts_config)

    return solver


if __name__ == "__main__":
    # Test integration
    print("MCTS-Transformer integration module implemented.")
    print("Features:")
    print("- MCTS-guided polynomial solving")
    print("- Benchmark evaluation")
    print("- Self-play training data generation")
    print("- Interactive solver mode")
    print("- Comprehensive verification integration")
