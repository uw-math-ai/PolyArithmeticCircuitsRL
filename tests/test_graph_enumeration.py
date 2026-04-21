import unittest
import warnings

from poly_circuit_rl.env.graph_enumeration import build_game_graph


class TestGraphEnumeration(unittest.TestCase):
    def test_build_game_graph_respects_time_budget(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = build_game_graph(steps=6, num_vars=2, max_seconds=0.0)

        self.assertGreater(graph.number_of_nodes(), 0)
        timeout_warnings = [w for w in caught if "exceeded max_seconds" in str(w.message)]
        self.assertGreaterEqual(len(timeout_warnings), 1)


if __name__ == "__main__":
    unittest.main()
