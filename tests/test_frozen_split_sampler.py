import json
import random
import tempfile
import unittest

from poly_circuit_rl.env.samplers import FrozenSplitSampler


class TestFrozenSplitSampler(unittest.TestCase):
    def _write_jsonl(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with tmp:
            for row in rows:
                tmp.write(json.dumps(row) + "\n")
        return tmp.name

    def test_fails_fast_when_split_has_no_usable_rows(self):
        path = self._write_jsonl([])
        with self.assertRaises(ValueError) as ctx:
            FrozenSplitSampler(path, n_vars=2)
        self.assertIn(path, str(ctx.exception))

    def test_sample_rejects_over_budget_fallback(self):
        path = self._write_jsonl(
            [
                {
                    "expr_str": "x0 + x1",
                    "shortest_length": 2,
                    "poly_key": [
                        [[1, 0], 1, 1],
                        [[0, 1], 1, 1],
                    ],
                }
            ]
        )
        sampler = FrozenSplitSampler(path, n_vars=2)
        with self.assertRaises(ValueError) as ctx:
            sampler.sample(random.Random(0), max_ops=1)
        self.assertIn("shortest_length <= 1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
