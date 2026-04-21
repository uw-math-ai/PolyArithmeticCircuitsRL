import unittest

from poly_circuit_rl.core.action_codec import ACTION_SET_OUTPUT, encode_action


class TestActionCodec(unittest.TestCase):
    def test_set_output_bounds_check(self):
        with self.assertRaises(ValueError):
            encode_action(ACTION_SET_OUTPUT, 4, None, L=4)


if __name__ == "__main__":
    unittest.main()
