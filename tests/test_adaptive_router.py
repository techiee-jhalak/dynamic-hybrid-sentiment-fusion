"""Unit tests for the Noise-Aware Adaptive Routing Module."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import math

from src.models.adaptive_router import (
    AdaptiveRouter,
    RoutingDecision,
    stable_sigmoid,
)


class TestAdaptiveRouter(unittest.TestCase):
    """Test suite verifying exact routing formulas, clamping invariants, and boundary conditions."""

    def setUp(self):
        self.router = AdaptiveRouter()

    def test_numerically_stable_sigmoid(self):
        """Verify sigmoid stability on extreme positive and negative inputs."""
        self.assertAlmostEqual(stable_sigmoid(0.0), 0.5, places=6)
        self.assertAlmostEqual(stable_sigmoid(100.0), 1.0, places=6)
        self.assertAlmostEqual(stable_sigmoid(-100.0), 0.0, places=6)

    def test_mandatory_boundary_case_n_zero(self):
        """Test N = 0: Must route to minimal alpha 0.02."""
        dec = self.router.route(noise_score=0.0, token_length=20)
        self.assertEqual(dec.alpha, 0.02)
        self.assertEqual(dec.routing_state, "low_noise_default")
        self.assertEqual(dec.noise_score, 0.0)

    def test_mandatory_boundary_case_n_threshold_point_twenty(self):
        """Test N = 0.20: Boundary threshold point -> alpha must be exactly 0.02."""
        dec = self.router.route(noise_score=0.20, token_length=20)
        self.assertEqual(dec.alpha, 0.02)
        self.assertEqual(dec.routing_state, "low_noise_default")
        self.assertEqual(dec.noise_score, 0.20)

    def test_mandatory_boundary_case_n_above_threshold_point_twenty_zero_zero_zero_one(self):
        """Test N = 0.200001: Just above threshold -> triggers active adaptive calculation."""
        dec = self.router.route(noise_score=0.200001, token_length=20)
        # z = 0.05*(20-20) + 12.0*0.200001 = 2.400012
        # alpha_raw = sigmoid(2.400012) ~ 0.9168
        # alpha = clamp(0.9168, 0.02, 0.25) = 0.25
        self.assertGreater(dec.alpha_raw, 0.90)
        self.assertEqual(dec.alpha, 0.25)
        self.assertEqual(dec.routing_state, "clamped_max")

    def test_mandatory_case_n_half(self):
        """Test N = 0.5."""
        dec = self.router.route(noise_score=0.5, token_length=20)
        # z = 0.05*0 + 12.0*0.5 = 6.0
        # alpha_raw = sigmoid(6.0) ~ 0.9975
        # alpha = 0.25
        self.assertEqual(dec.alpha, 0.25)
        self.assertGreater(dec.alpha_raw, 0.99)
        self.assertGreaterEqual(dec.alpha, 0.02)
        self.assertLessEqual(dec.alpha, 0.25)

    def test_mandatory_case_n_one(self):
        """Test N = 1.0 (Maximum noise)."""
        dec = self.router.route(noise_score=1.0, token_length=20)
        # z = 12.0 -> alpha_raw ~ 0.999994 -> alpha = 0.25
        self.assertEqual(dec.alpha, 0.25)
        self.assertGreater(dec.alpha_raw, 0.999)
        self.assertEqual(dec.routing_state, "clamped_max")

    def test_mandatory_case_length_one(self):
        """Test L = 1 (Very short text)."""
        # With moderate noise N = 0.21, L = 1:
        # z = 0.05*(20 - 1) + 12.0*0.21 = 0.05*19 + 2.52 = 0.95 + 2.52 = 3.47
        # alpha_raw = sigmoid(3.47) ~ 0.9698 -> alpha = 0.25
        dec = self.router.route(noise_score=0.21, token_length=1)
        self.assertEqual(dec.token_length, 1)
        self.assertEqual(dec.alpha, 0.25)

    def test_mandatory_case_length_twenty(self):
        """Test L = 20 (Reference length L0)."""
        # When L = 20, z = 12.0 * N
        dec_low = self.router.route(noise_score=0.15, token_length=20)
        self.assertEqual(dec_low.alpha, 0.02)

        dec_high = self.router.route(noise_score=0.30, token_length=20)
        self.assertEqual(dec_high.alpha, 0.25)

    def test_mandatory_case_length_one_hundred(self):
        """Test L = 100 (Long text: high length lowers z)."""
        # When N = 0.21, L = 100:
        # z = 0.05*(20 - 100) + 12.0*0.21 = 0.05*(-80) + 2.52 = -4.0 + 2.52 = -1.48
        # alpha_raw = sigmoid(-1.48) = 1 / (1 + exp(1.48)) ~ 0.18544
        # alpha = clamp(0.18544, 0.02, 0.25) = 0.18544 (adaptive active region!)
        dec = self.router.route(noise_score=0.21, token_length=100)
        self.assertEqual(dec.token_length, 100)
        self.assertAlmostEqual(dec.z_score, -1.48, places=4)
        expected_alpha = stable_sigmoid(-1.48)
        self.assertAlmostEqual(dec.alpha_raw, expected_alpha, places=4)
        self.assertAlmostEqual(dec.alpha, expected_alpha, places=4)
        self.assertEqual(dec.routing_state, "adaptive_active")
        self.assertGreater(dec.alpha, 0.02)
        self.assertLess(dec.alpha, 0.25)

    def test_lower_clamp_invariant_on_extreme_length(self):
        """Verify alpha is clamped to minimum 0.02 when z is deeply negative."""
        # N = 0.21, L = 200:
        # z = 0.05*(20 - 200) + 12.0*0.21 = 0.05*(-180) + 2.52 = -9.0 + 2.52 = -6.48
        # alpha_raw = sigmoid(-6.48) ~ 0.00153 < 0.02
        # alpha must clamp to 0.02
        dec = self.router.route(noise_score=0.21, token_length=200)
        self.assertLess(dec.alpha_raw, 0.02)
        self.assertEqual(dec.alpha, 0.02)
        self.assertEqual(dec.routing_state, "clamped_min")

    def test_alpha_global_invariants(self):
        """Exhaustive check: alpha can NEVER be below 0.02 or exceed 0.25 for any N, L."""
        test_noise_values = [0.0, 0.05, 0.199, 0.20, 0.200001, 0.25, 0.50, 0.75, 1.0]
        test_length_values = [0, 1, 5, 10, 20, 30, 50, 100, 128, 200, 500]

        for n in test_noise_values:
            for l in test_length_values:
                dec = self.router.route(noise_score=n, token_length=l)
                self.assertGreaterEqual(dec.alpha, 0.02, f"Failed min bound on N={n}, L={l}")
                self.assertLessEqual(dec.alpha, 0.25, f"Failed max bound on N={n}, L={l}")
                self.assertIsInstance(dec, RoutingDecision)
                d = dec.to_dict()
                self.assertIn("alpha", d)
                self.assertIn("alpha_raw", d)
                self.assertIn("z_score", d)
                self.assertIn("noise_score", d)
                self.assertIn("token_length", d)
                self.assertIn("routing_state", d)

    def test_determinism(self):
        """Multiple runs must produce identical results."""
        d1 = self.router.route(0.21, 100)
        d2 = self.router.route(0.21, 100)
        self.assertEqual(d1.alpha, d2.alpha)
        self.assertEqual(d1.alpha_raw, d2.alpha_raw)
        self.assertEqual(d1.z_score, d2.z_score)


if __name__ == "__main__":
    unittest.main()
