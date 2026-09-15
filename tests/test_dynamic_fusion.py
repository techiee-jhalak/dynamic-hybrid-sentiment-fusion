"""Unit tests for the Dynamic Hybrid Fusion module."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from unittest.mock import MagicMock

from src.models.dynamic_fusion import (
    fuse_scores,
    FusionResult,
    DynamicFusionFramework,
)


class TestDynamicFusion(unittest.TestCase):
    """Test suite for dynamic score interpolation, classification thresholding, and input validation."""

    def test_case_1_both_models_agree_positive(self):
        """Test 1: Both VADER and DistilBERT predict positive sentiment."""
        s_vader = 0.90
        s_distil = 0.80
        alpha = 0.10

        res = fuse_scores(s_vader, s_distil, alpha)
        # S_final = 0.10*0.90 + 0.90*0.80 = 0.09 + 0.72 = 0.81
        self.assertAlmostEqual(res.final_score, 0.81, places=5)
        self.assertEqual(res.prediction, 1)
        self.assertEqual(res.sentiment_label, "Positive")
        self.assertAlmostEqual(res.confidence, 0.81, places=5)

    def test_case_2_both_models_agree_negative(self):
        """Test 2: Both VADER and DistilBERT predict negative sentiment."""
        s_vader = 0.10
        s_distil = 0.20
        alpha = 0.15

        res = fuse_scores(s_vader, s_distil, alpha)
        # S_final = 0.15*0.10 + 0.85*0.20 = 0.015 + 0.170 = 0.185
        self.assertAlmostEqual(res.final_score, 0.185, places=5)
        self.assertEqual(res.prediction, 0)
        self.assertEqual(res.sentiment_label, "Negative")
        self.assertAlmostEqual(res.confidence, 1.0 - 0.185, places=5)

    def test_case_3_models_disagree(self):
        """Test 3: VADER indicates positive (0.80) while DistilBERT indicates negative (0.40)."""
        s_vader = 0.80
        s_distil = 0.40
        alpha = 0.20

        res = fuse_scores(s_vader, s_distil, alpha)
        # S_final = 0.20*0.80 + 0.80*0.40 = 0.16 + 0.32 = 0.48
        self.assertAlmostEqual(res.final_score, 0.48, places=5)
        self.assertEqual(res.prediction, 0)
        self.assertEqual(res.sentiment_label, "Negative")

    def test_case_4_alpha_minimum_bound_point_zero_two(self):
        """Test 4: Alpha = 0.02 (low noise condition: 98% weight on DistilBERT)."""
        s_vader = 0.90
        s_distil = 0.60
        alpha = 0.02

        res = fuse_scores(s_vader, s_distil, alpha)
        # S_final = 0.02*0.90 + 0.98*0.60 = 0.018 + 0.588 = 0.606
        self.assertAlmostEqual(res.final_score, 0.606, places=5)
        self.assertEqual(res.alpha, 0.02)
        self.assertEqual(res.prediction, 1)

    def test_case_5_alpha_maximum_bound_point_twenty_five(self):
        """Test 5: Alpha = 0.25 (high noise condition: 25% weight on VADER)."""
        s_vader = 0.80
        s_distil = 0.40
        alpha = 0.25

        res = fuse_scores(s_vader, s_distil, alpha)
        # S_final = 0.25*0.80 + 0.75*0.40 = 0.20 + 0.30 = 0.50
        self.assertAlmostEqual(res.final_score, 0.50, places=5)
        self.assertEqual(res.alpha, 0.25)
        self.assertEqual(res.prediction, 1)
        self.assertEqual(res.sentiment_label, "Positive")

    def test_case_6_final_score_exactly_point_fifty(self):
        """Test 6: S_final = 0.50 -> must classify as Positive per decision rule (>= 0.50)."""
        res = fuse_scores(s_vader=0.50, s_distilbert=0.50, alpha=0.10)
        self.assertEqual(res.final_score, 0.50)
        self.assertEqual(res.prediction, 1)
        self.assertEqual(res.sentiment_label, "Positive")
        self.assertEqual(res.confidence, 0.50)

    def test_case_7_final_score_just_below_point_fifty(self):
        """Test 7: S_final = 0.4999 -> must classify as Negative."""
        # s_vader=0.4999, s_distilbert=0.4999
        res = fuse_scores(s_vader=0.4999, s_distilbert=0.4999, alpha=0.10)
        self.assertAlmostEqual(res.final_score, 0.4999, places=5)
        self.assertEqual(res.prediction, 0)
        self.assertEqual(res.sentiment_label, "Negative")

    def test_case_8_final_score_just_above_point_fifty(self):
        """Test 8: S_final = 0.5001 -> must classify as Positive."""
        res = fuse_scores(s_vader=0.5001, s_distilbert=0.5001, alpha=0.10)
        self.assertAlmostEqual(res.final_score, 0.5001, places=5)
        self.assertEqual(res.prediction, 1)
        self.assertEqual(res.sentiment_label, "Positive")

    def test_probability_validation_out_of_bounds(self):
        """Verify ValueError raised when input probabilities are outside [0, 1]."""
        with self.assertRaises(ValueError):
            fuse_scores(s_vader=-0.1, s_distilbert=0.5, alpha=0.1)

        with self.assertRaises(ValueError):
            fuse_scores(s_vader=1.2, s_distilbert=0.5, alpha=0.1)

        with self.assertRaises(ValueError):
            fuse_scores(s_vader=0.5, s_distilbert=-0.05, alpha=0.1)

        with self.assertRaises(ValueError):
            fuse_scores(s_vader=0.5, s_distilbert=1.05, alpha=0.1)

    def test_end_to_end_framework_analysis(self):
        """Verify DynamicFusionFramework orchestrates components end-to-end."""
        # Mock subcomponents to test pipeline integration
        mock_vader = MagicMock()
        mock_vader.predict_score.return_value = 0.85

        mock_distil = MagicMock()
        mock_distil.predict_score.return_value = 0.75

        framework = DynamicFusionFramework(
            vader_model=mock_vader,
            distilbert_model=mock_distil,
        )

        result = framework.analyze("Mast movie thi yaar! 🔥🔥")
        self.assertIn("raw_text", result)
        self.assertIn("noise_features", result)
        self.assertIn("vader_score", result)
        self.assertIn("distilbert_score", result)
        self.assertIn("alpha", result)
        self.assertIn("fused_score", result)
        self.assertIn("prediction", result)
        self.assertIn("sentiment_label", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["prediction"], 1)


if __name__ == "__main__":
    unittest.main()
