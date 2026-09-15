"""Unit tests for the VADER sentiment component."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.models.vader_model import VaderSentimentModel, VaderSentimentOutput


class TestVaderSentimentModel(unittest.TestCase):
    """Test suite for VADER sentiment analysis, compound scaling, and interface compliance."""

    @classmethod
    def setUpClass(cls):
        cls.vader = VaderSentimentModel()

    def test_single_initialization(self):
        """Verify analyzer is created once and persists on the instance."""
        analyzer_ref1 = self.vader.analyzer
        self.assertIsNotNone(analyzer_ref1)
        _ = self.vader.predict_score("Test sentence.")
        analyzer_ref2 = self.vader.analyzer
        self.assertIs(analyzer_ref1, analyzer_ref2)

    def test_positive_sentiment(self):
        """Test strongly positive input text."""
        pos_text = "This movie is absolutely wonderful, fantastic, and glorious! 😍🔥"
        out = self.vader.predict(pos_text)

        self.assertIsInstance(out, VaderSentimentOutput)
        self.assertGreater(out.compound_score, 0.5)
        self.assertGreater(out.positive_prob, 0.75)
        self.assertLess(out.negative_prob, 0.25)
        self.assertEqual(out.predicted_label, 1)
        self.assertAlmostEqual(out.positive_prob + out.negative_prob, 1.0, places=6)

    def test_negative_sentiment(self):
        """Test strongly negative input text."""
        neg_text = "Worst movie ever! Completely awful, disgusting, and horrible. 😡"
        out = self.vader.predict(neg_text)

        self.assertIsInstance(out, VaderSentimentOutput)
        self.assertLess(out.compound_score, -0.5)
        self.assertLess(out.positive_prob, 0.25)
        self.assertGreater(out.negative_prob, 0.75)
        self.assertEqual(out.predicted_label, 0)
        self.assertAlmostEqual(out.positive_prob + out.negative_prob, 1.0, places=6)

    def test_compound_to_probability_mapping(self):
        """Verify linear scaling: S_vader = (compound + 1.0) / 2.0."""
        # Simulated compound values
        test_cases = [
            (-1.0, 0.0),
            (-0.5, 0.25),
            (0.0, 0.50),
            (0.5, 0.75),
            (1.0, 1.0),
        ]
        for compound, expected_prob in test_cases:
            calc_prob = (compound + 1.0) / 2.0
            self.assertAlmostEqual(calc_prob, expected_prob, places=6)
            self.assertGreaterEqual(calc_prob, 0.0)
            self.assertLessEqual(calc_prob, 1.0)

    def test_empty_and_invalid_inputs(self):
        """Verify graceful handling of empty, whitespace, and None text inputs."""
        empty_inputs = ["", "   ", "\t\n", None]
        for inp in empty_inputs:
            out = self.vader.predict(inp)
            self.assertEqual(out.compound_score, 0.0)
            self.assertEqual(out.positive_prob, 0.50)
            self.assertEqual(out.negative_prob, 0.50)
            self.assertEqual(out.predicted_label, 1)  # Threshold >= 0.50 -> Positive

    def test_batch_prediction(self):
        """Verify batch prediction returns matching results."""
        texts = [
            "Great acting! Loved it!",
            "Terrible plot and boring characters.",
            "Normal plain statement.",
        ]
        scores = self.vader.predict_scores_batch(texts)
        batch_outs = self.vader.predict_batch(texts)

        self.assertEqual(len(scores), len(texts))
        self.assertEqual(len(batch_outs), len(texts))

        for i, text in enumerate(texts):
            single_score = self.vader.predict_score(text)
            self.assertAlmostEqual(scores[i], single_score, places=6)
            self.assertAlmostEqual(batch_outs[i].positive_prob, single_score, places=6)

    def test_output_dict_serialization(self):
        """Verify VaderSentimentOutput dictionary serialization."""
        out = self.vader.predict("Excellent work!")
        d = out.to_dict()
        self.assertIn("positive_prob", d)
        self.assertIn("negative_prob", d)
        self.assertIn("predicted_label", d)
        self.assertIn("compound_score", d)
        self.assertEqual(d["positive_prob"], out.positive_prob)


if __name__ == "__main__":
    unittest.main()
