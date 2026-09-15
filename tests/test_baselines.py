"""Unit tests for the Evaluation Baseline Layer."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from src.models.baselines import (
    LogisticRegressionBaseline,
    StaticFusionBaseline,
    BERTweetBaseline,
)
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel
from src.models.dynamic_fusion import DynamicFusionFramework
from src.evaluation.metrics import ModelEvaluator


class DummyModelOutput:
    def __init__(self, logits):
        self.logits = logits


class TestBaselines(unittest.TestCase):
    """Test suite verifying all comparative baseline models, common interfaces, and evaluation metrics."""

    def test_logistic_regression_baseline(self):
        """Test Logistic Regression TF-IDF training, interface, and inference."""
        train_texts = [
            "This is a wonderful positive movie",
            "Fantastic storyline and great actors",
            "Terrible film, hated it completely",
            "Awful acting, waste of time and money",
        ]
        train_labels = [1, 1, 0, 0]

        lr = LogisticRegressionBaseline(ngram_range=(1, 2), max_features=100)
        # Verify not fitted error
        with self.assertRaises(RuntimeError):
            lr.predict_score("test")

        # Fit
        lr.fit(train_texts, train_labels)
        self.assertTrue(lr.is_fitted)

        # Predict score
        pos_score = lr.predict_score("wonderful movie")
        self.assertGreaterEqual(pos_score, 0.50)
        self.assertEqual(lr.predict_label("wonderful movie"), 1)

        neg_score = lr.predict_score("terrible waste")
        self.assertLessEqual(neg_score, 0.50)
        self.assertEqual(lr.predict_label("terrible waste"), 0)

        # Batch prediction
        batch_scores = lr.predict_scores_batch(["wonderful movie", "terrible waste"])
        self.assertEqual(len(batch_scores), 2)

        # Empty text
        self.assertEqual(lr.predict_score(""), 0.50)

    def test_static_fusion_baseline(self):
        """Test Static Fusion with fixed weight alpha = 0.15."""
        mock_vader = MagicMock(spec=VaderSentimentModel)
        mock_vader.predict_score.return_value = 0.80
        mock_vader.predict_scores_batch.return_value = [0.80]

        mock_distil = MagicMock(spec=DistilBertSentimentModel)
        mock_distil.predict_score.return_value = 0.60
        mock_distil.predict_scores_batch.return_value = [0.60]

        static_fusion = StaticFusionBaseline(
            vader_model=mock_vader,
            distilbert_model=mock_distil,
            fixed_alpha=0.15,
        )

        score = static_fusion.predict_score("Sample text")
        # S_static = 0.15 * 0.80 + 0.85 * 0.60 = 0.12 + 0.51 = 0.63
        self.assertAlmostEqual(score, 0.63, places=5)
        self.assertEqual(static_fusion.predict_label("Sample text"), 1)

        batch_scores = static_fusion.predict_scores_batch(["Sample text"])
        self.assertAlmostEqual(batch_scores[0], 0.63, places=5)

    def test_bertweet_baseline_disabled_by_default(self):
        """Verify BERTweet baseline is disabled by default and does not execute automatically."""
        bertweet = BERTweetBaseline()
        self.assertFalse(bertweet.enabled)

        # Should raise RuntimeError when attempting to predict or load while disabled
        with self.assertRaises(RuntimeError):
            bertweet.predict_score("Test sentence")

        with self.assertRaises(RuntimeError):
            bertweet.load_model()

        # Explicitly enabling
        bertweet.enable()
        self.assertTrue(bertweet.enabled)

    def test_model_evaluator_and_comparison(self):
        """Verify ModelEvaluator computes empirical accuracy, F1, precision, recall, and comparison table."""
        evaluator = ModelEvaluator()

        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 1, 0, 1, 1, 0]  # 5 correct out of 6 (1 FP)
        y_scores = [0.9, 0.8, 0.2, 0.6, 0.85, 0.1]

        metrics = evaluator.compute_metrics(y_true, y_pred, y_scores)
        self.assertAlmostEqual(metrics["accuracy"], 5 / 6, places=3)
        self.assertIn("macro_f1", metrics)
        self.assertIn("binary_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 3)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 0)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 2)

        # Test compare_baselines
        mock_model_1 = MagicMock(spec=VaderSentimentModel)
        mock_model_1.predict_scores_batch.return_value = y_scores

        df_comparison = evaluator.compare_baselines(
            models={"Model_A": mock_model_1},
            texts=["t1", "t2", "t3", "t4", "t5", "t6"],
            y_true=y_true,
        )
        self.assertIsInstance(df_comparison, pd.DataFrame)
        self.assertEqual(len(df_comparison), 1)
        self.assertIn("Accuracy", df_comparison.columns)
        self.assertIn("Macro F1", df_comparison.columns)


if __name__ == "__main__":
    unittest.main()
