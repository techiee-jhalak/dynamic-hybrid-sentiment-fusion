"""Unit tests for DistilBERT training pipeline configuration and metric computation."""

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
from transformers import EvalPrediction

from src.models.train_distilbert import (
    compute_metrics,
    DistilBertTrainingPipeline,
)


class TestDistilBertTraining(unittest.TestCase):
    """Test suite verifying training pipeline parameters, metric calculations, and dataset formatting."""

    def test_compute_metrics(self):
        """Verify metric calculations for exact accuracy, macro F1, binary F1, precision, recall."""
        # Logits for 4 samples: [[neg, pos], ...]
        logits = np.array([
            [-2.0, 2.0],  # Pred: 1, True: 1 -> TP
            [3.0, -1.0],  # Pred: 0, True: 0 -> TN
            [-1.5, 1.5],  # Pred: 1, True: 0 -> FP
            [2.0, -2.0],  # Pred: 0, True: 1 -> FN
        ])
        labels = np.array([1, 0, 0, 1])
        eval_pred = EvalPrediction(predictions=logits, label_ids=labels)

        metrics = compute_metrics(eval_pred)
        # 2 correct out of 4 -> Accuracy = 0.50
        self.assertEqual(metrics["accuracy"], 0.50)
        self.assertIn("f1", metrics)
        self.assertIn("f1_binary", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertGreaterEqual(metrics["f1"], 0.0)
        self.assertLessEqual(metrics["f1"], 1.0)

    def test_build_training_args(self):
        """Verify exact training hyperparameter configuration."""
        pipeline = DistilBertTrainingPipeline()
        args = pipeline.build_training_args(
            epochs=3,
            batch_size=16,
            learning_rate=2e-5,
            seed=42,
        )

        self.assertEqual(args.num_train_epochs, 3)
        self.assertEqual(args.per_device_train_batch_size, 16)
        self.assertEqual(args.per_device_eval_batch_size, 16)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.metric_for_best_model, "f1")
        self.assertTrue(args.load_best_model_at_end)
        self.assertEqual(args.report_to, [])

    def test_prepare_hf_dataset(self):
        """Verify dataset preparation and tokenization mapping."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda texts, **kwargs: {
            "input_ids": [[101, 102]] * len(texts),
            "attention_mask": [[1, 1]] * len(texts),
        }

        pipeline = DistilBertTrainingPipeline()
        df = pd.DataFrame({
            "text": ["Awesome movie!", "Boring film."],
            "label": [1, 0],
        })

        hf_ds = pipeline.prepare_hf_dataset(
            df,
            tokenizer=mock_tokenizer,
            max_length=128,
        )

        self.assertEqual(len(hf_ds), 2)
        self.assertIn("input_ids", hf_ds.column_names)
        self.assertIn("label", hf_ds.column_names)


if __name__ == "__main__":
    unittest.main()
