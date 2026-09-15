"""Unit tests for the DistilBERT inference component using mocked PyTorch/Transformers."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn

from src.models.distilbert_model import (
    DistilBertSentimentModel,
    DistilBertSentimentOutput,
    ModelLoader,
)


class DummyModelOutput:
    """Mock container for transformer model forward pass output."""
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class TestDistilBertModel(unittest.TestCase):
    """Test suite verifying DistilBERT inference logic, device handling, and output contracts."""

    def setUp(self):
        self.device = torch.device("cpu")

        # Create mock tokenizer
        self.mock_tokenizer = MagicMock()
        def mock_tokenize(texts, **kwargs):
            if isinstance(texts, str):
                batch_size = 1
            else:
                batch_size = len(texts)
            return {
                "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
            }
        self.mock_tokenizer.side_effect = mock_tokenize

        # Create mock model with configurable logits
        self.mock_model = MagicMock()
        # Default: positive logit higher -> [neg_logit, pos_logit] = [-1.0, 2.5]
        self.mock_model.return_value = DummyModelOutput(logits=torch.tensor([[-1.0, 2.5]]))

        self.model = DistilBertSentimentModel(
            model_path_or_name="mock-distilbert",
            max_length=128,
            device="cpu",
            tokenizer=self.mock_tokenizer,
            model=self.mock_model,
            lazy_load=False,
        )

    def test_device_resolution(self):
        """Test automatic and manual device resolution."""
        cpu_dev = ModelLoader.resolve_device("cpu")
        self.assertEqual(cpu_dev.type, "cpu")

        auto_dev = ModelLoader.resolve_device(None)
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(auto_dev.type, expected_type)

    def test_single_load_and_reuse(self):
        """Verify model and tokenizer are retained and not reloaded across predictions."""
        self.assertTrue(self.model.is_loaded())
        tok_id = id(self.model.tokenizer)
        mod_id = id(self.model.model)

        _ = self.model.predict("Sentence one.")
        _ = self.model.predict("Sentence two.")

        self.assertEqual(id(self.model.tokenizer), tok_id)
        self.assertEqual(id(self.model.model), mod_id)

    def test_positive_prediction(self):
        """Test inference with positive logits."""
        # Logits [-1.0, 2.5] -> softmax pos_prob ~ 0.9707
        self.mock_model.return_value = DummyModelOutput(logits=torch.tensor([[-1.0, 2.5]]))

        out = self.model.predict("This is an extraordinary movie!")
        self.assertIsInstance(out, DistilBertSentimentOutput)
        self.assertGreater(out.positive_prob, 0.90)
        self.assertLess(out.negative_prob, 0.10)
        self.assertEqual(out.predicted_label, 1)
        self.assertAlmostEqual(out.positive_prob + out.negative_prob, 1.0, places=5)
        self.assertIsNotNone(out.logits)

    def test_negative_prediction(self):
        """Test inference with negative logits."""
        # Logits [3.0, -2.0] -> softmax pos_prob ~ 0.0067
        self.mock_model.return_value = DummyModelOutput(logits=torch.tensor([[3.0, -2.0]]))

        out = self.model.predict("Worst movie experience ever.")
        self.assertIsInstance(out, DistilBertSentimentOutput)
        self.assertLess(out.positive_prob, 0.05)
        self.assertGreater(out.negative_prob, 0.95)
        self.assertEqual(out.predicted_label, 0)
        self.assertAlmostEqual(out.positive_prob + out.negative_prob, 1.0, places=5)

    def test_empty_and_invalid_inputs(self):
        """Test that empty, whitespace, and None text return neutral defaults without calling model."""
        self.mock_model.reset_mock()

        for empty_text in ["", "   ", "\t\n", None]:
            score = self.model.predict_score(empty_text)
            out = self.model.predict(empty_text)
            self.assertEqual(score, 0.50)
            self.assertEqual(out.positive_prob, 0.50)
            self.assertEqual(out.negative_prob, 0.50)
            self.assertEqual(out.predicted_label, 1)

        # Ensure forward pass was not invoked for empty inputs
        self.mock_model.assert_not_called()

    def test_batch_inference(self):
        """Test batched prediction processing."""
        texts = ["Text one", "Text two", "Text three"]
        # Return 3 pairs of logits
        batch_logits = torch.tensor([
            [-1.0, 2.0],  # Positive
            [2.5, -1.5],  # Negative
            [0.5, 0.5],   # Neutral / Boundary
        ])
        self.mock_model.return_value = DummyModelOutput(logits=batch_logits)

        scores = self.model.predict_scores_batch(texts, batch_size=3)
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], 0.5)
        self.assertLess(scores[1], 0.5)
        self.assertAlmostEqual(scores[2], 0.5, places=3)

        batch_outs = self.model.predict_batch(texts, batch_size=3)
        self.assertEqual(len(batch_outs), 3)
        self.assertEqual(batch_outs[0].predicted_label, 1)
        self.assertEqual(batch_outs[1].predicted_label, 0)

    def test_output_dict_serialization(self):
        """Verify DistilBertSentimentOutput conversion to dictionary."""
        out = DistilBertSentimentOutput(
            positive_prob=0.88,
            negative_prob=0.12,
            predicted_label=1,
            logits=[-0.5, 1.5],
        )
        d = out.to_dict()
        self.assertIn("positive_prob", d)
        self.assertIn("negative_prob", d)
        self.assertIn("predicted_label", d)
        self.assertIn("logits", d)
        self.assertEqual(d["positive_prob"], 0.88)


if __name__ == "__main__":
    unittest.main()
