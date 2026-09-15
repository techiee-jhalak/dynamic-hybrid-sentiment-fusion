"""Integration tests for the Unified Sentiment Inference Pipeline."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from unittest.mock import MagicMock
import torch

from src.pipeline import SentimentInferencePipeline, PipelineResult
from src.models.distilbert_model import DistilBertSentimentModel


class DummyModelOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class TestSentimentInferencePipeline(unittest.TestCase):
    """Integration test suite verifying end-to-end pipeline execution and output schemas."""

    def setUp(self):
        # Setup mock DistilBERT components to avoid external model downloads
        self.mock_tokenizer = MagicMock()
        def mock_tokenize(texts, **kwargs):
            batch_size = 1 if isinstance(texts, str) else len(texts)
            return {
                "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
            }
        self.mock_tokenizer.side_effect = mock_tokenize

        self.mock_model = MagicMock()
        self.mock_model.return_value = DummyModelOutput(logits=torch.tensor([[-1.0, 2.0]]))

        self.distilbert = DistilBertSentimentModel(
            device="cpu",
            tokenizer=self.mock_tokenizer,
            model=self.mock_model,
            lazy_load=False,
        )

        self.pipeline = SentimentInferencePipeline(distilbert_model=self.distilbert)

    def test_pipeline_positive_code_mixed_noisy_text(self):
        """Test full pipeline execution on noisy positive Hinglish text with emojis and repetitions."""
        text = "Kya mast movie thi yaar! 🔥🔥 Sooooo good! #SuperHit 10/10!!!"
        res = self.pipeline.predict(text)

        self.assertIsInstance(res, PipelineResult)
        self.assertEqual(res.raw_text, text)
        self.assertGreater(res.token_length, 5)

        # Noise features verification
        self.assertGreater(res.emoji_density, 0.0)
        self.assertGreater(res.repetition_score, 0.0)
        self.assertGreater(res.code_mixing_ratio, 0.0)
        self.assertGreater(res.symbol_density, 0.0)
        self.assertGreater(res.noise_score, 0.20)

        # Probabilities and routing
        self.assertGreaterEqual(res.vader_probability, 0.0)
        self.assertLessEqual(res.vader_probability, 1.0)
        self.assertGreaterEqual(res.distilbert_probability, 0.0)
        self.assertLessEqual(res.distilbert_probability, 1.0)
        self.assertGreaterEqual(res.alpha, 0.02)
        self.assertLessEqual(res.alpha, 0.25)

        # Fused decision
        self.assertGreaterEqual(res.final_score, 0.0)
        self.assertLessEqual(res.final_score, 1.0)
        self.assertEqual(res.prediction, 1)
        self.assertEqual(res.sentiment_label, "Positive")

    def test_pipeline_negative_text(self):
        """Test full pipeline on strongly negative review."""
        # Configure negative logits for DistilBERT mock
        self.mock_model.return_value = DummyModelOutput(logits=torch.tensor([[3.0, -2.0]]))

        text = "Bhai bilkul bakwaas experience tha... complete waste of money! 😡"
        res = self.pipeline.predict(text)

        self.assertEqual(res.prediction, 0)
        self.assertEqual(res.sentiment_label, "Negative")
        self.assertLess(res.final_score, 0.50)

    def test_pipeline_all_fields_returned(self):
        """Verify all 14 required fields in the prompt are present and valid."""
        text = "Awesome performance!"
        res = self.pipeline.predict(text)
        d = res.to_dict()

        required_fields = [
            "raw_text", "processed_text", "token_length",
            "emoji_density", "repetition_score", "code_mixing_ratio", "symbol_density", "noise_score",
            "vader_probability", "distilbert_probability",
            "alpha_raw", "alpha",
            "final_score", "prediction",
        ]
        for field in required_fields:
            self.assertIn(field, d)

    def test_pipeline_empty_and_null_input_handling(self):
        """Verify safe execution on empty/None inputs."""
        empty_cases = ["", "   ", "\t\n", None]
        for empty_text in empty_cases:
            res = self.pipeline.predict(empty_text)
            self.assertEqual(res.token_length, 0)
            self.assertEqual(res.noise_score, 0.0)
            self.assertEqual(res.alpha, 0.02)
            self.assertEqual(res.final_score, 0.50)

    def test_pipeline_batch_inference_consistency(self):
        """Verify batch prediction matches single predictions exactly."""
        texts = [
            "Amazing experience! ❤️",
            "Terrible and boring.",
            "Normal review statement.",
        ]
        # DistilBERT mock returning 3 logits
        batch_logits = torch.tensor([
            [-1.5, 2.0],
            [3.0, -2.0],
            [0.5, 0.5],
        ])
        self.mock_model.return_value = DummyModelOutput(logits=batch_logits)

        batch_results = self.pipeline.predict_batch(texts)
        self.assertEqual(len(batch_results), len(texts))

        for i, text in enumerate(texts):
            # Test individual call with corresponding logits
            self.mock_model.return_value = DummyModelOutput(logits=batch_logits[i : i + 1])
            single_res = self.pipeline.predict(text)
            self.assertAlmostEqual(batch_results[i].final_score, single_res.final_score, places=5)
            self.assertEqual(batch_results[i].prediction, single_res.prediction)

    def test_pipeline_determinism(self):
        """Verify multiple runs on identical text yield deterministic outputs."""
        text = "Super hit movie! 🔥 Full paisa vasool!"
        r1 = self.pipeline.predict(text)
        r2 = self.pipeline.predict(text)

        self.assertEqual(r1.final_score, r2.final_score)
        self.assertEqual(r1.alpha, r2.alpha)
        self.assertEqual(r1.noise_score, r2.noise_score)
        self.assertEqual(r1.prediction, r2.prediction)


if __name__ == "__main__":
    unittest.main()
