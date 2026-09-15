"""Unit tests for sentiment model interfaces and BaseSentimentModel contracts."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from unittest.mock import MagicMock
import torch

from src.models.base import BaseSentimentModel
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel
from src.models.baselines import LogisticRegressionBaseline, StaticFusionBaseline
from src.models.dynamic_fusion import DynamicFusionFramework


class DummyTransformerOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class TestSentimentModels(unittest.TestCase):
    """Test suite verifying VADER, DistilBERT, baselines, and fusion adhere to BaseSentimentModel."""

    def setUp(self):
        # Setup VADER
        self.vader = VaderSentimentModel()

        # Setup Mock DistilBERT
        self.mock_tokenizer = MagicMock()
        def mock_tokenize(texts, **kwargs):
            b_size = 1 if isinstance(texts, str) else len(texts)
            return {
                "input_ids": torch.ones((b_size, 8), dtype=torch.long),
                "attention_mask": torch.ones((b_size, 8), dtype=torch.long),
            }
        self.mock_tokenizer.side_effect = mock_tokenize
        self.mock_model = MagicMock()
        def mock_forward(**kwargs):
            b_size = kwargs["input_ids"].shape[0]
            # [neg_logit, pos_logit]
            return DummyTransformerOutput(logits=torch.tensor([[-0.5, 1.5]] * b_size))
        self.mock_model.side_effect = mock_forward

        self.distilbert = DistilBertSentimentModel(
            model_path_or_name="mock-distilbert",
            max_length=128,
            device="cpu",
            tokenizer=self.mock_tokenizer,
            model=self.mock_model,
            lazy_load=False,
        )

        # Setup Logistic Regression
        self.lr = LogisticRegressionBaseline(ngram_range=(1, 1), max_features=50)
        self.lr.fit(
            texts=["good positive movie", "bad negative movie"],
            labels=[1, 0],
        )

        # Setup Static Fusion
        self.static_fusion = StaticFusionBaseline(
            vader_model=self.vader,
            distilbert_model=self.distilbert,
            fixed_alpha=0.15,
        )

        # Setup Dynamic Fusion
        self.dynamic_fusion = DynamicFusionFramework(
            vader_model=self.vader,
            distilbert_model=self.distilbert,
        )

    def test_all_models_inherit_base_interface(self):
        """Verify all sentiment models subclass BaseSentimentModel."""
        models = [
            self.vader,
            self.distilbert,
            self.lr,
            self.static_fusion,
            self.dynamic_fusion,
        ]
        for model in models:
            self.assertIsInstance(model, BaseSentimentModel)

    def test_predict_score_contract_and_bounds(self):
        """Verify predict_score returns float strictly bounded in [0.0, 1.0]."""
        test_sentences = [
            "This movie is wonderfully directed and beautifully acted! 😍",
            "Worst film ever made, complete garbage.",
            "Plain neutral text statement.",
            "",
        ]
        models = [
            self.vader,
            self.distilbert,
            self.lr,
            self.static_fusion,
            self.dynamic_fusion,
        ]

        for model in models:
            for text in test_sentences:
                score = model.predict_score(text)
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_predict_scores_batch_contract(self):
        """Verify predict_scores_batch returns list of floats bounded in [0.0, 1.0]."""
        texts = ["Positive hit", "Negative flop"]
        models = [
            self.vader,
            self.distilbert,
            self.lr,
            self.static_fusion,
            self.dynamic_fusion,
        ]

        for model in models:
            batch_scores = model.predict_scores_batch(texts)
            self.assertIsInstance(batch_scores, list)
            self.assertEqual(len(batch_scores), len(texts))
            for s in batch_scores:
                self.assertIsInstance(s, float)
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)

    def test_predict_label_thresholding(self):
        """Verify predict_label returns 1 when score >= threshold else 0."""
        class MockScoreModel(BaseSentimentModel):
            def __init__(self, score):
                self.score = score
            def predict_score(self, text):
                return self.score
            def predict_scores_batch(self, texts):
                return [self.score] * len(texts)

        pos_model = MockScoreModel(0.50)
        self.assertEqual(pos_model.predict_label("sample", threshold=0.50), 1)

        pos_high = MockScoreModel(0.92)
        self.assertEqual(pos_high.predict_label("sample", threshold=0.50), 1)

        neg_model = MockScoreModel(0.4999)
        self.assertEqual(neg_model.predict_label("sample", threshold=0.50), 0)

        neg_low = MockScoreModel(0.12)
        self.assertEqual(neg_low.predict_label("sample", threshold=0.50), 0)


if __name__ == "__main__":
    unittest.main()
