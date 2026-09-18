"""Tests for Sentimix3ClassInferencePipeline and response structure."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.pipeline_3class import (
    Sentimix3ClassInferencePipeline,
    Sentimix3ClassPipelineResult,
)
from src.models.distilbert_3class import DistilBert3ClassOutput


class TestSentimix3ClassInferencePipeline:
    @pytest.fixture
    def mock_pipeline(self):
        mock_bert = MagicMock()
        mock_bert.predict.return_value = DistilBert3ClassOutput(
            positive_prob=0.65,
            negative_prob=0.15,
            neutral_prob=0.20,
            predicted_label=0,
            logits=[1.2, -0.8, -0.5],
        )
        return Sentimix3ClassInferencePipeline(distilbert_model=mock_bert)

    def test_predict_single_text_structure(self, mock_pipeline):
        res = mock_pipeline.predict("bohot badhiya movie thi bhai! 😊")
        assert isinstance(res, Sentimix3ClassPipelineResult)
        assert res.predicted_label in ("positive", "negative", "neutral")
        assert res.predicted_class_index in (0, 1, 2)
        assert 0.0 <= res.confidence <= 1.0

        # Noise metrics present
        assert hasattr(res, "emoji_density")
        assert hasattr(res, "repetition_score")
        assert hasattr(res, "code_mixing_ratio")
        assert hasattr(res, "symbol_density")
        assert hasattr(res, "noise_score")
        assert res.noise_band in ("LOW", "MODERATE", "HIGH", "EXTREME")

        # Vectors present and 3-element
        assert len(res.vader_vector) == 3
        assert len(res.distilbert_vector) == 3
        assert len(res.fused_vector) == 3
        assert sum(res.fused_vector) == pytest.approx(1.0, rel=1e-5)

        # Alpha present
        assert 0.02 <= res.alpha <= 0.25

    def test_predict_batch_structure(self, mock_pipeline):
        texts = ["great movie", "bakwas acting", "kal aana"]
        results = mock_pipeline.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, Sentimix3ClassPipelineResult)
            assert len(r.fused_vector) == 3

    def test_pipeline_result_to_dict(self, mock_pipeline):
        res = mock_pipeline.predict("kya baat hai")
        d = res.to_dict()
        assert isinstance(d, dict)
        required_keys = [
            "raw_text", "processed_text", "token_length",
            "emoji_density", "repetition_score", "code_mixing_ratio",
            "symbol_density", "noise_score", "noise_band",
            "vader_vector", "distilbert_vector", "fused_vector",
            "alpha", "routing_state", "predicted_label",
            "predicted_class_index", "confidence",
        ]
        for k in required_keys:
            assert k in d, f"Missing key in to_dict(): {k}"
