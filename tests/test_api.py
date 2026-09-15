"""Unit tests for FastAPI routes and schemas.

Tests cover:
- /health endpoint (status logic, model_loaded flag)
- /predict endpoint (valid inputs, expected fields, mocking inference)
- /analyze endpoint (explainability fields, routing deterministic text)
- validation (empty text, blank text, exceeding max length)
- error handling (503 on inference failure, JSON structure)
- Model lifecycle check (dependencies behavior)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import ModelManager, get_pipeline
from src.api.schemas import _MAX_TEXT_CHARS
from src.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Test Fixtures & Mocks
# ---------------------------------------------------------------------------

client = TestClient(app)


def mock_pipeline_result(text: str) -> PipelineResult:
    """Create a dummy but perfectly structured PipelineResult for testing."""
    return PipelineResult(
        raw_text=text,
        processed_text=text.lower(),
        token_length=len(text.split()),
        emoji_density=0.1,
        repetition_score=0.2,
        code_mixing_ratio=0.3,
        symbol_density=0.4,
        noise_score=0.25,
        vader_probability=0.6,
        distilbert_probability=0.8,
        alpha_raw=0.15,
        alpha=0.15,
        final_score=0.77,
        prediction=1,
        sentiment_label="Positive",
        confidence=0.77,
        routing_state="adaptive_active",
    )


@pytest.fixture(autouse=True)
def _mock_model_manager_is_ready():
    """By default, pretend the ModelManager is ready for all tests."""
    with patch.object(ModelManager, "is_ready", return_value=True):
        yield


@pytest.fixture
def mock_pipeline():
    """Provide a mock SentimentInferencePipeline dependency."""
    pipeline_mock = MagicMock()
    # By default, make it succeed
    pipeline_mock.predict.side_effect = lambda text: mock_pipeline_result(text)

    # Override the FastAPI dependency
    app.dependency_overrides[get_pipeline] = lambda: pipeline_mock
    yield pipeline_mock
    # Cleanup after test
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /health Tests
# ---------------------------------------------------------------------------

def test_health_check_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "version" in data
    assert "device" in data


def test_health_check_degraded_when_not_ready():
    with patch.object(ModelManager, "is_ready", return_value=False):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# POST /predict Tests
# ---------------------------------------------------------------------------

def test_predict_success(mock_pipeline):
    response = client.post("/api/predict", json={"text": "Hello world"})
    assert response.status_code == 200
    data = response.json()

    assert data["sentiment"] == "Positive"
    assert data["final_score"] == 0.77
    assert data["alpha"] == 0.15
    assert data["noise_score"] == 0.25

    mock_pipeline.predict.assert_called_once_with("Hello world")


# ---------------------------------------------------------------------------
# POST /analyze Tests
# ---------------------------------------------------------------------------

def test_analyze_success(mock_pipeline):
    response = client.post("/api/analyze", json={"text": "Hello world"})
    assert response.status_code == 200
    data = response.json()

    # Core prediction
    assert data["sentiment"] == "Positive"
    assert data["final_score"] == 0.77
    assert data["prediction"] == 1
    assert data["confidence"] == 0.77

    # Component scores
    assert data["vader_score"] == 0.6
    assert data["distilbert_score"] == 0.8

    # Routing explanation
    router = data["router"]
    assert router["alpha"] == 0.15
    assert router["routing_state"] == "adaptive_active"
    assert "explanation" in router
    assert "0.20" in router["explanation"]  # Threshold should be mentioned

    # Noise features
    assert data["noise_score"] == 0.25
    features = data["noise_features"]
    assert features["emoji_density"] == 0.1
    assert features["repetition_ratio"] == 0.2
    assert features["codemix_intensity"] == 0.3
    assert features["symbol_density"] == 0.4
    assert features["composite_noise"] == 0.25
    assert features["noise_band"] == "MODERATE"

    mock_pipeline.predict.assert_called_once_with("Hello world")


def test_analyze_low_noise_explanation(mock_pipeline):
    """Test the deterministic explanation for low noise regime."""
    def _low_noise_result(text):
        res = mock_pipeline_result(text)
        # Override fields to simulate low noise
        return PipelineResult(
            raw_text=res.raw_text, processed_text=res.processed_text,
            token_length=res.token_length, emoji_density=res.emoji_density,
            repetition_score=res.repetition_score, code_mixing_ratio=res.code_mixing_ratio,
            symbol_density=res.symbol_density, noise_score=0.10,
            vader_probability=res.vader_probability, distilbert_probability=res.distilbert_probability,
            alpha_raw=0.01, alpha=0.02, final_score=0.8, prediction=1,
            sentiment_label="Positive", confidence=0.8,
            routing_state="low_noise_default"
        )
    mock_pipeline.predict.side_effect = _low_noise_result

    response = client.post("/api/analyze", json={"text": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["noise_score"] == 0.10
    assert data["noise_features"]["noise_band"] == "LOW"
    assert "Low-noise regime: α fixed at minimum value 0.02" in data["router"]["explanation"]


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

def test_validation_empty_text(mock_pipeline):
    response = client.post("/api/predict", json={"text": ""})
    assert response.status_code == 422
    assert "String should have at least 1 character" in response.text


def test_validation_blank_text(mock_pipeline):
    response = client.post("/api/predict", json={"text": "   \n\t  "})
    assert response.status_code == 422
    assert "text must not be blank or whitespace-only" in response.text


def test_validation_exceeds_max_length(mock_pipeline):
    text = "A" * (_MAX_TEXT_CHARS + 1)
    response = client.post("/api/predict", json={"text": text})
    assert response.status_code == 422
    assert "String should have at most" in response.text


def test_validation_missing_text_field(mock_pipeline):
    response = client.post("/api/predict", json={"something_else": "test"})
    assert response.status_code == 422
    assert "Field required" in response.text


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------

def test_inference_failure_returns_503(mock_pipeline):
    mock_pipeline.predict.side_effect = RuntimeError("GPU Out of Memory")
    response = client.post("/api/predict", json={"text": "test"})

    assert response.status_code == 503
    data = response.json()
    assert data["detail"] == "Inference failed. Please try again later."
    assert "GPU" not in data["detail"]


def test_global_exception_handler_returns_500():
    """Test that unexpected non-HTTP exceptions are caught globally."""
    def failing_dependency():
        raise ValueError("Critical internal logic failure")

    app.dependency_overrides[get_pipeline] = failing_dependency

    # Disable TestClient raising exceptions to test the 500 handler
    error_client = TestClient(app, raise_server_exceptions=False)
    response = error_client.post("/api/predict", json={"text": "test"})
    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "An internal server error occurred."
    assert data["code"] == "INTERNAL_ERROR"
    assert "logic failure" not in data["detail"]

    app.dependency_overrides.clear()
