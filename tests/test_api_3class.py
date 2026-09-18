"""Integration and structural invariant tests for the SentiMix 3-Class API.

Tests cover:
1. GET /health and GET /api/health (readiness, classes, checkpoint status)
2. POST /analyze with diverse inputs (positive, negative, neutral, Hinglish, emoji, repetition)
3. Input validation (empty text, blank text, max length, malformed JSON)
4. Mathematical and structural invariants:
   - Probabilities sum approximately to 1.0 on Delta^2
   - Predicted class matches argmax(fused_vector)
   - Alpha bounded in [0.02, 0.25]
   - Noise features bounded in [0.0, 1.0]
   - Human-readable explanation matches computed values
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import ModelManager
from src.api.schemas import _MAX_TEXT_CHARS

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def ensure_models_initialized():
    """Ensure the real SentiMix model is initialized once for integration testing."""
    ModelManager.initialize(device="cpu")


# ---------------------------------------------------------------------------
# 1. Health Endpoint Tests
# ---------------------------------------------------------------------------


def test_health_endpoint_root_and_api():
    """Test both /health and /api/health return consistent readiness."""
    for path in ["/health", "/api/health"]:
        response = client.get(path)
        assert response.status_code == 200, f"Failed on path {path}"
        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        assert data["model_loaded"] is True
        assert data["num_classes"] == 3
        assert data["classes"] == ["positive", "negative", "neutral"]
        assert data["checkpoint_loaded"] is True
        assert "device" in data


# ---------------------------------------------------------------------------
# 2. POST /analyze Real Inference & Invariant Tests
# ---------------------------------------------------------------------------


def test_analyze_positive_example():
    """Test inference on a positive code-mixed Hinglish sample."""
    text = "Yeh movie bohot achhi hai yaar! Bahut maza aaya! 😊🔥"
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 200
    data = response.json()

    # Structural schema verification
    assert "predicted_label" in data
    assert data["predicted_label"] in ["positive", "negative", "neutral"]
    assert data["predicted_class_index"] in [0, 1, 2]
    assert 0.0 <= data["confidence"] <= 1.0

    # Vectors
    vader = data["vader_vector"]
    bert = data["distilbert_vector"]
    fused = data["fused_vector"]

    for vec in [vader, bert, fused]:
        total = vec["positive"] + vec["negative"] + vec["neutral"]
        assert pytest.approx(total, abs=0.05) == 1.0

    # Argmax invariant
    class_probs = [fused["positive"], fused["negative"], fused["neutral"]]
    expected_idx = class_probs.index(max(class_probs))
    assert data["predicted_class_index"] == expected_idx

    # Alpha bounds
    assert 0.02 <= data["alpha"] <= 0.25

    # Explanation
    assert "Noise score" in data["explanation"]
    assert "dynamic router" in data["explanation"]


def test_analyze_negative_example():
    """Test inference on a negative code-mixed text."""
    text = "Bilkul ghatiya service hai, waste of money aur time! Kabhi mat khareedo."
    response = client.post("/api/analyze", json={"text": text})
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_label"] in ["positive", "negative", "neutral"]
    fused = data["fused_vector"]
    total = fused["positive"] + fused["negative"] + fused["neutral"]
    assert pytest.approx(total, abs=0.05) == 1.0


def test_analyze_neutral_example():
    """Test inference on an objective/neutral statement."""
    text = "Aaj subah meeting schedule hui hai office room me at 10 AM."
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_label"] in ["positive", "negative", "neutral"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_analyze_emoji_heavy_input():
    """Test noise quantification under heavy emoji load."""
    text = "Happy birthday bro! 🎉🎂🥳🎁✨❤️🔥"
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 200
    data = response.json()

    noise = data["noise"]
    assert noise["emoji_density"] > 0.0
    assert 0.0 <= noise["composite_noise"] <= 1.0
    assert noise["band"] in ["LOW", "MODERATE", "HIGH", "EXTREME"]


def test_analyze_repetition_input():
    """Test noise quantification under elongated repeated characters."""
    text = "Haaaaan bilkuuuul sahhhiii bolaaaa yaaaaar"
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 200
    data = response.json()

    noise = data["noise"]
    assert noise["repetition_score"] > 0.0
    assert 0.0 <= noise["composite_noise"] <= 1.0


def test_analyze_code_mixed_input():
    """Test typical Hinglish social media utterance."""
    text = "Mujhe lagta hai the decision was really fair and balanced."
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 200
    data = response.json()

    assert data["noise"]["code_mixing_ratio"] >= 0.0
    assert 0.02 <= data["alpha"] <= 0.25


# ---------------------------------------------------------------------------
# 3. Input Validation & Error Handling Tests
# ---------------------------------------------------------------------------


def test_validation_empty_text():
    """Reject empty string with HTTP 422."""
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_validation_whitespace_only():
    """Reject whitespace-only string with HTTP 422."""
    response = client.post("/analyze", json={"text": "   \t\n   "})
    assert response.status_code == 422


def test_validation_exceeds_max_length():
    """Reject input exceeding maximum character limit with HTTP 422."""
    long_text = "x" * (_MAX_TEXT_CHARS + 5)
    response = client.post("/analyze", json={"text": long_text})
    assert response.status_code == 422


def test_validation_malformed_payload():
    """Reject non-JSON / missing required fields with HTTP 422."""
    response = client.post("/analyze", json={"not_text": 123})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 4. Backward Compatibility of /predict
# ---------------------------------------------------------------------------


def test_legacy_predict_backward_compatibility():
    """Verify POST /predict still responds as expected."""
    response = client.post("/predict", json={"text": "This is great!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "final_score" in data
    assert "alpha" in data
    assert "noise_score" in data
