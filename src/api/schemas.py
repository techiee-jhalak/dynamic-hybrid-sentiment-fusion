"""Pydantic request and response schemas for FastAPI endpoints.

All field descriptions are sourced directly from PROJECT_SPEC.md and the
research paper formulas. No fabricated default values are introduced here.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Configuration-derived constants (keep in sync with configs/config.py)
# ---------------------------------------------------------------------------

# DistilBERT max_length = 128 tokens → approximate character budget
# We allow a generous character limit; empty text is rejected separately.
_MAX_TEXT_CHARS: int = int(os.environ.get("API_MAX_TEXT_CHARS", "2048"))


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_TEXT_CHARS,
        description=(
            "Input text to classify. Must be non-empty and at most "
            f"{_MAX_TEXT_CHARS} characters. Emojis, Hinglish tokens, and "
            "repeated characters are preserved for noise quantification."
        ),
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace-only")
        return v


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_TEXT_CHARS,
        description=(
            "Input text to analyze. Must be non-empty and at most "
            f"{_MAX_TEXT_CHARS} characters."
        ),
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace-only")
        return v


# ---------------------------------------------------------------------------
# Sub-schemas used in responses
# ---------------------------------------------------------------------------


class NoiseFeaturesSchema(BaseModel):
    """Four-dimensional noise feature vector (E, R, C, S) and composite N.

    All values are in [0, 1] as defined in PROJECT_SPEC §3.1-3.2.
    """

    emoji_density: float = Field(
        ..., ge=0.0, le=1.0,
        description="E — emoji count / max(token_count, 1)"
    )
    repetition_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="R — elongated-character token ratio"
    )
    codemix_intensity: float = Field(
        ..., ge=0.0, le=1.0,
        description="C — Hinglish/code-mixing token ratio"
    )
    symbol_density: float = Field(
        ..., ge=0.0, le=1.0,
        description="S — special-symbol token ratio"
    )
    composite_noise: float = Field(
        ..., ge=0.0, le=1.0,
        description="N = 0.25E + 0.25R + 0.30C + 0.20S"
    )
    noise_band: str = Field(
        ...,
        description="Qualitative noise band: LOW | MODERATE | HIGH | EXTREME"
    )


class ClassVector3Schema(BaseModel):
    """Three-class probability vector on the simplex Delta^2."""

    positive: float = Field(..., ge=0.0, le=1.0, description="Positive class probability")
    negative: float = Field(..., ge=0.0, le=1.0, description="Negative class probability")
    neutral: float = Field(..., ge=0.0, le=1.0, description="Neutral class probability")


class NoiseSummarySchema(BaseModel):
    """Summary of noise features and qualitative noise band."""

    emoji_density: float = Field(..., ge=0.0, le=1.0, description="Emoji density (E)")
    repetition_score: float = Field(..., ge=0.0, le=1.0, description="Repetition score (R)")
    code_mixing_ratio: float = Field(..., ge=0.0, le=1.0, description="Code-mixing ratio (C)")
    symbol_density: float = Field(..., ge=0.0, le=1.0, description="Symbol density (S)")
    composite_noise: float = Field(..., ge=0.0, le=1.0, description="Composite noise score (N)")
    band: str = Field(..., description="Noise band: LOW | MODERATE | HIGH | EXTREME")


class RouterExplanationSchema(BaseModel):
    """Concise deterministic explanation of the adaptive router decision.

    Only facts derivable from the routing formula and routing_state are
    included; no free-form or fabricated text.
    """

    routing_state: str = Field(
        ...,
        description=(
            "One of: low_noise_default | adaptive_active | "
            "clamped_min | clamped_max"
        ),
    )
    alpha: float = Field(
        ..., ge=0.02, le=0.25,
        description="Final clamped α in [0.02, 0.25]"
    )
    alpha_raw: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw sigmoid(z) value before clamping"
    )
    z_score: float = Field(
        ...,
        description="z = w1*(L0 - L) + w2*N intermediate value"
    )
    explanation: str = Field(
        ...,
        description=(
            "Deterministic human-readable summary of why α took this value, "
            "derived purely from routing_state and formula constants."
        ),
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PredictResponse(BaseModel):
    """Minimal typed response for POST /predict.

    Contains the four fields mandated by the task specification.
    """

    sentiment: str = Field(
        ..., description="'Positive' or 'Negative'"
    )
    final_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="S_final = α·S_vader + (1−α)·S_distilbert"
    )
    alpha: float = Field(
        ..., ge=0.02, le=0.25,
        description="Dynamic fusion weight α ∈ [0.02, 0.25]"
    )
    noise_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Composite noise score N ∈ [0, 1]"
    )


class AnalyzeResponse(BaseModel):
    """Full explainability response for POST /analyze.

    Returns the complete 3-class production schema and retains backward-compatible
    aliases for binary clients and tests.
    """

    # SentiMix 3-class primary schema
    text: str = Field(..., description="Input text analyzed")
    predicted_label: str = Field(..., description="'positive', 'negative', or 'neutral'")
    predicted_class_index: int = Field(..., description="0 = positive, 1 = negative, 2 = neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence in [0, 1]")

    vader_vector: ClassVector3Schema = Field(..., description="VADER 3-class probability distribution")
    distilbert_vector: ClassVector3Schema = Field(..., description="DistilBERT 3-class probability distribution")
    fused_vector: ClassVector3Schema = Field(..., description="Dynamic fused 3-class probability distribution")

    alpha: float = Field(..., ge=0.02, le=0.25, description="Dynamic routing weight α ∈ [0.02, 0.25]")
    noise: NoiseSummarySchema = Field(..., description="Noise feature breakdown and composite score")
    explanation: str = Field(..., description="Human-readable explanation of router and noise decision")

    # Backward-compatible fields
    sentiment: str = Field(..., description="'Positive', 'Negative', or 'Neutral'")
    final_score: float = Field(..., ge=0.0, le=1.0, description="S_final or fused confidence in [0, 1]")
    prediction: int = Field(..., description="Predicted class index (0, 1, or 2)")
    vader_score: float = Field(..., ge=0.0, le=1.0, description="S_vader positive score ∈ [0, 1]")
    distilbert_score: float = Field(..., ge=0.0, le=1.0, description="S_distilbert positive score ∈ [0, 1]")
    router: RouterExplanationSchema = Field(..., description="Detailed router decision and explanation")
    noise_score: float = Field(..., ge=0.0, le=1.0, description="Composite noise score N ∈ [0, 1]")
    noise_features: NoiseFeaturesSchema = Field(..., description="Noise features object")
    raw_text: str = Field(..., description="Original input text")
    processed_text: str = Field(..., description="Preprocessed text")
    token_length: int = Field(..., ge=0, description="Token count after preprocessing")


class HealthResponse(BaseModel):
    """GET /health response schema."""

    status: Literal["ok", "degraded"] = Field(
        ..., description="'ok' when all models are loaded"
    )
    version: str = Field(..., description="API version string")
    model_loaded: bool = Field(..., description="True when pipeline is ready")
    device: str = Field(..., description="Compute device (cpu | cuda)")
    model_type: Optional[str] = Field(
        default="SentiMix 3-Class Dynamic Hybrid Fusion",
        description="Active model architecture"
    )
    checkpoint_loaded: Optional[bool] = Field(
        default=None,
        description="Whether the real trained checkpoint is loaded"
    )
    num_classes: Optional[int] = Field(
        default=3,
        description="Number of sentiment target classes (3: positive, negative, neutral)"
    )
    classes: Optional[List[str]] = Field(
        default_factory=lambda: ["positive", "negative", "neutral"],
        description="Ordered class names: 0=positive, 1=negative, 2=neutral"
    )


class ErrorResponse(BaseModel):
    """Structured JSON error body returned on 4xx/5xx responses."""

    detail: str = Field(..., description="Human-readable error description")
    code: str = Field(..., description="Machine-readable error code")
