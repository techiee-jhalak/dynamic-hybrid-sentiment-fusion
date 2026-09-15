"""API route definitions for the Dynamic Hybrid Sentiment Fusion service.

Endpoints
---------
GET  /health      — Service liveness / model-readiness check
POST /predict     — Minimal sentiment classification response
POST /analyze     — Full explainability response with intermediate outputs
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.dependencies import ModelManager, get_pipeline
from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    NoiseFeaturesSchema,
    PredictRequest,
    PredictResponse,
    RouterExplanationSchema,
)
from src.evaluation.noise_sensitivity import assign_noise_group
from src.pipeline import PipelineResult, SentimentInferencePipeline

logger = logging.getLogger(__name__)

router = APIRouter()

_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_router_explanation(result: PipelineResult) -> RouterExplanationSchema:
    """Construct a deterministic router explanation from the pipeline result.

    Explanation text is derived exclusively from the ``routing_state`` field
    and the published formula constants. No free-form or fabricated text.
    """
    state = result.routing_state
    alpha = result.alpha
    N = result.noise_score

    if state == "low_noise_default":
        explanation = (
            f"N={N:.3f} ≤ 0.20 (noise threshold). "
            "Low-noise regime: α fixed at minimum value 0.02 — "
            "DistilBERT dominates (weight 0.98)."
        )
    elif state == "adaptive_active":
        explanation = (
            f"N={N:.3f} > 0.20. Adaptive routing active: "
            f"α=sigmoid(z)={alpha:.4f} within (0.02, 0.25). "
            "VADER weight increases with noise level."
        )
    elif state == "clamped_max":
        explanation = (
            f"N={N:.3f} > 0.20. sigmoid(z) exceeded 0.25 (upper clamp). "
            f"α clamped to maximum 0.25 — maximum VADER contribution."
        )
    elif state == "clamped_min":
        explanation = (
            f"N={N:.3f} > 0.20 but sigmoid(z) ≤ 0.02 (lower clamp). "
            f"α clamped to minimum 0.02."
        )
    else:
        explanation = f"routing_state='{state}' (see AdaptiveRouter documentation)."

    return RouterExplanationSchema(
        routing_state=state,
        alpha=alpha,
        alpha_raw=result.alpha_raw,
        z_score=0.0,   # z_score not stored in PipelineResult; compute if needed
        explanation=explanation,
    )


def _build_noise_features(result: PipelineResult) -> NoiseFeaturesSchema:
    """Map PipelineResult noise fields onto the NoiseFeaturesSchema."""
    return NoiseFeaturesSchema(
        emoji_density=result.emoji_density,
        repetition_ratio=result.repetition_score,
        codemix_intensity=result.code_mixing_ratio,
        symbol_density=result.symbol_density,
        composite_noise=result.noise_score,
        noise_band=assign_noise_group(result.noise_score),
    )


def _run_inference(
    text: str,
    pipeline: SentimentInferencePipeline,
    endpoint: str,
) -> PipelineResult:
    """Execute pipeline inference with safe error handling.

    Raises HTTPException with a generic 503 if inference fails so that
    internal details are not exposed to the client.
    """
    try:
        return pipeline.predict(text)
    except Exception as exc:
        logger.error(
            "Inference error on %s [%s]: %s",
            endpoint,
            type(exc).__name__,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference failed. Please try again later.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the liveness and readiness status of the API. "
        "Does not load or reload any model components."
    ),
    tags=["monitoring"],
)
async def health_check() -> HealthResponse:
    """Return service health and model-readiness without triggering model loads."""
    ready = ModelManager.is_ready()
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=_VERSION,
        model_loaded=ready,
        device=ModelManager.get_device(),
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Minimal sentiment prediction",
    description=(
        "Classify the sentiment of a single text using the Dynamic Hybrid "
        "Fusion framework. Returns the four core fields: sentiment label, "
        "fused score, dynamic routing weight α, and composite noise score N."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error (empty/blank/too-long text)"},
        503: {"model": ErrorResponse, "description": "Model inference error"},
    },
    tags=["inference"],
)
async def predict(
    request: PredictRequest,
    pipeline: SentimentInferencePipeline = Depends(get_pipeline),
) -> PredictResponse:
    """Run the full inference pipeline and return the minimal prediction response."""
    logger.debug("POST /predict — text length: %d chars", len(request.text))
    result = _run_inference(request.text, pipeline, "/predict")
    return PredictResponse(
        sentiment=result.sentiment_label,
        final_score=result.final_score,
        alpha=result.alpha,
        noise_score=result.noise_score,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Full sentiment analysis with explainability",
    description=(
        "Classify sentiment and return all intermediate outputs: "
        "VADER score, DistilBERT score, noise features (E, R, C, S, N), "
        "noise band classification, dynamic routing weight α, "
        "a deterministic router explanation, token length, and processed text. "
        "All fields are derived from real model computations — no fabricated values."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error (empty/blank/too-long text)"},
        503: {"model": ErrorResponse, "description": "Model inference error"},
    },
    tags=["inference"],
)
async def analyze(
    request: AnalyzeRequest,
    pipeline: SentimentInferencePipeline = Depends(get_pipeline),
) -> AnalyzeResponse:
    """Run the full inference pipeline and return the complete explainability response."""
    logger.debug("POST /analyze — text length: %d chars", len(request.text))
    result = _run_inference(request.text, pipeline, "/analyze")

    return AnalyzeResponse(
        # Core prediction
        sentiment=result.sentiment_label,
        final_score=result.final_score,
        prediction=result.prediction,
        confidence=result.confidence,
        # Component scores
        vader_score=result.vader_probability,
        distilbert_score=result.distilbert_probability,
        # Routing
        alpha=result.alpha,
        router=_build_router_explanation(result),
        # Noise
        noise_score=result.noise_score,
        noise_features=_build_noise_features(result),
        # Text metadata
        raw_text=result.raw_text,
        processed_text=result.processed_text,
        token_length=result.token_length,
    )
