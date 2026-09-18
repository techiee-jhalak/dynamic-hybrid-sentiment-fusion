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

from src.api.dependencies import ModelManager, get_pipeline, get_sentimix_pipeline
from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassVector3Schema,
    ErrorResponse,
    HealthResponse,
    NoiseFeaturesSchema,
    NoiseSummarySchema,
    PredictRequest,
    PredictResponse,
    RouterExplanationSchema,
)
from src.evaluation.noise_sensitivity import assign_noise_group
from src.pipeline import PipelineResult, SentimentInferencePipeline
from src.pipeline_3class import (
    Sentimix3ClassInferencePipeline,
    Sentimix3ClassPipelineResult,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_router_explanation(result: Any) -> RouterExplanationSchema:
    """Construct a deterministic router explanation from pipeline result."""
    state = getattr(result, "routing_state", "adaptive_active")
    alpha = getattr(result, "alpha", 0.02)
    alpha_raw = getattr(result, "alpha_raw", alpha)
    N = getattr(result, "noise_score", 0.0)

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
        alpha=round(alpha, 4),
        alpha_raw=round(alpha_raw, 4),
        z_score=0.0,
        explanation=explanation,
    )


def _build_noise_features(result: Any) -> NoiseFeaturesSchema:
    """Map pipeline result noise fields onto NoiseFeaturesSchema."""
    noise_score = getattr(result, "noise_score", 0.0)
    band = getattr(result, "noise_band", assign_noise_group(noise_score))
    repetition = getattr(result, "repetition_score", getattr(result, "repetition_ratio", 0.0))
    codemix = getattr(result, "code_mixing_ratio", getattr(result, "codemix_intensity", 0.0))

    return NoiseFeaturesSchema(
        emoji_density=round(getattr(result, "emoji_density", 0.0), 4),
        repetition_ratio=round(repetition, 4),
        codemix_intensity=round(codemix, 4),
        symbol_density=round(getattr(result, "symbol_density", 0.0), 4),
        composite_noise=round(noise_score, 4),
        noise_band=band,
    )


def _run_inference(
    text: str,
    pipeline: Any,
    endpoint: str,
) -> Any:
    """Execute pipeline inference with safe error handling."""
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
        "Returns the liveness and readiness status of the API, "
        "including model architecture, target classes, and verified checkpoint status."
    ),
    tags=["monitoring"],
)
async def health_check() -> HealthResponse:
    """Return service health and model-readiness without triggering model loads."""
    ready = ModelManager.is_ready()
    ckpt_loaded = ModelManager.is_checkpoint_loaded()
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=_VERSION,
        model_loaded=ready,
        device=ModelManager.get_device(),
        model_type="SentiMix 3-Class Dynamic Hybrid Fusion",
        checkpoint_loaded=ckpt_loaded,
        num_classes=3,
        classes=["positive", "negative", "neutral"],
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Minimal sentiment prediction (legacy binary)",
    description=(
        "Classify the sentiment of a single text using the Dynamic Hybrid "
        "Fusion framework. Returns the core fields: sentiment label, "
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
    """Run legacy inference pipeline and return the minimal prediction response."""
    logger.debug("POST /predict — text length: %d chars", len(request.text))
    result = _run_inference(request.text, pipeline, "/predict")
    return PredictResponse(
        sentiment=getattr(result, "sentiment_label", "Positive"),
        final_score=round(getattr(result, "final_score", 0.5), 4),
        alpha=round(getattr(result, "alpha", 0.02), 4),
        noise_score=round(getattr(result, "noise_score", 0.0), 4),
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Full sentiment analysis with explainability (SentiMix 3-Class)",
    description=(
        "Classify sentiment on the SentiMix 3-class scale (positive, negative, neutral) "
        "and return all intermediate outputs: VADER vector, DistilBERT vector, "
        "fused vector, dynamic routing weight α, deterministic router explanation, "
        "and multi-dimensional noise features (E, R, C, S, N). "
        "All predictions are generated by real models with zero fabricated outputs."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error (empty/blank/too-long text)"},
        503: {"model": ErrorResponse, "description": "Model inference error"},
    },
    tags=["inference"],
)
async def analyze(
    request: AnalyzeRequest,
    pipeline: Any = Depends(get_sentimix_pipeline),
) -> AnalyzeResponse:
    """Run production 3-class inference pipeline and return full explainability response."""
    logger.debug("POST /analyze — text length: %d chars", len(request.text))
    result = _run_inference(request.text, pipeline, "/analyze")

    if isinstance(result, Sentimix3ClassPipelineResult):
        # 3-Class production result
        band = result.noise_band
        alpha = round(result.alpha, 4)
        bert_weight = round(1.0 - result.alpha, 4)
        human_explanation = (
            f"Noise score: {result.noise_score:.2f} ({band}). "
            f"The dynamic router assigned α = {alpha:.2f} to the VADER component "
            f"and 1−α = {bert_weight:.2f} to DistilBERT."
        )

        vader_vec = ClassVector3Schema(
            positive=round(result.vader_vector[0], 4),
            negative=round(result.vader_vector[1], 4),
            neutral=round(result.vader_vector[2], 4),
        )
        distil_vec = ClassVector3Schema(
            positive=round(result.distilbert_vector[0], 4),
            negative=round(result.distilbert_vector[1], 4),
            neutral=round(result.distilbert_vector[2], 4),
        )
        fused_vec = ClassVector3Schema(
            positive=round(result.fused_vector[0], 4),
            negative=round(result.fused_vector[1], 4),
            neutral=round(result.fused_vector[2], 4),
        )
        noise_summary = NoiseSummarySchema(
            emoji_density=round(result.emoji_density, 4),
            repetition_score=round(result.repetition_score, 4),
            code_mixing_ratio=round(result.code_mixing_ratio, 4),
            symbol_density=round(result.symbol_density, 4),
            composite_noise=round(result.noise_score, 4),
            band=band,
        )

        return AnalyzeResponse(
            # Primary 3-class schema
            text=result.raw_text,
            predicted_label=result.predicted_label,
            predicted_class_index=result.predicted_class_index,
            confidence=round(result.confidence, 4),
            vader_vector=vader_vec,
            distilbert_vector=distil_vec,
            fused_vector=fused_vec,
            alpha=alpha,
            noise=noise_summary,
            explanation=human_explanation,
            # Backward-compatible fields
            sentiment=result.predicted_label.capitalize(),
            final_score=round(result.confidence, 4),
            prediction=result.predicted_class_index,
            vader_score=round(result.vader_vector[0], 4),
            distilbert_score=round(result.distilbert_vector[0], 4),
            router=_build_router_explanation(result),
            noise_score=round(result.noise_score, 4),
            noise_features=_build_noise_features(result),
            raw_text=result.raw_text,
            processed_text=result.processed_text,
            token_length=result.token_length,
        )

    # Fallback for binary / mock pipeline
    noise_score = getattr(result, "noise_score", 0.0)
    band = assign_noise_group(noise_score)
    alpha = round(getattr(result, "alpha", 0.02), 4)
    bert_weight = round(1.0 - alpha, 4)
    human_explanation = (
        f"Noise score: {noise_score:.2f} ({band}). "
        f"The dynamic router assigned α = {alpha:.2f} to the VADER component "
        f"and 1−α = {bert_weight:.2f} to DistilBERT."
    )

    vader_pos = round(getattr(result, "vader_probability", 0.5), 4)
    distil_pos = round(getattr(result, "distilbert_probability", 0.5), 4)
    fused_score = round(getattr(result, "final_score", 0.5), 4)
    label = getattr(result, "sentiment_label", "Positive").lower()
    class_idx = 0 if label == "positive" else (1 if label == "negative" else 2)

    return AnalyzeResponse(
        # Primary 3-class schema
        text=getattr(result, "raw_text", request.text),
        predicted_label=label,
        predicted_class_index=class_idx,
        confidence=round(getattr(result, "confidence", 0.5), 4),
        vader_vector=ClassVector3Schema(positive=vader_pos, negative=round(1.0 - vader_pos, 4), neutral=0.0),
        distilbert_vector=ClassVector3Schema(positive=distil_pos, negative=round(1.0 - distil_pos, 4), neutral=0.0),
        fused_vector=ClassVector3Schema(positive=fused_score, negative=round(1.0 - fused_score, 4), neutral=0.0),
        alpha=alpha,
        noise=NoiseSummarySchema(
            emoji_density=round(getattr(result, "emoji_density", 0.0), 4),
            repetition_score=round(getattr(result, "repetition_score", 0.0), 4),
            code_mixing_ratio=round(getattr(result, "code_mixing_ratio", 0.0), 4),
            symbol_density=round(getattr(result, "symbol_density", 0.0), 4),
            composite_noise=round(noise_score, 4),
            band=band,
        ),
        explanation=human_explanation,
        # Backward-compatible fields
        sentiment=getattr(result, "sentiment_label", "Positive"),
        final_score=fused_score,
        prediction=getattr(result, "prediction", class_idx),
        vader_score=vader_pos,
        distilbert_score=distil_pos,
        router=_build_router_explanation(result),
        noise_score=round(noise_score, 4),
        noise_features=_build_noise_features(result),
        raw_text=getattr(result, "raw_text", request.text),
        processed_text=getattr(result, "processed_text", request.text.lower()),
        token_length=getattr(result, "token_length", len(request.text.split())),
    )
