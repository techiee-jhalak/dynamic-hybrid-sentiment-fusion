"""Pydantic request and response schemas for FastAPI endpoints."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze for sentiment")


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of input texts to analyze")


class NoiseFeaturesResponse(BaseModel):
    emoji_density: float = Field(..., ge=0.0, le=1.0)
    repetition_ratio: float = Field(..., ge=0.0, le=1.0)
    codemix_intensity: float = Field(..., ge=0.0, le=1.0)
    symbol_density: float = Field(..., ge=0.0, le=1.0)
    composite_noise: float = Field(..., ge=0.0, le=1.0)


class SentimentAnalysisResponse(BaseModel):
    text: str
    cleaned_text: str
    token_length: int
    noise_features: NoiseFeaturesResponse
    vader_score: float = Field(..., ge=0.0, le=1.0)
    distilbert_score: float = Field(..., ge=0.0, le=1.0)
    alpha: float = Field(..., ge=0.02, le=0.25)
    fused_score: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., description="0: Negative, 1: Positive")
    sentiment_label: str = Field(..., description="Negative or Positive")
    confidence: float = Field(..., ge=0.5, le=1.0)


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
