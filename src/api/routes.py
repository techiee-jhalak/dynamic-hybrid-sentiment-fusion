"""API route definitions for sentiment analysis and health checks."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import (
    SentimentRequest,
    BatchSentimentRequest,
    SentimentAnalysisResponse,
    HealthResponse,
)
from src.api.dependencies import get_fusion_model
from src.models.dynamic_fusion import DynamicFusionFramework

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health check endpoint."""
    raise NotImplementedError


@router.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    model: DynamicFusionFramework = Depends(get_fusion_model),
) -> SentimentAnalysisResponse:
    """Analyze sentiment for a single input text using dynamic hybrid fusion."""
    raise NotImplementedError


@router.post("/analyze/batch", response_model=List[SentimentAnalysisResponse])
async def analyze_sentiment_batch(
    request: BatchSentimentRequest,
    model: DynamicFusionFramework = Depends(get_fusion_model),
) -> List[SentimentAnalysisResponse]:
    """Batch sentiment analysis endpoint."""
    raise NotImplementedError
