"""Evaluation package."""
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.engine import EvaluationEngine, ModelResult, SamplePrediction
from src.evaluation.ablation import (
    AblationConfig,
    AblationResult,
    AblationRunner,
    ALL_VARIANTS,
    VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D,
    build_router,
)

__all__ = [
    "ModelEvaluator",
    "EvaluationEngine", "ModelResult", "SamplePrediction",
    "AblationConfig", "AblationResult", "AblationRunner",
    "ALL_VARIANTS", "VARIANT_A", "VARIANT_B", "VARIANT_C", "VARIANT_D",
    "build_router",
]
