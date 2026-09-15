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
from src.evaluation.noise_sensitivity import (
    NoiseSensitivityAnalyzer,
    NoiseGroupMetrics,
    assign_noise_group,
    ALL_NOISE_GROUPS,
    NOISE_GROUP_LOW,
    NOISE_GROUP_MODERATE,
    NOISE_GROUP_HIGH,
    NOISE_GROUP_EXTREME,
)
from src.evaluation.statistical_tests import (
    mcnemar_test,
    StatisticalComparator,
    McNemarResult,
)
from src.evaluation.error_analysis import (
    ErrorAnalyzer,
    categorize_error,
    ErrorRecord,
    ALL_ERROR_CATEGORIES,
)

__all__ = [
    "ModelEvaluator",
    "EvaluationEngine", "ModelResult", "SamplePrediction",
    "AblationConfig", "AblationResult", "AblationRunner",
    "ALL_VARIANTS", "VARIANT_A", "VARIANT_B", "VARIANT_C", "VARIANT_D",
    "build_router",
    "NoiseSensitivityAnalyzer", "NoiseGroupMetrics", "assign_noise_group", "ALL_NOISE_GROUPS",
    "NOISE_GROUP_LOW", "NOISE_GROUP_MODERATE", "NOISE_GROUP_HIGH", "NOISE_GROUP_EXTREME",
    "mcnemar_test", "StatisticalComparator", "McNemarResult",
    "ErrorAnalyzer", "categorize_error", "ErrorRecord", "ALL_ERROR_CATEGORIES",
]
