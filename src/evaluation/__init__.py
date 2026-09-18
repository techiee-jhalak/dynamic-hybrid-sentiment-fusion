"""Evaluation package.

Lazy exports: EvaluationEngine, AblationRunner, ErrorAnalyzer and related heavy
imports (pandas, scipy, sklearn) are NOT imported at package level to keep
application startup fast. Import directly from submodules when needed.
"""

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

_NOISE_SENSITIVITY_NAMES = (
    "NoiseSensitivityAnalyzer", "NoiseGroupMetrics", "assign_noise_group",
    "ALL_NOISE_GROUPS", "NOISE_GROUP_LOW", "NOISE_GROUP_MODERATE",
    "NOISE_GROUP_HIGH", "NOISE_GROUP_EXTREME",
)

_ABLATION_NAMES = (
    "AblationConfig", "AblationResult", "AblationRunner",
    "ALL_VARIANTS", "VARIANT_A", "VARIANT_B", "VARIANT_C", "VARIANT_D", "build_router",
)

_STATS_NAMES = ("mcnemar_test", "StatisticalComparator", "McNemarResult")

_ERROR_NAMES = ("ErrorAnalyzer", "categorize_error", "ErrorRecord", "ALL_ERROR_CATEGORIES")


def __getattr__(name: str):
    if name == "ModelEvaluator":
        from src.evaluation.metrics import ModelEvaluator
        globals()[name] = ModelEvaluator
        return ModelEvaluator
    if name in ("EvaluationEngine", "ModelResult", "SamplePrediction"):
        from src.evaluation.engine import EvaluationEngine, ModelResult, SamplePrediction
        globals()["EvaluationEngine"] = EvaluationEngine
        globals()["ModelResult"] = ModelResult
        globals()["SamplePrediction"] = SamplePrediction
        return globals()[name]
    if name in _ABLATION_NAMES:
        from src.evaluation.ablation import (
            AblationConfig, AblationResult, AblationRunner,
            ALL_VARIANTS, VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D, build_router,
        )
        for n, v in [
            ("AblationConfig", AblationConfig), ("AblationResult", AblationResult),
            ("AblationRunner", AblationRunner), ("ALL_VARIANTS", ALL_VARIANTS),
            ("VARIANT_A", VARIANT_A), ("VARIANT_B", VARIANT_B),
            ("VARIANT_C", VARIANT_C), ("VARIANT_D", VARIANT_D),
            ("build_router", build_router),
        ]:
            globals()[n] = v
        return globals()[name]
    if name in _NOISE_SENSITIVITY_NAMES:
        from src.evaluation.noise_sensitivity import (
            NoiseSensitivityAnalyzer, NoiseGroupMetrics, assign_noise_group,
            ALL_NOISE_GROUPS, NOISE_GROUP_LOW, NOISE_GROUP_MODERATE,
            NOISE_GROUP_HIGH, NOISE_GROUP_EXTREME,
        )
        for n, v in [
            ("NoiseSensitivityAnalyzer", NoiseSensitivityAnalyzer),
            ("NoiseGroupMetrics", NoiseGroupMetrics),
            ("assign_noise_group", assign_noise_group),
            ("ALL_NOISE_GROUPS", ALL_NOISE_GROUPS),
            ("NOISE_GROUP_LOW", NOISE_GROUP_LOW),
            ("NOISE_GROUP_MODERATE", NOISE_GROUP_MODERATE),
            ("NOISE_GROUP_HIGH", NOISE_GROUP_HIGH),
            ("NOISE_GROUP_EXTREME", NOISE_GROUP_EXTREME),
        ]:
            globals()[n] = v
        return globals()[name]
    if name in _STATS_NAMES:
        from src.evaluation.statistical_tests import mcnemar_test, StatisticalComparator, McNemarResult
        globals()["mcnemar_test"] = mcnemar_test
        globals()["StatisticalComparator"] = StatisticalComparator
        globals()["McNemarResult"] = McNemarResult
        return globals()[name]
    if name in _ERROR_NAMES:
        from src.evaluation.error_analysis import (
            ErrorAnalyzer, categorize_error, ErrorRecord, ALL_ERROR_CATEGORIES,
        )
        globals()["ErrorAnalyzer"] = ErrorAnalyzer
        globals()["categorize_error"] = categorize_error
        globals()["ErrorRecord"] = ErrorRecord
        globals()["ALL_ERROR_CATEGORIES"] = ALL_ERROR_CATEGORIES
        return globals()[name]
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")
