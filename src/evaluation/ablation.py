"""Ablation Study Framework for Dynamic Hybrid Sentiment Fusion.

Compares four architectural variants without duplicating any fusion or routing
implementation. Each variant is constructed by injecting a differently-configured
router (or a fixed-alpha shim) into the existing DynamicFusionFramework.

Variants
--------
A. Full Dynamic Fusion (proposed system — baseline reference for delta)
B. Static Fusion          — dynamic routing disabled; constant alpha = alpha_min = 0.02
C. No Length Contribution — routing z-score uses only noise term (w1 forced to 0)
D. No Gated Threshold     — N <= 0.20 guard removed; sigmoid always active

All metrics are computed from actual predictions via EvaluationEngine.
No results are hardcoded.

Output
------
results/ablation_results.csv
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from configs.config import RoutingConfig, config as _default_config
from src.models.adaptive_router import AdaptiveRouter, RoutingDecision, stable_sigmoid
from src.models.dynamic_fusion import DynamicFusionFramework
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel
from src.features.noise_quantifier import NoiseQuantifier
from src.data.preprocessor import TextPreprocessor
from src.evaluation.engine import EvaluationEngine, ModelResult


# ---------------------------------------------------------------------------
# Ablation configuration flags
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationConfig:
    """Configuration flags for a single ablation variant.

    Attributes
    ----------
    name:
        Human-readable variant label recorded in the CSV.
    use_static_alpha:
        If True, alpha is always ``static_alpha`` (routing disabled entirely).
    static_alpha:
        Fixed alpha used when ``use_static_alpha=True``.
    disable_length_contribution:
        If True, w1 is set to 0 so token length has no effect on z.
    disable_noise_gate:
        If True, the N <= 0.20 early-return is removed; sigmoid always runs.
    routing_config:
        Base ``RoutingConfig`` used by the router for all other constants.
    """
    name: str
    use_static_alpha: bool = False
    static_alpha: float = 0.02
    disable_length_contribution: bool = False
    disable_noise_gate: bool = False
    routing_config: RoutingConfig = _default_config.routing


# Pre-defined ablation variant definitions
VARIANT_A = AblationConfig(
    name="A_Full_Dynamic_Fusion",
    use_static_alpha=False,
    disable_length_contribution=False,
    disable_noise_gate=False,
)

VARIANT_B = AblationConfig(
    name="B_Static_Fusion_No_Routing",
    use_static_alpha=True,
    static_alpha=_default_config.routing.alpha_min,   # alpha = 0.02 (minimal VADER weight)
)

VARIANT_C = AblationConfig(
    name="C_No_Length_Contribution",
    use_static_alpha=False,
    disable_length_contribution=True,   # w1 -> 0
    disable_noise_gate=False,
)

VARIANT_D = AblationConfig(
    name="D_No_Gated_Threshold",
    use_static_alpha=False,
    disable_length_contribution=False,
    disable_noise_gate=True,            # N <= 0.20 guard removed
)

ALL_VARIANTS: List[AblationConfig] = [VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D]


# ---------------------------------------------------------------------------
# Variant-specific router implementations
# ---------------------------------------------------------------------------

class _StaticAlphaRouter(AdaptiveRouter):
    """Router that always returns a constant alpha (Variant B).

    Wraps AdaptiveRouter so the rest of the pipeline is unchanged.
    """

    def __init__(self, alpha: float, routing_config: RoutingConfig) -> None:
        super().__init__(routing_config)
        self._fixed_alpha = float(
            min(routing_config.alpha_max, max(routing_config.alpha_min, alpha))
        )

    def route(self, noise_score: float, token_length: int) -> RoutingDecision:
        n = float(noise_score)
        l = int(max(0, token_length))
        # z and alpha_raw still computed for record-keeping; alpha is overridden
        z = self.cfg.w1 * (self.cfg.reference_length - l) + self.cfg.w2 * n
        alpha_raw = stable_sigmoid(z)
        return RoutingDecision(
            alpha=self._fixed_alpha,
            alpha_raw=float(alpha_raw),
            z_score=float(z),
            noise_score=n,
            token_length=l,
            routing_state="static_alpha",
        )


class _NoLengthRouter(AdaptiveRouter):
    """Router where w1 = 0 — length has no influence on alpha (Variant C).

    z = 0 * (L0 - L) + w2 * N  =>  z = w2 * N
    All other routing logic (threshold gate, clamping) is preserved exactly.
    """

    def route(self, noise_score: float, token_length: int) -> RoutingDecision:
        n = float(noise_score)
        l = int(max(0, token_length))

        # w1 forced to 0: length term omitted
        z = self.cfg.w2 * n
        alpha_raw = stable_sigmoid(z)

        if n <= self.cfg.noise_threshold:
            alpha = self.cfg.alpha_min
            state = "low_noise_default"
        else:
            clamped = min(self.cfg.alpha_max, max(self.cfg.alpha_min, float(alpha_raw)))
            alpha = clamped
            if alpha_raw <= self.cfg.alpha_min:
                state = "clamped_min"
            elif alpha_raw >= self.cfg.alpha_max:
                state = "clamped_max"
            else:
                state = "adaptive_active"

        final_alpha = min(self.cfg.alpha_max, max(self.cfg.alpha_min, alpha))
        return RoutingDecision(
            alpha=final_alpha,
            alpha_raw=float(alpha_raw),
            z_score=float(z),
            noise_score=n,
            token_length=l,
            routing_state=state,
        )


class _NoGateRouter(AdaptiveRouter):
    """Router without the N <= 0.20 guard — sigmoid always active (Variant D).

    The N <= 0.20 early-return is removed. All other constants and clamping
    logic are preserved exactly from the specification.
    """

    def route(self, noise_score: float, token_length: int) -> RoutingDecision:
        n = float(noise_score)
        l = int(max(0, token_length))

        z = self.cfg.w1 * (self.cfg.reference_length - l) + self.cfg.w2 * n
        alpha_raw = stable_sigmoid(z)

        # No threshold gate — sigmoid always active
        clamped = min(self.cfg.alpha_max, max(self.cfg.alpha_min, float(alpha_raw)))
        if alpha_raw <= self.cfg.alpha_min:
            state = "clamped_min"
        elif alpha_raw >= self.cfg.alpha_max:
            state = "clamped_max"
        else:
            state = "adaptive_active"

        return RoutingDecision(
            alpha=clamped,
            alpha_raw=float(alpha_raw),
            z_score=float(z),
            noise_score=n,
            token_length=l,
            routing_state=state,
        )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def build_router(ablation_cfg: AblationConfig) -> AdaptiveRouter:
    """Instantiate the correct router for a given ablation configuration."""
    rc = ablation_cfg.routing_config
    if ablation_cfg.use_static_alpha:
        return _StaticAlphaRouter(alpha=ablation_cfg.static_alpha, routing_config=rc)
    if ablation_cfg.disable_length_contribution:
        return _NoLengthRouter(routing_config=rc)
    if ablation_cfg.disable_noise_gate:
        return _NoGateRouter(routing_config=rc)
    # Default: full AdaptiveRouter (Variant A)
    return AdaptiveRouter(routing_config=rc)


# ---------------------------------------------------------------------------
# Ablation result record
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Metrics for one ablation variant."""
    variant_name: str
    total_samples: int
    accuracy: float
    precision: float
    recall: float
    binary_f1: float
    # Delta relative to Variant A (Full Dynamic Fusion); None for Variant A itself
    f1_delta: Optional[float]

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "variant": self.variant_name,
            "samples": self.total_samples,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "binary_f1": self.binary_f1,
            "f1_delta_vs_full": (
                "" if self.f1_delta is None
                else round(self.f1_delta, 6)
            ),
        }


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

class AblationRunner:
    """Runs the ablation study and writes results/ablation_results.csv.

    Reuses EvaluationEngine.evaluate_single for all metric computation.
    Does not duplicate any fusion or routing logic.

    Parameters
    ----------
    vader_model:
        Shared VADER instance (loaded once).
    distilbert_model:
        Shared DistilBERT instance (loaded once).
    noise_quantifier:
        Shared NoiseQuantifier instance.
    preprocessor:
        Shared TextPreprocessor instance.
    output_dir:
        Directory where ablation_results.csv will be written.
    threshold:
        Binary classification decision threshold (default 0.50).
    variants:
        Ablation variant configurations to evaluate. Defaults to all four.
    """

    def __init__(
        self,
        vader_model: Optional[VaderSentimentModel] = None,
        distilbert_model: Optional[DistilBertSentimentModel] = None,
        noise_quantifier: Optional[NoiseQuantifier] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        output_dir: Union[str, Path] = "results",
        threshold: float = 0.50,
        variants: Optional[List[AblationConfig]] = None,
    ) -> None:
        # Shared component instances — loaded once, reused across all variants
        self._vader = vader_model or VaderSentimentModel()
        self._distilbert = distilbert_model or DistilBertSentimentModel()
        self._noise_quantifier = noise_quantifier or NoiseQuantifier()
        self._preprocessor = preprocessor or TextPreprocessor()
        self.output_dir = Path(output_dir)
        self.threshold = float(threshold)
        self.variants = variants if variants is not None else ALL_VARIANTS

    def _build_fusion_model(self, ablation_cfg: AblationConfig) -> DynamicFusionFramework:
        """Construct a DynamicFusionFramework with the variant-specific router.

        Shared model instances are injected — no reload per variant.
        """
        router = build_router(ablation_cfg)
        return DynamicFusionFramework(
            vader_model=self._vader,
            distilbert_model=self._distilbert,
            noise_quantifier=self._noise_quantifier,
            router=router,
            preprocessor=self._preprocessor,
            routing_config=ablation_cfg.routing_config,
        )

    def run(
        self,
        texts: List[str],
        labels: List[int],
        sample_ids: Optional[List[Union[int, str]]] = None,
    ) -> List[AblationResult]:
        """Evaluate all variants and return structured AblationResult objects.

        Args:
            texts:       Test-set texts (must be the held-out test split only).
            labels:      Ground-truth binary labels.
            sample_ids:  Optional stable sample identifiers.

        Returns:
            List of AblationResult, one per variant, in registration order.
        """
        engine = EvaluationEngine(
            output_dir=self.output_dir,
            threshold=self.threshold,
            experiment_name="ablation_study",
        )

        model_results: List[ModelResult] = []
        for variant_cfg in self.variants:
            fusion_model = self._build_fusion_model(variant_cfg)
            result = engine.evaluate_single(
                name=variant_cfg.name,
                model=fusion_model,
                texts=texts,
                labels=labels,
                sample_ids=sample_ids,
            )
            model_results.append(result)

        # Compute F1 delta relative to Variant A (first registered variant)
        reference_f1 = model_results[0].binary_f1 if model_results else 0.0

        ablation_results: List[AblationResult] = []
        for i, mr in enumerate(model_results):
            delta = None if i == 0 else round(mr.binary_f1 - reference_f1, 6)
            ablation_results.append(
                AblationResult(
                    variant_name=mr.model_name,
                    total_samples=mr.total_samples,
                    accuracy=mr.accuracy,
                    precision=mr.precision,
                    recall=mr.recall,
                    binary_f1=mr.binary_f1,
                    f1_delta=delta,
                )
            )

        return ablation_results

    def save_csv(self, results: List[AblationResult]) -> Path:
        """Write results/ablation_results.csv."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "ablation_results.csv"

        fieldnames = [
            "variant", "samples", "accuracy", "precision",
            "recall", "binary_f1", "f1_delta_vs_full",
        ]
        rows = [r.to_csv_row() for r in results]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return out_path

    def run_and_save(
        self,
        texts: List[str],
        labels: List[int],
        sample_ids: Optional[List[Union[int, str]]] = None,
    ) -> Path:
        """Evaluate all variants and write results/ablation_results.csv.

        Returns the path to the written CSV file.
        """
        results = self.run(texts, labels, sample_ids)
        return self.save_csv(results)
