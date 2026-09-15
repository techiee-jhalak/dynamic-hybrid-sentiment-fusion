"""Dynamic Hybrid 3-Class Vector Fusion Framework for SentiMix Hinglish.

Extends the paper's scalar fusion to a 3-class probability simplex Delta^2:
    P_final = alpha * P_VADER + (1 - alpha) * P_DistilBERT

Prediction Rule:
    y_hat = argmax(P_final) in {0: positive, 1: negative, 2: neutral}

METHODOLOGICAL FIDELITY:
- The router formula and constants (L0=20, w1=0.05, w2=12.0, threshold=0.20,
  alpha in [0.02, 0.25]) are PRESERVED EXACTLY.
- Noise features (E, R, C, S) and composite score N are PRESERVED EXACTLY.
- No learned fusion parameters or neural gating networks are introduced.
- Alpha strictly maintains its role: weight of lexicon contribution.
- The original binary pipeline in src/models/dynamic_fusion.py is UNTOUCHED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from configs.config import RoutingConfig, config
from src.data.preprocessor import TextPreprocessor, PreprocessingResult
from src.features.noise_quantifier import NoiseQuantifier, NoiseFeatures
from src.models.adaptive_router import AdaptiveRouter, RoutingDecision
from src.models.vader_3class import (
    VADER3ClassAdapter,
    Vader3ClassOutput,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    CLASS_NAMES,
)
from src.models.distilbert_3class import (
    DistilBert3ClassModel,
    DistilBert3ClassOutput,
)


@dataclass(frozen=True)
class Fusion3ClassResult:
    """Structured container for 3-class dynamic fusion inference output."""
    p_vader: List[float]          # [p_pos, p_neg, p_neu] from VADER adapter
    p_distilbert: List[float]     # [p_pos, p_neg, p_neu] from DistilBERT
    p_final: List[float]          # Fused vector: alpha * p_vader + (1 - alpha) * p_distilbert
    alpha: float                  # Dynamic or static weight in [0.0, 1.0]
    predicted_label: int          # 0 (positive), 1 (negative), 2 (neutral)
    sentiment_label: str          # "positive", "negative", or "neutral"
    confidence: float             # Max probability in p_final in [1/3, 1.0]
    noise_score: float            # Composite noise score N in [0, 1]
    token_length: int             # Token count L
    mode: str                     # "dynamic", "static", "distilbert_only", "vader_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_vader": list(self.p_vader),
            "p_distilbert": list(self.p_distilbert),
            "p_final": list(self.p_final),
            "alpha": self.alpha,
            "predicted_label": self.predicted_label,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "noise_score": self.noise_score,
            "token_length": self.token_length,
            "mode": self.mode,
        }


def fuse_3class_vectors(
    p_vader: Sequence[float],
    p_distilbert: Sequence[float],
    alpha: float,
    mode: str = "dynamic",
    noise_score: float = 0.0,
    token_length: int = 0,
) -> Fusion3ClassResult:
    """Execute 3-class vector interpolation on the probability simplex.

    Formula:
        P_final = alpha * P_VADER + (1 - alpha) * P_DistilBERT
        y_hat = argmax(P_final)
    """
    if len(p_vader) != 3:
        raise ValueError(f"p_vader must have length 3, got {len(p_vader)}")
    if len(p_distilbert) != 3:
        raise ValueError(f"p_distilbert must have length 3, got {len(p_distilbert)}")

    # Clamp alpha to valid interval [0.0, 1.0]
    clamped_alpha = min(1.0, max(0.0, float(alpha)))

    # Compute convex combination component-wise
    p_final = [
        clamped_alpha * float(pv) + (1.0 - clamped_alpha) * float(pd)
        for pv, pd in zip(p_vader, p_distilbert)
    ]

    # Ensure numerical stability: re-normalize to strictly sum to 1.0
    tot = sum(p_final)
    if tot > 0.0:
        p_final = [float(x / tot) for x in p_final]
    else:
        p_final = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    # Argmax classification
    pred = int(np.argmax(p_final))
    label_str = CLASS_NAMES.get(pred, "unknown")
    conf = float(p_final[pred])

    return Fusion3ClassResult(
        p_vader=[float(x) for x in p_vader],
        p_distilbert=[float(x) for x in p_distilbert],
        p_final=p_final,
        alpha=clamped_alpha,
        predicted_label=pred,
        sentiment_label=label_str,
        confidence=conf,
        noise_score=float(noise_score),
        token_length=int(token_length),
        mode=mode,
    )


class DynamicFusion3ClassFramework:
    """3-Class Multi-Stage Dynamic Fusion Architecture for SentiMix Hinglish.

    Coordinates:
    1. Preprocessing (preserving emojis, repetition, code-mixing)
    2. Multi-Dimensional Noise Quantification (E, R, C, S, composite N)
    3. Adaptive Routing (alpha in [0.02, 0.25])
    4. VADER 3-Class Adapter (P_VADER in Delta^2)
    5. DistilBERT 3-Class Model (P_DistilBERT in Delta^2)
    6. Vector Fusion (P_final in Delta^2, y_hat = argmax(P_final))
    """

    def __init__(
        self,
        vader_adapter: Optional[VADER3ClassAdapter] = None,
        distilbert_model: Optional[DistilBert3ClassModel] = None,
        noise_quantifier: Optional[NoiseQuantifier] = None,
        router: Optional[AdaptiveRouter] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        routing_config: RoutingConfig = config.routing,
    ) -> None:
        self.vader_adapter = vader_adapter or VADER3ClassAdapter()
        self.distilbert_model = distilbert_model or DistilBert3ClassModel()
        self.noise_quantifier = noise_quantifier or NoiseQuantifier()
        self.router = router or AdaptiveRouter(routing_config)
        self.preprocessor = preprocessor or TextPreprocessor()
        self.cfg = routing_config

    def analyze(
        self,
        text: Optional[str],
        mode: str = "dynamic",
        fixed_alpha: float = 0.02,
    ) -> Dict[str, Any]:
        """Perform full multi-stage inference returning all intermediate and final outputs."""
        # 1. Conservative Preprocessing
        prep_res: PreprocessingResult = self.preprocessor.preprocess(text)

        # 2. Multi-Dimensional Noise Quantification
        noise_feats: NoiseFeatures = self.noise_quantifier.extract_features(prep_res.processed_text)

        # 3. Component Inferences
        vader_out: Vader3ClassOutput = self.vader_adapter.predict(prep_res.processed_text)
        distil_out: DistilBert3ClassOutput = self.distilbert_model.predict(prep_res.processed_text)

        # 4. Routing Determination based on mode
        if mode == "dynamic":
            routing_dec: RoutingDecision = self.router.route(
                noise_score=noise_feats.noise_score,
                token_length=prep_res.token_length,
            )
            alpha = routing_dec.alpha
            router_info = routing_dec.to_dict()
        elif mode == "static":
            alpha = min(self.cfg.alpha_max, max(self.cfg.alpha_min, float(fixed_alpha)))
            router_info = {"alpha": alpha, "routing_state": "static_fixed"}
        elif mode == "distilbert_only":
            alpha = 0.0
            router_info = {"alpha": 0.0, "routing_state": "distilbert_standalone"}
        elif mode == "vader_only":
            alpha = 1.0
            router_info = {"alpha": 1.0, "routing_state": "vader_standalone"}
        else:
            raise ValueError(f"Unknown fusion mode: '{mode}'. Expected dynamic, static, distilbert_only, or vader_only.")

        # 5. Vector Fusion
        p_vader = vader_out.scores_vector
        p_distil = [distil_out.positive_prob, distil_out.negative_prob, distil_out.neutral_prob]

        fusion_res = fuse_3class_vectors(
            p_vader=p_vader,
            p_distilbert=p_distil,
            alpha=alpha,
            mode=mode,
            noise_score=noise_feats.noise_score,
            token_length=prep_res.token_length,
        )

        return {
            "raw_text": prep_res.raw_text,
            "processed_text": prep_res.processed_text,
            "tokens": prep_res.tokens,
            "token_length": prep_res.token_length,
            "noise_features": noise_feats.to_dict(),
            "routing": router_info,
            "vader_3class": vader_out.to_dict(),
            "distilbert_3class": distil_out.to_dict(),
            "fusion": fusion_res.to_dict(),
            "prediction": fusion_res.predicted_label,
            "sentiment_label": fusion_res.sentiment_label,
            "confidence": fusion_res.confidence,
        }

    def predict(
        self,
        text: Optional[str],
        mode: str = "dynamic",
        fixed_alpha: float = 0.02,
    ) -> Fusion3ClassResult:
        """Run 3-class prediction and return Fusion3ClassResult."""
        analysis = self.analyze(text, mode=mode, fixed_alpha=fixed_alpha)
        f_dict = analysis["fusion"]
        return Fusion3ClassResult(**f_dict)

    def predict_batch(
        self,
        texts: List[str],
        mode: str = "dynamic",
        fixed_alpha: float = 0.02,
    ) -> List[Fusion3ClassResult]:
        """Run batch prediction."""
        return [self.predict(t, mode=mode, fixed_alpha=fixed_alpha) for t in texts]
