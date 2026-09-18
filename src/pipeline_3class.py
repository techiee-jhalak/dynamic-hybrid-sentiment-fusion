"""Unified End-to-End 3-Class Inference Pipeline for SentiMix Hinglish.

Orchestrates full production inference flow:
Input text -> Preprocessing -> Multi-Dimensional Noise Quantification (E, R, C, S, N)
-> Noise Band Classification -> VADER 3-Class Adapter -> DistilBERT 3-Class Model
-> Adaptive Router (alpha) -> Dynamic Vector Fusion -> Final 3-Class Prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from configs.config import config
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
from src.models.fusion_3class import (
    DynamicFusion3ClassFramework,
    Fusion3ClassResult,
    fuse_3class_vectors,
)
from src.evaluation.noise_sensitivity import assign_noise_group


@dataclass(frozen=True)
class Sentimix3ClassPipelineResult:
    """Structured container for end-to-end 3-class inference results."""
    # Text metadata
    raw_text: str
    processed_text: str
    token_length: int

    # Noise features (E, R, C, S, N)
    emoji_density: float
    repetition_score: float
    code_mixing_ratio: float
    symbol_density: float
    noise_score: float
    noise_band: str  # "LOW", "MODERATE", "HIGH", "EXTREME"

    # Vectors on Delta^2
    vader_vector: List[float]       # [p_pos, p_neg, p_neu]
    distilbert_vector: List[float]  # [p_pos, p_neg, p_neu]
    fused_vector: List[float]       # [p_pos, p_neg, p_neu]

    # Adaptive routing
    alpha: float
    routing_state: str

    # Prediction
    predicted_label: str            # "positive", "negative", "neutral"
    predicted_class_index: int      # 0, 1, 2
    confidence: float               # max(fused_vector)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "processed_text": self.processed_text,
            "token_length": self.token_length,
            "emoji_density": self.emoji_density,
            "repetition_score": self.repetition_score,
            "code_mixing_ratio": self.code_mixing_ratio,
            "symbol_density": self.symbol_density,
            "noise_score": self.noise_score,
            "noise_band": self.noise_band,
            "vader_vector": list(self.vader_vector),
            "distilbert_vector": list(self.distilbert_vector),
            "fused_vector": list(self.fused_vector),
            "alpha": self.alpha,
            "routing_state": self.routing_state,
            "predicted_label": self.predicted_label,
            "predicted_class_index": self.predicted_class_index,
            "confidence": self.confidence,
        }


class Sentimix3ClassInferencePipeline:
    """Unified production inference engine for SentiMix 3-class sentiment analysis."""

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        noise_quantifier: Optional[NoiseQuantifier] = None,
        vader_adapter: Optional[VADER3ClassAdapter] = None,
        distilbert_model: Optional[DistilBert3ClassModel] = None,
        router: Optional[AdaptiveRouter] = None,
    ) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self.noise_quantifier = noise_quantifier or NoiseQuantifier()
        self.vader_adapter = vader_adapter or VADER3ClassAdapter()
        self.router = router or AdaptiveRouter()

        ckpt = checkpoint_dir or (Path(config.paths.saved_models_dir) / "sentimix_distilbert_best")
        if distilbert_model is not None:
            self.distilbert_model = distilbert_model
        else:
            self.distilbert_model = DistilBert3ClassModel(
                model_path_or_name=str(ckpt) if Path(ckpt).exists() else config.training.model_name,
                lazy_load=True,
            )

    def predict(
        self,
        text: Optional[str],
        mode: str = "dynamic",
        fixed_alpha: float = 0.02,
    ) -> Sentimix3ClassPipelineResult:
        """Run full end-to-end 3-class sentiment inference on a single text input."""
        # 1. Preprocessing
        prep_res: PreprocessingResult = self.preprocessor.preprocess(text)

        # 2. Multi-Dimensional Noise Quantification
        noise_feats: NoiseFeatures = self.noise_quantifier.extract_features(prep_res.processed_text)
        n = noise_feats.noise_score
        band = assign_noise_group(n)

        # 3. Model Predictions
        vader_out: Vader3ClassOutput = self.vader_adapter.predict(prep_res.processed_text)
        distil_out: DistilBert3ClassOutput = self.distilbert_model.predict(prep_res.processed_text)

        # 4. Routing Determination
        if mode == "dynamic":
            routing_dec: RoutingDecision = self.router.route(
                noise_score=n,
                token_length=prep_res.token_length,
            )
            alpha = routing_dec.alpha
            state = routing_dec.routing_state
        elif mode == "static":
            alpha = min(0.25, max(0.02, float(fixed_alpha)))
            state = "static_fixed"
        elif mode == "distilbert_only":
            alpha = 0.0
            state = "distilbert_standalone"
        elif mode == "vader_only":
            alpha = 1.0
            state = "vader_standalone"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 5. Dynamic Vector Fusion on Delta^2
        p_vader = vader_out.scores_vector
        p_distil = [distil_out.positive_prob, distil_out.negative_prob, distil_out.neutral_prob]

        fusion_res: FusionResult = fuse_3class_vectors(
            p_vader=p_vader,
            p_distilbert=p_distil,
            alpha=alpha,
            mode=mode,
            noise_score=n,
            token_length=prep_res.token_length,
        )

        return Sentimix3ClassPipelineResult(
            raw_text=prep_res.raw_text,
            processed_text=prep_res.processed_text,
            token_length=prep_res.token_length,
            emoji_density=noise_feats.emoji_density,
            repetition_score=noise_feats.repetition_score,
            code_mixing_ratio=noise_feats.code_mixing_ratio,
            symbol_density=noise_feats.symbol_density,
            noise_score=noise_feats.noise_score,
            noise_band=band,
            vader_vector=p_vader,
            distilbert_vector=p_distil,
            fused_vector=fusion_res.p_final,
            alpha=alpha,
            routing_state=state,
            predicted_label=fusion_res.sentiment_label,
            predicted_class_index=fusion_res.predicted_label,
            confidence=fusion_res.confidence,
        )

    def predict_batch(
        self,
        texts: List[str],
        mode: str = "dynamic",
        fixed_alpha: float = 0.02,
    ) -> List[Sentimix3ClassPipelineResult]:
        """Run batched 3-class inference."""
        return [self.predict(t, mode=mode, fixed_alpha=fixed_alpha) for t in texts]
