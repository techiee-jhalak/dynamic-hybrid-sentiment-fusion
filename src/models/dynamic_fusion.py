"""Dynamic Hybrid Fusion Framework for Code-Mixed Sentiment Analysis.

Combines rule-based lexicon probability S_vader and contextual transformer
probability S_distilbert using dynamic noise-aware weighting coefficient alpha:

S_final = alpha * S_vader + (1 - alpha) * S_distilbert

Classification Decision Rule:
if S_final >= 0.50:
    Positive (1)
else:
    Negative (0)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import numpy as np

from src.models.base import BaseSentimentModel
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel
from src.models.adaptive_router import AdaptiveRouter, RoutingDecision
from src.features.noise_quantifier import NoiseQuantifier, NoiseFeatures
from src.data.preprocessor import TextPreprocessor, PreprocessingResult
from configs.config import RoutingConfig, config


@dataclass(frozen=True)
class FusionResult:
    """Structured container for dynamic fusion inference output."""
    vader_probability: float          # S_vader in [0, 1]
    distilbert_probability: float     # S_distilbert in [0, 1]
    alpha: float                      # Dynamic weight alpha in [0.02, 0.25]
    final_score: float                # Fused sentiment score S_final in [0, 1]
    prediction: int                   # 0 (Negative) or 1 (Positive)
    sentiment_label: str              # "Negative" or "Positive"
    confidence: float                 # Classification confidence in [0.5, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "vader_probability": self.vader_probability,
            "distilbert_probability": self.distilbert_probability,
            "alpha": self.alpha,
            "final_score": self.final_score,
            "prediction": self.prediction,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
        }


def fuse_scores(
    s_vader: float,
    s_distilbert: float,
    alpha: float,
    decision_threshold: float = 0.50,
) -> FusionResult:
    """Execute dynamic interpolation formula and binary classification.

    Formula: S_final = alpha * S_vader + (1 - alpha) * S_distilbert
    """
    # 1. Validate inputs
    if not (0.0 <= s_vader <= 1.0):
        raise ValueError(f"s_vader probability must be in [0.0, 1.0]. Got: {s_vader}")
    if not (0.0 <= s_distilbert <= 1.0):
        raise ValueError(f"s_distilbert probability must be in [0.0, 1.0]. Got: {s_distilbert}")

    # Ensure alpha is strictly bounded to [alpha_min, alpha_max]
    alpha_clamped = min(0.25, max(0.02, float(alpha)))

    # 2. Dynamic Fusion Calculation
    s_final = alpha_clamped * float(s_vader) + (1.0 - alpha_clamped) * float(s_distilbert)
    s_final = min(1.0, max(0.0, float(s_final)))

    # 3. Binary Classification Decision
    prediction = 1 if s_final >= decision_threshold else 0
    label = "Positive" if prediction == 1 else "Negative"
    confidence = s_final if prediction == 1 else (1.0 - s_final)

    return FusionResult(
        vader_probability=float(s_vader),
        distilbert_probability=float(s_distilbert),
        alpha=alpha_clamped,
        final_score=s_final,
        prediction=prediction,
        sentiment_label=label,
        confidence=float(confidence),
    )


class DynamicFusionFramework(BaseSentimentModel):
    """End-to-end research architecture orchestrating preprocessing, noise extraction,

    VADER lexicon inference, DistilBERT contextual inference, adaptive routing, and fusion.
    """

    def __init__(
        self,
        vader_model: Optional[VaderSentimentModel] = None,
        distilbert_model: Optional[DistilBertSentimentModel] = None,
        noise_quantifier: Optional[NoiseQuantifier] = None,
        router: Optional[AdaptiveRouter] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        routing_config: RoutingConfig = config.routing,
    ) -> None:
        self.vader_model = vader_model or VaderSentimentModel()
        self.distilbert_model = distilbert_model or DistilBertSentimentModel()
        self.noise_quantifier = noise_quantifier or NoiseQuantifier()
        self.router = router or AdaptiveRouter(routing_config)
        self.preprocessor = preprocessor or TextPreprocessor()
        self.cfg = routing_config

    def analyze(self, text: Optional[str]) -> Dict[str, Any]:
        """Perform complete multi-stage analysis returning all intermediate and final outputs."""
        # 1. Conservative Preprocessing
        prep_res: PreprocessingResult = self.preprocessor.preprocess(text)

        # 2. Multi-Dimensional Noise Quantification
        noise_feats: NoiseFeatures = self.noise_quantifier.extract_features(prep_res.processed_text)

        # 3. Component Inferences
        s_vader = self.vader_model.predict_score(prep_res.processed_text)
        s_distilbert = self.distilbert_model.predict_score(prep_res.processed_text)

        # 4. Adaptive Routing
        routing_dec: RoutingDecision = self.router.route(
            noise_score=noise_feats.noise_score,
            token_length=prep_res.token_length,
        )

        # 5. Dynamic Fusion
        fusion_res: FusionResult = fuse_scores(
            s_vader=s_vader,
            s_distilbert=s_distilbert,
            alpha=routing_dec.alpha,
            decision_threshold=self.cfg.decision_threshold,
        )

        return {
            "raw_text": prep_res.raw_text,
            "processed_text": prep_res.processed_text,
            "tokens": prep_res.tokens,
            "token_length": prep_res.token_length,
            "noise_features": noise_feats.to_dict(),
            "routing": routing_dec.to_dict(),
            "fusion": fusion_res.to_dict(),
            "vader_score": s_vader,
            "distilbert_score": s_distilbert,
            "alpha": routing_dec.alpha,
            "fused_score": fusion_res.final_score,
            "prediction": fusion_res.prediction,
            "sentiment_label": fusion_res.sentiment_label,
            "confidence": fusion_res.confidence,
        }

    def predict_score(self, text: Optional[str]) -> float:
        """Predict continuous fused sentiment probability S_final in [0, 1]."""
        analysis = self.analyze(text)
        return analysis["fused_score"]

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        """Predict fused scores for a batch of texts."""
        return [self.predict_score(t) for t in texts]
