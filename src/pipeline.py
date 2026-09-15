"""Unified End-to-End Inference Pipeline for Dynamic Hybrid Sentiment Fusion.

Orchestrates the complete research inference flow:
Input text -> Preprocessing -> Noise Quantification -> VADER -> DistilBERT -> Adaptive Router -> Dynamic Fusion -> Final Prediction.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import numpy as np

from src.data.preprocessor import TextPreprocessor, PreprocessingResult
from src.features.noise_quantifier import NoiseQuantifier, NoiseFeatures
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel
from src.models.adaptive_router import AdaptiveRouter, RoutingDecision
from src.models.dynamic_fusion import fuse_scores, FusionResult


@dataclass(frozen=True)
class PipelineResult:
    """Structured container for end-to-end inference results containing all intermediate and final outputs."""
    # Text metadata
    raw_text: str
    processed_text: str
    token_length: int

    # Noise features
    emoji_density: float
    repetition_score: float
    code_mixing_ratio: float
    symbol_density: float
    noise_score: float

    # Model probabilities
    vader_probability: float
    distilbert_probability: float

    # Adaptive routing
    alpha_raw: float
    alpha: float

    # Final fusion & prediction
    final_score: float
    prediction: int
    sentiment_label: str
    confidence: float
    routing_state: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline result to a flat dictionary."""
        return {
            "raw_text": self.raw_text,
            "processed_text": self.processed_text,
            "token_length": self.token_length,
            "emoji_density": self.emoji_density,
            "repetition_score": self.repetition_score,
            "code_mixing_ratio": self.code_mixing_ratio,
            "symbol_density": self.symbol_density,
            "noise_score": self.noise_score,
            "vader_probability": self.vader_probability,
            "distilbert_probability": self.distilbert_probability,
            "alpha_raw": self.alpha_raw,
            "alpha": self.alpha,
            "final_score": self.final_score,
            "prediction": self.prediction,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "routing_state": self.routing_state,
        }


class SentimentInferencePipeline:
    """Unified inference engine composing all modular components without code duplication."""

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        noise_quantifier: Optional[NoiseQuantifier] = None,
        vader_model: Optional[VaderSentimentModel] = None,
        distilbert_model: Optional[DistilBertSentimentModel] = None,
        router: Optional[AdaptiveRouter] = None,
    ) -> None:
        # Re-use existing instances or initialize singletons
        self.preprocessor = preprocessor or TextPreprocessor()
        self.noise_quantifier = noise_quantifier or NoiseQuantifier()
        self.vader_model = vader_model or VaderSentimentModel()
        self.distilbert_model = distilbert_model or DistilBertSentimentModel()
        self.router = router or AdaptiveRouter()

    def predict(self, text: Optional[str]) -> PipelineResult:
        """Run full end-to-end sentiment inference on a single text input."""
        # 1. Preprocessing
        prep_res: PreprocessingResult = self.preprocessor.preprocess(text)

        # 2. Multi-Dimensional Noise Quantification (E, R, C, S -> N)
        noise_feats: NoiseFeatures = self.noise_quantifier.extract_features(prep_res.processed_text)

        # 3. Model Predictions (S_vader, S_distilbert in [0, 1])
        s_vader = self.vader_model.predict_score(prep_res.processed_text)
        s_distil = self.distilbert_model.predict_score(prep_res.processed_text)

        # 4. Noise-Aware Adaptive Routing (alpha in [0.02, 0.25])
        routing_dec: RoutingDecision = self.router.route(
            noise_score=noise_feats.noise_score,
            token_length=prep_res.token_length,
        )

        # 5. Dynamic Hybrid Fusion & Final Classification
        fusion_res: FusionResult = fuse_scores(
            s_vader=s_vader,
            s_distilbert=s_distil,
            alpha=routing_dec.alpha,
        )

        # 6. Structured Result Assembly
        return PipelineResult(
            raw_text=prep_res.raw_text,
            processed_text=prep_res.processed_text,
            token_length=prep_res.token_length,
            emoji_density=noise_feats.emoji_density,
            repetition_score=noise_feats.repetition_score,
            code_mixing_ratio=noise_feats.code_mixing_ratio,
            symbol_density=noise_feats.symbol_density,
            noise_score=noise_feats.noise_score,
            vader_probability=fusion_res.vader_probability,
            distilbert_probability=fusion_res.distilbert_probability,
            alpha_raw=routing_dec.alpha_raw,
            alpha=routing_dec.alpha,
            final_score=fusion_res.final_score,
            prediction=fusion_res.prediction,
            sentiment_label=fusion_res.sentiment_label,
            confidence=fusion_res.confidence,
            routing_state=routing_dec.routing_state,
        )

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[PipelineResult]:
        """Run batched end-to-end sentiment inference."""
        if not texts:
            return []

        # 1. Batch Preprocessing
        prep_results = [self.preprocessor.preprocess(t) for t in texts]
        processed_texts = [p.processed_text for p in prep_results]

        # 2. Batch Noise Quantification
        noise_features_list = [self.noise_quantifier.extract_features(p.processed_text) for p in prep_results]

        # 3. Batch Model Predictions
        vader_scores = self.vader_model.predict_scores_batch(processed_texts)
        distil_scores = self.distilbert_model.predict_scores_batch(processed_texts, batch_size=batch_size)

        results: List[PipelineResult] = []
        for i, prep_res in enumerate(prep_results):
            noise_feats = noise_features_list[i]
            s_vader = vader_scores[i]
            s_distil = distil_scores[i]

            # 4. Adaptive Routing
            routing_dec = self.router.route(
                noise_score=noise_feats.noise_score,
                token_length=prep_res.token_length,
            )

            # 5. Dynamic Fusion
            fusion_res = fuse_scores(
                s_vader=s_vader,
                s_distilbert=s_distil,
                alpha=routing_dec.alpha,
            )

            results.append(
                PipelineResult(
                    raw_text=prep_res.raw_text,
                    processed_text=prep_res.processed_text,
                    token_length=prep_res.token_length,
                    emoji_density=noise_feats.emoji_density,
                    repetition_score=noise_feats.repetition_score,
                    code_mixing_ratio=noise_feats.code_mixing_ratio,
                    symbol_density=noise_feats.symbol_density,
                    noise_score=noise_feats.noise_score,
                    vader_probability=fusion_res.vader_probability,
                    distilbert_probability=fusion_res.distilbert_probability,
                    alpha_raw=routing_dec.alpha_raw,
                    alpha=routing_dec.alpha,
                    final_score=fusion_res.final_score,
                    prediction=fusion_res.prediction,
                    sentiment_label=fusion_res.sentiment_label,
                    confidence=fusion_res.confidence,
                    routing_state=routing_dec.routing_state,
                )
            )

        return results
