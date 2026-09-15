"""VADER 3-Class Adapter for SentiMix Hindi-English Code-Mixed Sentiment.

This module provides a strictly deterministic, non-learned adapter that converts
VADER's native lexical polarity proportions into a 3-class score representation
aligned with the SentiMix label scheme:
    LABEL_POSITIVE = 0
    LABEL_NEGATIVE = 1
    LABEL_NEUTRAL  = 2

IMPORTANT METHODOLOGICAL DECLARATION:
- This adapter is NOT a learned or calibrated probability model.
- It introduces ZERO learned parameters and ZERO dataset-fitted thresholds.
- It uses ONLY the native lexical polarity proportions ('pos', 'neg', 'neu') computed
  by NLTK VADER's rule-based lexicon analyzer.
- The resulting scores sum strictly to 1.0 (L1 normalized) and form a deterministic
  vector in the 2-simplex Delta^2 for convex vector fusion.
- The existing binary VADER model (src/models/vader_model.py) remains completely
  unmodified and isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Canonical class order matching SentiMix CoNLL specifications
LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1
LABEL_NEUTRAL  = 2

CLASS_NAMES: Dict[int, str] = {
    LABEL_POSITIVE: "positive",
    LABEL_NEGATIVE: "negative",
    LABEL_NEUTRAL:  "neutral",
}


@dataclass(frozen=True)
class Vader3ClassOutput:
    """Structured container for 3-class VADER polarity scores.

    Attributes:
        positive_prob:   Normalized positive proportion in [0, 1]
        negative_prob:   Normalized negative proportion in [0, 1]
        neutral_prob:    Normalized neutral proportion in [0, 1]
        predicted_label: Argmax class: 0 (positive), 1 (negative), 2 (neutral)
        scores_vector:   Deterministic [p_pos, p_neg, p_neu] summing to 1.0
        compound_score:  Raw VADER continuous compound score in [-1.0, 1.0]
    """
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    predicted_label: int
    scores_vector: List[float]
    compound_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive_prob": self.positive_prob,
            "negative_prob": self.negative_prob,
            "neutral_prob": self.neutral_prob,
            "predicted_label": self.predicted_label,
            "scores_vector": list(self.scores_vector),
            "compound_score": self.compound_score,
            "sentiment_label": CLASS_NAMES.get(self.predicted_label, "unknown"),
        }


class VADER3ClassAdapter:
    """Deterministic, uncalibrated 3-class adapter for VADER lexicon outputs.

    Extracts native VADER polarity proportions {'pos', 'neg', 'neu'} and L1-normalizes
    them into a 3-element score vector ordered as [positive, negative, neutral].
    """

    def __init__(self, analyzer: Optional[SentimentIntensityAnalyzer] = None) -> None:
        if analyzer is not None:
            self.analyzer = analyzer
        else:
            try:
                self.analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: Optional[str]) -> Vader3ClassOutput:
        """Compute deterministic 3-class score vector from text.

        Handles empty/None strings gracefully by assigning 100% weight to neutral.
        """
        if text is None:
            return self._neutral_fallback(0.0)

        text_str = str(text).strip()
        if not text_str:
            return self._neutral_fallback(0.0)

        raw_scores = self.analyzer.polarity_scores(text_str)
        pos = float(raw_scores.get("pos", 0.0))
        neg = float(raw_scores.get("neg", 0.0))
        neu = float(raw_scores.get("neu", 0.0))
        compound = float(raw_scores.get("compound", 0.0))

        tot = pos + neg + neu
        if tot <= 0.0:
            return self._neutral_fallback(compound)

        # L1-normalize to ensure the scores sum strictly to 1.0
        p_pos = pos / tot
        p_neg = neg / tot
        p_neu = neu / tot

        # Determine argmax prediction (ties broken deterministically: pos, neg, neu)
        scores_list = [p_pos, p_neg, p_neu]
        # Argmax implementation
        best_idx = 0
        best_val = scores_list[0]
        for idx in (1, 2):
            if scores_list[idx] > best_val:
                best_val = scores_list[idx]
                best_idx = idx

        return Vader3ClassOutput(
            positive_prob=p_pos,
            negative_prob=p_neg,
            neutral_prob=p_neu,
            predicted_label=best_idx,
            scores_vector=scores_list,
            compound_score=compound,
        )

    def predict_scores(self, text: Optional[str]) -> List[float]:
        """Return raw 3-element vector [p_pos, p_neg, p_neu]."""
        return self.predict(text).scores_vector

    def predict_batch(self, texts: List[str]) -> List[Vader3ClassOutput]:
        """Compute structured outputs for a sequence of texts."""
        return [self.predict(t) for t in texts]

    def predict_scores_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute 3-element score vectors for a sequence of texts."""
        return [self.predict_scores(t) for t in texts]

    @staticmethod
    def _neutral_fallback(compound: float) -> Vader3ClassOutput:
        """Deterministic neutral representation for empty or uninformative inputs."""
        return Vader3ClassOutput(
            positive_prob=0.0,
            negative_prob=0.0,
            neutral_prob=1.0,
            predicted_label=LABEL_NEUTRAL,
            scores_vector=[0.0, 0.0, 1.0],
            compound_score=compound,
        )
