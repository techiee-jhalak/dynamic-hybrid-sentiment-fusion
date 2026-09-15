"""VADER Sentiment Lexicon Model Wrapper.

Extracts rule-based sentiment polarity from text and normalizes the compound score
from [-1, 1] to continuous probability S_vader in [0, 1] as required by PROJECT_SPEC.md.

Mapping:
S_vader = (compound + 1.0) / 2.0
- compound = -1.0 -> S_vader = 0.0 (Strongly Negative)
- compound =  0.0 -> S_vader = 0.5 (Neutral)
- compound = +1.0 -> S_vader = 1.0 (Strongly Positive)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.models.base import BaseSentimentModel


@dataclass(frozen=True)
class VaderSentimentOutput:
    """Structured container for VADER sentiment prediction results."""
    positive_prob: float      # S_vader in [0, 1]
    negative_prob: float      # 1 - S_vader in [0, 1]
    predicted_label: int      # 1 (Positive) or 0 (Negative)
    compound_score: float     # Raw compound in [-1, 1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary."""
        return {
            "positive_prob": self.positive_prob,
            "negative_prob": self.negative_prob,
            "predicted_label": self.predicted_label,
            "compound_score": self.compound_score,
        }


class VaderSentimentModel(BaseSentimentModel):
    """Singleton-ready VADER sentiment analyzer providing continuous probability S_vader."""

    def __init__(self) -> None:
        """Initialize NLTK VADER analyzer once and reuse across all predictions."""
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            self.analyzer = SentimentIntensityAnalyzer()

    def get_compound_score(self, text: Optional[str]) -> float:
        """Extract raw VADER compound score in [-1.0, 1.0]. Safe on empty/None input."""
        if text is None:
            return 0.0
        text_str = str(text).strip()
        if not text_str:
            return 0.0
        scores = self.analyzer.polarity_scores(text_str)
        return float(scores.get("compound", 0.0))

    def predict_score(self, text: Optional[str]) -> float:
        """Compute positive-class probability S_vader in [0, 1] from compound score.

        Formula: S_vader = (compound + 1.0) / 2.0
        """
        compound = self.get_compound_score(text)
        s_vader = (compound + 1.0) / 2.0
        return min(1.0, max(0.0, float(s_vader)))

    def predict(self, text: Optional[str], threshold: float = 0.50) -> VaderSentimentOutput:
        """Perform full sentiment prediction returning probabilities, label, and compound score."""
        compound = self.get_compound_score(text)
        pos_prob = (compound + 1.0) / 2.0
        pos_prob = min(1.0, max(0.0, float(pos_prob)))
        neg_prob = 1.0 - pos_prob
        predicted_label = 1 if pos_prob >= threshold else 0

        return VaderSentimentOutput(
            positive_prob=pos_prob,
            negative_prob=neg_prob,
            predicted_label=predicted_label,
            compound_score=compound,
        )

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        """Compute continuous S_vader probabilities for a batch of texts."""
        return [self.predict_score(t) for t in texts]

    def predict_batch(self, texts: List[str], threshold: float = 0.50) -> List[VaderSentimentOutput]:
        """Compute structured sentiment outputs for a batch of texts."""
        return [self.predict(t, threshold=threshold) for t in texts]
