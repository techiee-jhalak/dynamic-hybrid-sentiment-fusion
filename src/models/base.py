"""Abstract base class for all sentiment analysis models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class BaseSentimentModel(ABC):
    """Abstract interface defining required contract for sentiment models."""

    @abstractmethod
    def predict_score(self, text: str) -> float:
        """Predict a continuous sentiment score S in [0, 1].

        S -> 0 is Negative, S -> 1 is Positive.
        """
        pass

    @abstractmethod
    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        """Predict continuous sentiment scores for a batch of texts."""
        pass

    def predict_label(self, text: str, threshold: float = 0.50) -> int:
        """Predict binary sentiment label (0: Negative, 1: Positive)."""
        score = self.predict_score(text)
        return 1 if score >= threshold else 0
