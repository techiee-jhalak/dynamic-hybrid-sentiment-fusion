"""Baseline sentiment analysis models for comparative benchmarking.

Includes:
1. Logistic Regression (TF-IDF unigrams + bigrams)
2. Static Fusion (Fixed-weight lexicon-transformer baseline)
3. BERTweet (Optional social media domain baseline, disabled by default)
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseSentimentModel
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel, ModelLoader


class LogisticRegressionBaseline(BaseSentimentModel):
    """TF-IDF (1-2 n-grams) + Logistic Regression benchmark model."""

    def __init__(
        self,
        ngram_range: tuple = (1, 2),
        max_features: int = 10000,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=C,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> "LogisticRegressionBaseline":
        """Fit TF-IDF vectorizer and Logistic Regression classifier."""
        if not texts or not labels:
            raise ValueError("Training texts and labels cannot be empty.")

        cleaned_texts = [str(t) if t is not None else "" for t in texts]
        X = self.vectorizer.fit_transform(cleaned_texts)
        self.classifier.fit(X, np.array(labels, dtype=int))
        self.is_fitted = True
        return self

    def predict_score(self, text: Optional[str]) -> float:
        """Predict continuous positive probability in [0, 1]. Safe on empty/None input."""
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionBaseline must be fitted before predicting.")

        if text is None or not str(text).strip():
            return 0.50

        X = self.vectorizer.transform([str(text)])
        probs = self.classifier.predict_proba(X)
        return float(probs[0, 1])

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        """Predict continuous positive probabilities for a batch of texts."""
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionBaseline must be fitted before predicting.")

        if not texts:
            return []

        cleaned_texts = [str(t) if t is not None else "" for t in texts]
        X = self.vectorizer.transform(cleaned_texts)
        probs = self.classifier.predict_proba(X)
        return [float(p[1]) for p in probs]


class StaticFusionBaseline(BaseSentimentModel):
    """Fixed-weight Lexicon-Transformer fusion baseline (e.g. constant alpha = 0.15)."""

    def __init__(
        self,
        vader_model: Optional[VaderSentimentModel] = None,
        distilbert_model: Optional[DistilBertSentimentModel] = None,
        fixed_alpha: float = 0.15,
    ) -> None:
        self.vader_model = vader_model or VaderSentimentModel()
        self.distilbert_model = distilbert_model or DistilBertSentimentModel()
        self.fixed_alpha = min(1.0, max(0.0, float(fixed_alpha)))

    def predict_score(self, text: Optional[str]) -> float:
        """Predict fused sentiment score with constant fixed alpha."""
        s_vader = self.vader_model.predict_score(text)
        s_distil = self.distilbert_model.predict_score(text)
        s_static = self.fixed_alpha * s_vader + (1.0 - self.fixed_alpha) * s_distil
        return min(1.0, max(0.0, float(s_static)))

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        """Predict fused sentiment scores for a batch of texts with constant fixed alpha."""
        if not texts:
            return []
        vader_scores = self.vader_model.predict_scores_batch(texts)
        distil_scores = self.distilbert_model.predict_scores_batch(texts)

        results = [
            min(1.0, max(0.0, float(self.fixed_alpha * v + (1.0 - self.fixed_alpha) * d)))
            for v, d in zip(vader_scores, distil_scores)
        ]
        return results


class BERTweetBaseline(BaseSentimentModel):
    """Optional social media domain baseline (BERTweet).

    Kept disabled by default; does not download or execute models automatically.
    """

    def __init__(
        self,
        model_name_or_path: str = "vinai/bertweet-base",
        enabled: bool = False,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.enabled = enabled
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

    def enable(self) -> None:
        """Explicitly enable the optional BERTweet baseline."""
        self.enabled = True

    def disable(self) -> None:
        """Disable BERTweet baseline."""
        self.enabled = False

    def load_model(self) -> None:
        """Load BERTweet weights only when explicitly enabled."""
        if not self.enabled:
            raise RuntimeError(
                "BERTweetBaseline is disabled by default per research specification. "
                "Call enable() explicitly if you wish to initialize it."
            )
        if self.tokenizer is None or self.model is None:
            self.tokenizer, self.model, self.device = ModelLoader.load_transformer(
                self.model_name_or_path,
                device=self.device,
                num_labels=2,
            )

    def predict_score(self, text: Optional[str]) -> float:
        """Predict sentiment score with BERTweet."""
        if not self.enabled:
            raise RuntimeError("BERTweetBaseline is disabled by default.")
        if text is None or not str(text).strip():
            return 0.50
        if self.tokenizer is None or self.model is None:
            self.load_model()

        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            str(text),
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pos_prob = float(probs[0, 1].item())

        return min(1.0, max(0.0, pos_prob))

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        """Batch predict sentiment scores with BERTweet."""
        if not self.enabled:
            raise RuntimeError("BERTweetBaseline is disabled by default.")
        return [self.predict_score(t) for t in texts]
