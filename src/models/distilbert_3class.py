"""DistilBERT 3-class configuration and output for SentiMix experiments.

This module EXTENDS distilbert_model.py for 3-class (positive/negative/neutral)
classification WITHOUT modifying the existing binary implementation.

Design decisions:
- num_labels = 3 maps to [positive=0, negative=1, neutral=2] per SENTIMIX_LABEL_STR2INT
- The existing binary DistilBertSentimentModel remains unchanged.
- This class is to be used ONLY when training on SentiMix; NOT for the binary fusion pipeline.

FUSION COMPATIBILITY WARNING:
The existing fusion formula S_final = alpha*S_VADER + (1-alpha)*S_DistilBERT operates
on scalar positive-class probability scores in [0, 1].  This formula cannot directly
apply to 3-class probability distributions [p_pos, p_neg, p_neutral] without a
methodological decision.  See SENTIMIX_INTEGRATION_REPORT.md for the documented
design gap.  DO NOT attempt 3-class fusion until this gap is resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.models.distilbert_model import ModelLoader
from src.models.base import BaseSentimentModel
from configs.config import config

# Label mapping mirrors src/data/sentimix_loader.py
LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1
LABEL_NEUTRAL  = 2


@dataclass(frozen=True)
class DistilBert3ClassOutput:
    """3-class structured output for SentiMix experiments.

    Fields:
        positive_prob:   P(positive) in [0, 1]
        negative_prob:   P(negative) in [0, 1]
        neutral_prob:    P(neutral)  in [0, 1]
        predicted_label: argmax class (0=positive, 1=negative, 2=neutral)
        logits:          Raw logits list [l_pos, l_neg, l_neu]
    """
    positive_prob:   float
    negative_prob:   float
    neutral_prob:    float
    predicted_label: int
    logits: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive_prob":   self.positive_prob,
            "negative_prob":   self.negative_prob,
            "neutral_prob":    self.neutral_prob,
            "predicted_label": self.predicted_label,
            "logits":          self.logits,
        }


class DistilBert3ClassModel:
    """DistilBERT 3-class sentiment model for SentiMix (positive / negative / neutral).

    Use ONLY for SentiMix experiments. The binary pipeline (DistilBertSentimentModel)
    remains active and unmodified for the original research fusion.

    Training configuration (do NOT train until Phase 6):
        epochs:        3
        batch_size:    16
        learning_rate: 2e-5
        max_length:    <= 128
    """

    def __init__(
        self,
        model_path_or_name: str = config.training.model_name,
        max_length: int = config.training.max_length,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
        lazy_load: bool = True,
    ) -> None:
        self.model_path_or_name = model_path_or_name
        self.max_length = min(max_length, 128)
        self.device = ModelLoader.resolve_device(device)
        self.tokenizer = tokenizer
        self.model = model
        self.num_labels = 3

        if not lazy_load and (self.tokenizer is None or self.model is None):
            self.load_model()

    def load_model(self) -> None:
        """Load 3-class tokenizer and model weights."""
        if self.tokenizer is None or self.model is None:
            self.tokenizer, self.model, self.device = ModelLoader.load_transformer(
                self.model_path_or_name,
                device=self.device,
                num_labels=self.num_labels,
            )

    def is_loaded(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def predict(self, text: Optional[str]) -> DistilBert3ClassOutput:
        """Run 3-class inference on a single text."""
        if text is None or not str(text).strip():
            # Uniform prior for empty input
            return DistilBert3ClassOutput(
                positive_prob=1 / 3,
                negative_prob=1 / 3,
                neutral_prob=1 / 3,
                predicted_label=LABEL_NEUTRAL,
                logits=[0.0, 0.0, 0.0],
            )

        if not self.is_loaded():
            self.load_model()

        inputs = self.tokenizer(
            str(text),
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_tensor = outputs.logits          # shape [1, 3]
            probs = F.softmax(logits_tensor, dim=-1)[0]  # [3]

        pos_prob = float(probs[LABEL_POSITIVE].item())
        neg_prob = float(probs[LABEL_NEGATIVE].item())
        neu_prob = float(probs[LABEL_NEUTRAL].item())
        predicted = int(torch.argmax(probs).item())
        raw_logits = [float(x) for x in logits_tensor[0].tolist()]

        return DistilBert3ClassOutput(
            positive_prob=pos_prob,
            negative_prob=neg_prob,
            neutral_prob=neu_prob,
            predicted_label=predicted,
            logits=raw_logits,
        )

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[DistilBert3ClassOutput]:
        """Run 3-class inference on a batch of texts."""
        if not texts:
            return []
        if not self.is_loaded():
            self.load_model()

        results: List[DistilBert3ClassOutput] = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if t else "" for t in texts[i: i + batch_size]]
            inputs = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
                probs_batch = F.softmax(out.logits, dim=-1)
                for j in range(len(batch)):
                    p = probs_batch[j]
                    results.append(DistilBert3ClassOutput(
                        positive_prob=float(p[LABEL_POSITIVE].item()),
                        negative_prob=float(p[LABEL_NEGATIVE].item()),
                        neutral_prob=float(p[LABEL_NEUTRAL].item()),
                        predicted_label=int(torch.argmax(p).item()),
                        logits=[float(x) for x in out.logits[j].tolist()],
                    ))
        return results
