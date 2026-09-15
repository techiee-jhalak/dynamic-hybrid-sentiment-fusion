"""DistilBERT Contextual Sentiment Model Wrapper for Inference.

Implements Hugging Face Transformer inference with:
- PyTorch backend
- Automatic CPU / GPU acceleration
- Single-load persistent weights (no reload per request)
- torch.no_grad() inference evaluation
- Configurable checkpoint and max_length (<= 128)
- Safe handling of empty/invalid inputs
- Structured prediction container
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.base import BaseSentimentModel
from configs.config import config


@dataclass(frozen=True)
class DistilBertSentimentOutput:
    """Structured output container for DistilBERT sentiment predictions."""
    positive_prob: float      # S_distilbert in [0, 1]
    negative_prob: float      # 1 - S_distilbert in [0, 1]
    predicted_label: int      # 1 (Positive) or 0 (Negative)
    logits: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary."""
        return {
            "positive_prob": self.positive_prob,
            "negative_prob": self.negative_prob,
            "predicted_label": self.predicted_label,
            "logits": self.logits,
        }


class ModelLoader:
    """Utility for loading and initializing transformer tokenizers and model weights once."""

    @staticmethod
    def resolve_device(requested_device: Optional[str] = None) -> torch.device:
        """Resolve target computation device (CUDA if available, otherwise CPU)."""
        if requested_device:
            return torch.device(requested_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def load_transformer(
        cls,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        num_labels: int = 2,
    ) -> Tuple[Any, Any, torch.device]:
        """Load tokenizer and classification model onto the designated device."""
        target_device = device if isinstance(device, torch.device) else cls.resolve_device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
        )
        model.to(target_device)
        model.eval()
        return tokenizer, model, target_device


class DistilBertSentimentModel(BaseSentimentModel):
    """DistilBERT contextual sentiment inference engine providing S_distilbert in [0, 1]."""

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

        if not lazy_load and (self.tokenizer is None or self.model is None):
            self.load_model()

    def load_model(self) -> None:
        """Load tokenizer and model weights if not already initialized."""
        if self.tokenizer is None or self.model is None:
            self.tokenizer, self.model, self.device = ModelLoader.load_transformer(
                self.model_path_or_name,
                device=self.device,
                num_labels=2,
            )

    def is_loaded(self) -> bool:
        """Check whether tokenizer and model are ready in memory."""
        return self.tokenizer is not None and self.model is not None

    def predict_score(self, text: Optional[str]) -> float:
        """Compute positive-class probability S_distilbert in [0, 1]. Safe on empty/None input."""
        if text is None or not str(text).strip():
            return 0.50

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
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            pos_prob = float(probs[0, 1].item())

        return min(1.0, max(0.0, pos_prob))

    def predict(self, text: Optional[str], threshold: float = 0.50) -> DistilBertSentimentOutput:
        """Full contextual sentiment prediction returning positive/negative probabilities and label."""
        if text is None or not str(text).strip():
            return DistilBertSentimentOutput(
                positive_prob=0.50,
                negative_prob=0.50,
                predicted_label=1,
                logits=[0.0, 0.0],
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
            logits_tensor = outputs.logits
            probs = F.softmax(logits_tensor, dim=-1)
            pos_prob = min(1.0, max(0.0, float(probs[0, 1].item())))
            neg_prob = 1.0 - pos_prob
            raw_logits = [float(x) for x in logits_tensor[0].tolist()]

        predicted_label = 1 if pos_prob >= threshold else 0

        return DistilBertSentimentOutput(
            positive_prob=pos_prob,
            negative_prob=neg_prob,
            predicted_label=predicted_label,
            logits=raw_logits,
        )

    def predict_scores_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """Compute S_distilbert probabilities for a batch of texts."""
        if not texts:
            return []

        if not self.is_loaded():
            self.load_model()

        results: List[float] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if t is not None else "" for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                batch_pos_probs = probs[:, 1].tolist()
                results.extend([min(1.0, max(0.0, float(p))) for p in batch_pos_probs])

        return results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        threshold: float = 0.50,
    ) -> List[DistilBertSentimentOutput]:
        """Compute structured sentiment outputs for a batch of texts."""
        if not texts:
            return []

        if not self.is_loaded():
            self.load_model()

        outputs_list: List[DistilBertSentimentOutput] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if t is not None else "" for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)

                for j in range(len(batch_texts)):
                    pos_prob = min(1.0, max(0.0, float(probs[j, 1].item())))
                    neg_prob = 1.0 - pos_prob
                    label = 1 if pos_prob >= threshold else 0
                    raw_logits = [float(x) for x in logits[j].tolist()]

                    outputs_list.append(
                        DistilBertSentimentOutput(
                            positive_prob=pos_prob,
                            negative_prob=neg_prob,
                            predicted_label=label,
                            logits=raw_logits,
                        )
                    )

        return outputs_list
