"""Deterministic Error Analysis Module for Dynamic Hybrid Sentiment Fusion.

Categorizes prediction errors strictly based on deterministic linguistic & structural rules:
1. severe cross-script code-mixing: Devanagari alongside Latin script, or high C-index (C >= 0.40).
2. dual-sarcasm/polysemous emojis: Discordance between emoji polarity and text/label sentiment.
3. structural/implicit irony: Discourse contrastive shifts ("but", "however", "lekin") or ironic quotes.
4. slang/syntax drift: Severe character elongation (R >= 0.35) or heavy informal slang tokens.
5. unavailable: Assigned whenever deterministic patterns do NOT provide confident attribution.
   (Never invents or fabricates explanations).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import emoji

from src.features.noise_quantifier import (
    NoiseQuantifier,
    DEVANAGARI_PATTERN,
    HINGLISH_MARKERS,
    PUNCTUATION_SYMBOLS,
)


CAT_CROSS_SCRIPT = "severe cross-script code-mixing"
CAT_DUAL_SARCASM = "dual-sarcasm/polysemous emojis"
CAT_IMPLICIT_IRONY = "structural/implicit irony"
CAT_SLANG_DRIFT = "slang/syntax drift"
CAT_UNAVAILABLE = "unavailable"

ALL_ERROR_CATEGORIES = [
    CAT_CROSS_SCRIPT,
    CAT_DUAL_SARCASM,
    CAT_IMPLICIT_IRONY,
    CAT_SLANG_DRIFT,
    CAT_UNAVAILABLE,
]

# Emojis commonly associated with positive/laughing sentiment
POSITIVE_LAUGHING_EMOJIS = {
    "😍", "🔥", "❤️", "🎉", "👏", "💯", "🙌", "✨", "🥰", "👍", "😁", "😃", "😂", "🤣", "😆",
}

# Emojis commonly associated with negative/angry/sad sentiment
NEGATIVE_EMOJIS = {
    "😡", "🤬", "😢", "😭", "💔", "🤮", "🤢", "👎", "😠", "😞", "😒", "🙄", "💩",
}

# Contrastive discourse markers
CONTRASTIVE_MARKERS = {
    "but", "however", "though", "although", "yet", "still", "except", "lekin", "magar", "par",
}

# Sarcastic / ironic cues
IRONY_MARKERS = [
    re.compile(r'\b(yeah right|as if|slow claps?|slow-clap|what a joke|sarcasm|wah kya)\b', re.IGNORECASE),
    re.compile(r'["\'](masterpiece|best|great|genius|amazing|superhit)["\']', re.IGNORECASE),
]

# Heavy informal social media slang
HEAVY_SLANG_TOKENS = {
    "chutiyapa", "ghanta", "jhakaas", "dhamaal", "bakwaas", "bakwas",
    "chirkut", "chapri", "bawaal", "bhasad", "locha", "kat gaya",
}


@dataclass(frozen=True)
class ErrorRecord:
    """Individual misclassified sample with deterministic categorization."""
    sample_id: Union[int, str]
    model_name: str
    raw_text: str
    true_label: int
    predicted_label: int
    confidence: float
    noise_score: float
    error_category: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model": self.model_name,
            "text": self.raw_text,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "noise_score": round(self.noise_score, 4),
            "error_category": self.error_category,
            "rationale": self.rationale,
        }


def categorize_error(
    text: str,
    true_label: int,
    predicted_label: int,
    noise_quantifier: Optional[NoiseQuantifier] = None,
) -> tuple[str, str]:
    """Deterministically categorize prediction error without inventing explanations.

    Returns:
        (category, rationale)
    """
    if true_label == predicted_label:
        return "not_an_error", "Prediction is correct."

    nq = noise_quantifier or NoiseQuantifier()
    feats = nq.extract_features(text)
    lower_text = text.lower()
    words = set(re.findall(r'\b\w+\b', lower_text))

    emojis_present = [c for c in text if emoji.is_emoji(c)]
    has_pos_emoji = any(c in POSITIVE_LAUGHING_EMOJIS for c in emojis_present)
    has_neg_emoji = any(c in NEGATIVE_EMOJIS for c in emojis_present)

    # 1. Check severe cross-script code-mixing
    # Presence of Devanagari characters alongside Latin characters or high C-index
    has_devanagari = bool(DEVANAGARI_PATTERN.search(text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    if (has_devanagari and has_latin) or (feats.code_mixing_ratio >= 0.40 and len(words) >= 4):
        return (
            CAT_CROSS_SCRIPT,
            f"Cross-script script-switching detected (Devanagari={has_devanagari}, C={feats.code_mixing_ratio:.2f})."
        )

    # 2. Check dual-sarcasm / polysemous emojis
    # Mismatch between emoji polarity and ground truth
    if true_label == 0 and has_pos_emoji and not has_neg_emoji:
        return (
            CAT_DUAL_SARCASM,
            "Negative sample with positive/laughing emojis creating surface polarity clash."
        )
    if true_label == 1 and has_neg_emoji and not has_pos_emoji:
        return (
            CAT_DUAL_SARCASM,
            "Positive sample with negative/crying emojis creating surface polarity clash."
        )

    # 3. Check structural / implicit irony
    has_contrast = bool(words & CONTRASTIVE_MARKERS)
    has_irony_marker = any(p.search(text) for p in IRONY_MARKERS)
    if has_irony_marker or (has_contrast and len(words) >= 5):
        return (
            CAT_IMPLICIT_IRONY,
            f"Discourse contrast or irony marker detected (contrast_words={list(words & CONTRASTIVE_MARKERS)})."
        )

    # 4. Check slang / syntax drift
    # Heavy character repetition or non-standard slang terms
    has_slang = bool(words & HEAVY_SLANG_TOKENS)
    if feats.repetition_score >= 0.35 or has_slang:
        return (
            CAT_SLANG_DRIFT,
            f"Significant orthographic repetition (R={feats.repetition_score:.2f}) or slang markers ({list(words & HEAVY_SLANG_TOKENS)})."
        )

    # 5. Unavailable — do NOT guess
    return (
        CAT_UNAVAILABLE,
        "No deterministic heuristic satisfied; category marked as unavailable per specification."
    )


class ErrorAnalyzer:
    """Orchestrates deterministic error analysis across model predictions."""

    def __init__(
        self,
        noise_quantifier: Optional[NoiseQuantifier] = None,
        output_dir: Union[str, Path] = "results",
    ) -> None:
        self.nq = noise_quantifier or NoiseQuantifier()
        self.output_dir = Path(output_dir)

    def analyze_errors(
        self,
        model_name: str,
        texts: List[str],
        true_labels: List[int],
        predicted_labels: List[int],
        positive_scores: List[float],
        sample_ids: Optional[List[Union[int, str]]] = None,
    ) -> List[ErrorRecord]:
        """Identify and categorize misclassified instances."""
        ids = sample_ids if sample_ids is not None else list(range(len(texts)))
        errors: List[ErrorRecord] = []

        for i, text in enumerate(texts):
            y_true = int(true_labels[i])
            y_pred = int(predicted_labels[i])
            if y_true == y_pred:
                continue

            score = float(positive_scores[i])
            conf = score if y_pred == 1 else (1.0 - score)
            n_score = self.nq.extract_features(text).noise_score

            category, rationale = categorize_error(
                text=text,
                true_label=y_true,
                predicted_label=y_pred,
                noise_quantifier=self.nq,
            )

            errors.append(
                ErrorRecord(
                    sample_id=ids[i],
                    model_name=model_name,
                    raw_text=text,
                    true_label=y_true,
                    predicted_label=y_pred,
                    confidence=conf,
                    noise_score=n_score,
                    error_category=category,
                    rationale=rationale,
                )
            )

        return errors

    def save_artifacts(
        self,
        errors: List[ErrorRecord],
        total_evaluated_samples: int,
        model_name: str = "Dynamic_Fusion",
    ) -> Dict[str, Path]:
        """Save error analysis artifacts to results/error_analysis/."""
        target_dir = self.output_dir / "error_analysis"
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = model_name.replace(" ", "_").replace("/", "-")
        csv_path = target_dir / f"error_analysis_{safe_name}.csv"
        json_path = target_dir / f"error_summary_{safe_name}.json"

        fieldnames = [
            "sample_id", "model", "true_label", "predicted_label",
            "confidence", "noise_score", "error_category", "rationale", "text",
        ]
        rows = [e.to_dict() for e in errors]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Categorical distribution
        counts = {cat: 0 for cat in ALL_ERROR_CATEGORIES}
        for e in errors:
            counts[e.error_category] = counts.get(e.error_category, 0) + 1

        total_errs = len(errors)
        error_rate = total_errs / total_evaluated_samples if total_evaluated_samples > 0 else 0.0

        summary = {
            "model_name": model_name,
            "total_samples": total_evaluated_samples,
            "total_errors": total_errs,
            "error_rate": round(error_rate, 4),
            "category_distribution": counts,
            "category_percentages": {
                k: round(v / total_errs, 4) if total_errs > 0 else 0.0
                for k, v in counts.items()
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "csv": csv_path,
            "json": json_path,
        }
