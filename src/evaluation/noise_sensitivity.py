"""Noise-Sensitivity Analysis Module for Dynamic Hybrid Sentiment Fusion.

Partitions test data into four research-defined noise groups based on composite noise score N:
- LOW:       0.0 <= N <= 0.2
- MODERATE:  0.2 <  N <= 0.5
- HIGH:      0.5 <  N <= 0.8
- EXTREME:   0.8 <  N <= 1.0

Computes performance metrics (Accuracy, F1, Precision, Recall) per noise band
for each evaluated model when sufficient samples exist.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.features.noise_quantifier import (
    NoiseQuantifier,
    assign_noise_group,
    NOISE_GROUP_LOW,
    NOISE_GROUP_MODERATE,
    NOISE_GROUP_HIGH,
    NOISE_GROUP_EXTREME,
    ALL_NOISE_GROUPS,
)
from src.evaluation.metrics import ModelEvaluator
from src.models.base import BaseSentimentModel


@dataclass(frozen=True)
class NoiseGroupMetrics:
    """Metrics for a specific model evaluated on a specific noise group."""
    model_name: str
    noise_group: str
    sample_count: int
    accuracy: Optional[float]
    macro_f1: Optional[float]
    binary_f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    status: str  # "evaluated" or "insufficient_samples"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "noise_group": self.noise_group,
            "samples": self.sample_count,
            "accuracy": self.accuracy if self.accuracy is not None else "",
            "macro_f1": self.macro_f1 if self.macro_f1 is not None else "",
            "binary_f1": self.binary_f1 if self.binary_f1 is not None else "",
            "precision": self.precision if self.precision is not None else "",
            "recall": self.recall if self.recall is not None else "",
            "status": self.status,
        }


class NoiseSensitivityAnalyzer:
    """Evaluates sentiment models across discrete noise sensitivity tiers.

    Reuses NoiseQuantifier and ModelEvaluator without hardcoding any metrics.
    """

    def __init__(
        self,
        noise_quantifier: Optional[NoiseQuantifier] = None,
        min_samples_per_group: int = 1,
        output_dir: Union[str, Path] = "results",
    ) -> None:
        self.quantifier = noise_quantifier or NoiseQuantifier()
        self.min_samples = min_samples_per_group
        self.output_dir = Path(output_dir)
        self._evaluator = ModelEvaluator()

    def partition_samples(
        self,
        texts: List[str],
        labels: List[int],
        sample_ids: Optional[List[Union[int, str]]] = None,
    ) -> Dict[str, Dict[str, List[Any]]]:
        """Compute noise score N for all texts and partition them into noise groups."""
        ids = sample_ids if sample_ids is not None else list(range(len(texts)))

        partitions: Dict[str, Dict[str, List[Any]]] = {
            group: {"texts": [], "labels": [], "ids": [], "noise_scores": []}
            for group in ALL_NOISE_GROUPS
        }

        for i, text in enumerate(texts):
            features = self.quantifier.extract_features(text)
            n_score = features.noise_score
            group = assign_noise_group(n_score)

            partitions[group]["texts"].append(text)
            partitions[group]["labels"].append(labels[i])
            partitions[group]["ids"].append(ids[i])
            partitions[group]["noise_scores"].append(n_score)

        return partitions

    def evaluate_model_on_groups(
        self,
        model_name: str,
        model: BaseSentimentModel,
        partitions: Dict[str, Dict[str, List[Any]]],
        threshold: float = 0.50,
    ) -> List[NoiseGroupMetrics]:
        """Evaluate a single model across all partitioned noise groups."""
        group_results: List[NoiseGroupMetrics] = []

        for group in ALL_NOISE_GROUPS:
            group_data = partitions[group]
            grp_texts = group_data["texts"]
            grp_labels = group_data["labels"]
            sample_count = len(grp_texts)

            if sample_count < self.min_samples:
                group_results.append(
                    NoiseGroupMetrics(
                        model_name=model_name,
                        noise_group=group,
                        sample_count=sample_count,
                        accuracy=None,
                        macro_f1=None,
                        binary_f1=None,
                        precision=None,
                        recall=None,
                        status="insufficient_samples",
                    )
                )
                continue

            scores = model.predict_scores_batch(grp_texts)
            preds = [1 if s >= threshold else 0 for s in scores]

            metrics = self._evaluator.compute_metrics(
                y_true=grp_labels,
                y_pred=preds,
                y_scores=scores,
            )

            group_results.append(
                NoiseGroupMetrics(
                    model_name=model_name,
                    noise_group=group,
                    sample_count=sample_count,
                    accuracy=metrics["accuracy"],
                    macro_f1=metrics["macro_f1"],
                    binary_f1=metrics["binary_f1"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    status="evaluated",
                )
            )

        return group_results

    def analyze(
        self,
        models: Dict[str, BaseSentimentModel],
        texts: List[str],
        labels: List[int],
        sample_ids: Optional[List[Union[int, str]]] = None,
        threshold: float = 0.50,
    ) -> List[NoiseGroupMetrics]:
        """Run complete noise sensitivity evaluation across all models and noise bands."""
        if not texts or not labels:
            raise ValueError("Input texts and labels cannot be empty for noise sensitivity analysis.")

        partitions = self.partition_samples(texts, labels, sample_ids)

        all_metrics: List[NoiseGroupMetrics] = []
        for model_name, model in models.items():
            metrics = self.evaluate_model_on_groups(
                model_name=model_name,
                model=model,
                partitions=partitions,
                threshold=threshold,
            )
            all_metrics.extend(metrics)

        return all_metrics

    def save_artifacts(
        self,
        results: List[NoiseGroupMetrics],
        partitions: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    ) -> Dict[str, Path]:
        """Save noise sensitivity CSV and JSON summary to results/noise_sensitivity/."""
        target_dir = self.output_dir / "noise_sensitivity"
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_path = target_dir / "noise_sensitivity.csv"
        json_path = target_dir / "noise_sensitivity_summary.json"

        fieldnames = [
            "model", "noise_group", "samples", "accuracy",
            "macro_f1", "binary_f1", "precision", "recall", "status",
        ]
        rows = [r.to_dict() for r in results]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Summary JSON
        group_counts = {
            grp: len(partitions[grp]["texts"]) if partitions and grp in partitions else sum(
                r.sample_count for r in results if r.noise_group == grp and r.model_name == (results[0].model_name if results else "")
            )
            for grp in ALL_NOISE_GROUPS
        }

        summary = {
            "noise_group_definitions": {
                NOISE_GROUP_LOW: "0.0 <= N <= 0.2",
                NOISE_GROUP_MODERATE: "0.2 < N <= 0.5",
                NOISE_GROUP_HIGH: "0.5 < N <= 0.8",
                NOISE_GROUP_EXTREME: "0.8 < N <= 1.0",
            },
            "sample_counts": group_counts,
            "results": rows,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "csv": csv_path,
            "json": json_path,
        }
