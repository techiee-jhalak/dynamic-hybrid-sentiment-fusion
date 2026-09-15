"""Evaluation Engine for Dynamic Hybrid Sentiment Fusion.

Runs every registered model against a test split and writes:
  results/metrics.csv                   -- per-model scalar metrics
  results/predictions.csv               -- per-sample predictions with IDs
  results/confusion_matrices/<model>.csv -- raw confusion matrix
  results/experiment_summary.json        -- full structured experiment record

Rules:
- Metrics are computed from actual predictions only. Nothing is hardcoded.
- Test-set integrity is enforced: the engine never touches train / val splits.
- Sample IDs are preserved throughout.
- Inference latency is measured consistently via time.perf_counter.
- ROC-AUC is computed only when probabilities span both classes.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluation.metrics import ModelEvaluator
from src.models.base import BaseSentimentModel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SamplePrediction:
    """Single-sample inference record."""
    sample_id: Union[int, str]
    model_name: str
    raw_text: str
    true_label: int
    predicted_label: int
    positive_score: float  # continuous probability in [0, 1]


@dataclass
class ModelResult:
    """Aggregated per-model evaluation result."""
    model_name: str
    total_samples: int
    accuracy: float
    macro_f1: float
    binary_f1: float
    precision: float
    recall: float
    roc_auc: Optional[float]
    confusion_matrix: Dict[str, int]          # tn, fp, fn, tp
    mean_latency_ms: float                    # per-sample mean
    p50_latency_ms: float
    p95_latency_ms: float
    per_sample_predictions: List[SamplePrediction] = field(default_factory=list)

    def to_scalar_dict(self) -> Dict[str, Any]:
        """Flat dictionary for CSV row (excludes per-sample data)."""
        cm = self.confusion_matrix
        return {
            "model": self.model_name,
            "samples": self.total_samples,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "binary_f1": self.binary_f1,
            "precision": self.precision,
            "recall": self.recall,
            "roc_auc": self.roc_auc if self.roc_auc is not None else "",
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"],
            "mean_latency_ms": round(self.mean_latency_ms, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 4),
        }


# ---------------------------------------------------------------------------
# Latency helper
# ---------------------------------------------------------------------------

def _measure_latency(
    model: BaseSentimentModel,
    texts: List[str],
) -> Tuple[List[float], List[float]]:
    """
    Run per-sample inference and record wall-clock latency.

    Returns:
        scores    -- list of positive-class probabilities
        latencies -- per-sample latency in milliseconds
    """
    scores: List[float] = []
    latencies: List[float] = []

    for text in texts:
        t0 = time.perf_counter()
        score = model.predict_score(text)
        t1 = time.perf_counter()
        scores.append(float(score))
        latencies.append((t1 - t0) * 1000.0)  # convert to ms

    return scores, latencies


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """
    Orchestrates evaluation of multiple sentiment models against a fixed test set.

    Usage::

        engine = EvaluationEngine(output_dir="results")
        engine.register("VADER", vader_model)
        engine.register("LR", lr_model)
        results = engine.run(
            texts=test_texts,
            labels=test_labels,
            sample_ids=test_ids,          # optional; defaults to 0-indexed
        )
        engine.save_all(results)
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "results",
        threshold: float = 0.50,
        experiment_name: str = "experiment",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.threshold = float(threshold)
        self.experiment_name = experiment_name
        self._models: Dict[str, BaseSentimentModel] = {}
        self._evaluator = ModelEvaluator()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, model: BaseSentimentModel) -> "EvaluationEngine":
        """Register a model for evaluation. Returns self for chaining."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name must be a non-empty string.")
        if not isinstance(model, BaseSentimentModel):
            raise TypeError(f"Model '{name}' must subclass BaseSentimentModel.")
        self._models[name] = model
        return self

    def registered_models(self) -> List[str]:
        """Return list of registered model names."""
        return list(self._models.keys())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        name: str,
        model: BaseSentimentModel,
        texts: List[str],
        labels: List[int],
        sample_ids: Optional[List[Union[int, str]]] = None,
    ) -> ModelResult:
        """Evaluate one model and return a structured ModelResult."""
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must have the same length."
            )
        if not texts:
            raise ValueError("Cannot evaluate on an empty test set.")

        ids = sample_ids if sample_ids is not None else list(range(len(texts)))

        # --- Latency-measured inference (per-sample) -------------------
        scores, latencies = _measure_latency(model, texts)

        # --- Binary predictions from threshold -------------------------
        preds = [1 if s >= self.threshold else 0 for s in scores]

        # --- Metrics ---------------------------------------------------
        metrics = self._evaluator.compute_metrics(
            y_true=labels,
            y_pred=preds,
            y_scores=scores,
        )

        lat_arr = np.array(latencies, dtype=float)
        mean_lat = float(np.mean(lat_arr))
        p50_lat = float(np.percentile(lat_arr, 50))
        p95_lat = float(np.percentile(lat_arr, 95))

        # --- Per-sample records ----------------------------------------
        per_sample = [
            SamplePrediction(
                sample_id=ids[i],
                model_name=name,
                raw_text=texts[i],
                true_label=labels[i],
                predicted_label=preds[i],
                positive_score=scores[i],
            )
            for i in range(len(texts))
        ]

        return ModelResult(
            model_name=name,
            total_samples=int(metrics["total_samples"]),
            accuracy=float(metrics["accuracy"]),
            macro_f1=float(metrics["macro_f1"]),
            binary_f1=float(metrics["binary_f1"]),
            precision=float(metrics["precision"]),
            recall=float(metrics["recall"]),
            roc_auc=metrics.get("roc_auc"),
            confusion_matrix=metrics["confusion_matrix"],
            mean_latency_ms=mean_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            per_sample_predictions=per_sample,
        )

    def run(
        self,
        texts: List[str],
        labels: List[int],
        sample_ids: Optional[List[Union[int, str]]] = None,
    ) -> List[ModelResult]:
        """
        Evaluate all registered models against (texts, labels).

        Args:
            texts:      Test-set input texts (MUST be test split only).
            labels:     Ground-truth binary labels (0 = Negative, 1 = Positive).
            sample_ids: Optional stable identifiers for each sample.

        Returns:
            List of ModelResult objects, one per registered model.
        """
        if not self._models:
            raise RuntimeError(
                "No models registered. Call register(name, model) before run()."
            )

        results: List[ModelResult] = []
        for name, model in self._models.items():
            result = self.evaluate_single(
                name=name,
                model=model,
                texts=texts,
                labels=labels,
                sample_ids=sample_ids,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_metrics_csv(self, results: List[ModelResult]) -> Path:
        """Write results/metrics.csv — one row per model."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "metrics.csv"

        rows = [r.to_scalar_dict() for r in results]
        if not rows:
            return out_path

        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return out_path

    def save_predictions_csv(self, results: List[ModelResult]) -> Path:
        """Write results/predictions.csv — one row per (model, sample) pair."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "predictions.csv"

        rows: List[Dict[str, Any]] = []
        for result in results:
            for sp in result.per_sample_predictions:
                rows.append({
                    "model": sp.model_name,
                    "sample_id": sp.sample_id,
                    "true_label": sp.true_label,
                    "predicted_label": sp.predicted_label,
                    "positive_score": round(sp.positive_score, 6),
                    "text": sp.raw_text,
                })

        if not rows:
            return out_path

        fieldnames = ["model", "sample_id", "true_label", "predicted_label", "positive_score", "text"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return out_path

    def save_confusion_matrices(self, results: List[ModelResult]) -> Path:
        """Write results/confusion_matrices/<model_name>.csv — one file per model."""
        cm_dir = self.output_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            safe_name = result.model_name.replace(" ", "_").replace("/", "-")
            cm_path = cm_dir / f"{safe_name}.csv"
            cm = result.confusion_matrix

            # Write as 2×2 labelled matrix
            with open(cm_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["", "Predicted_Negative", "Predicted_Positive"])
                writer.writerow(["Actual_Negative", cm["tn"], cm["fp"]])
                writer.writerow(["Actual_Positive", cm["fn"], cm["tp"]])

        return cm_dir

    def save_experiment_summary(
        self,
        results: List[ModelResult],
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write results/experiment_summary.json with full structured record."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "experiment_summary.json"

        summary: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "threshold": self.threshold,
            "num_models": len(results),
            "config": extra_config or {},
            "models": [],
        }

        for result in results:
            entry: Dict[str, Any] = {
                "model_name": result.model_name,
                "total_samples": result.total_samples,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "binary_f1": result.binary_f1,
                "precision": result.precision,
                "recall": result.recall,
                "roc_auc": result.roc_auc,
                "confusion_matrix": result.confusion_matrix,
                "latency": {
                    "mean_ms": round(result.mean_latency_ms, 4),
                    "p50_ms": round(result.p50_latency_ms, 4),
                    "p95_ms": round(result.p95_latency_ms, 4),
                },
            }
            summary["models"].append(entry)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        return out_path

    def save_all(
        self,
        results: List[ModelResult],
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Write all four output artifacts.

        Returns:
            dict with keys: metrics_csv, predictions_csv, confusion_dir, summary_json
        """
        return {
            "metrics_csv": self.save_metrics_csv(results),
            "predictions_csv": self.save_predictions_csv(results),
            "confusion_dir": self.save_confusion_matrices(results),
            "summary_json": self.save_experiment_summary(results, extra_config),
        }
