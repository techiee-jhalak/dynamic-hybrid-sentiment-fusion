"""Evaluation and comparative benchmark reporting metrics module.

Computes empirical research metrics:
- Accuracy
- Macro F1 / Binary F1
- Precision / Recall
- Confusion Matrix (TN, FP, FN, TP)
- ROC AUC (when probabilities provided)

Zero hardcoding of reference literature metrics.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from src.models.base import BaseSentimentModel


class ModelEvaluator:
    """Computes empirical evaluation metrics from model predictions."""

    @staticmethod
    def compute_metrics(
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        y_scores: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive empirical performance metrics."""
        y_t = np.array(y_true, dtype=int)
        y_p = np.array(y_pred, dtype=int)

        if len(y_t) == 0:
            raise ValueError("Evaluation label arrays cannot be empty.")

        acc = float(accuracy_score(y_t, y_p))
        macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        binary_f1 = float(f1_score(y_t, y_p, average="binary", zero_division=0))
        prec = float(precision_score(y_t, y_p, average="binary", zero_division=0))
        rec = float(recall_score(y_t, y_p, average="binary", zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        metrics: Dict[str, Any] = {
            "total_samples": int(len(y_t)),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "binary_f1": round(binary_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
        }

        # ROC AUC
        if y_scores is not None and len(np.unique(y_t)) > 1:
            try:
                auc = float(roc_auc_score(y_t, np.array(y_scores, dtype=float)))
                metrics["roc_auc"] = round(auc, 4)
            except Exception:
                metrics["roc_auc"] = None

        return metrics

    def evaluate_model(
        self,
        model: BaseSentimentModel,
        texts: List[str],
        y_true: List[int],
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Evaluate a single sentiment model on empirical test data."""
        y_scores = model.predict_scores_batch(texts)
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        return self.compute_metrics(y_true=y_true, y_pred=y_pred, y_scores=y_scores)

    def compare_baselines(
        self,
        models: Dict[str, BaseSentimentModel],
        texts: List[str],
        y_true: List[int],
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """Evaluate and tabulate comparative performance metrics across all models."""
        records = []
        for name, model in models.items():
            res = self.evaluate_model(model, texts, y_true, threshold=threshold)
            records.append({
                "Model": name,
                "Accuracy": res["accuracy"],
                "Macro F1": res["macro_f1"],
                "Binary F1": res["binary_f1"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "ROC AUC": res.get("roc_auc", "-"),
            })

        return pd.DataFrame(records)
