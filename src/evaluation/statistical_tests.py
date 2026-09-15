"""Statistical Hypothesis Testing for Sentiment Model Comparison.

Implements McNemar's Test with continuity correction for paired nominal predictions:
- Compares Dynamic Fusion against DistilBERT and Static Fusion on exact same samples.
- Pre-set significance level: alpha = 0.05.
- Computes 2x2 contingency matrix (both correct, discordant A/B, both incorrect).
- Provides chi-square test statistic, exact p-value, and significance determination.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class McNemarResult:
    """Structured container for McNemar test comparison between two models."""
    model_a: str
    model_b: str
    sample_count: int
    both_correct: int       # a: M_A correct, M_B correct
    model_a_only: int       # b: M_A correct, M_B incorrect
    model_b_only: int       # c: M_A incorrect, M_B correct
    both_incorrect: int     # d: M_A incorrect, M_B incorrect
    statistic: float        # Chi-square statistic with continuity correction
    p_value: float          # p-value (df=1)
    alpha: float            # Significance threshold (default 0.05)
    is_significant: bool    # True if p_value < alpha
    superior_model: str     # Name of better model or "tied"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "sample_count": self.sample_count,
            "contingency_table": {
                "both_correct": self.both_correct,
                "model_a_only": self.model_a_only,
                "model_b_only": self.model_b_only,
                "both_incorrect": self.both_incorrect,
            },
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "is_significant": self.is_significant,
            "superior_model": self.superior_model,
        }

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "samples": self.sample_count,
            "both_correct": self.both_correct,
            "model_a_only": self.model_a_only,
            "model_b_only": self.model_b_only,
            "both_incorrect": self.both_incorrect,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "significant_at_005": self.is_significant,
            "superior_model": self.superior_model,
        }


def mcnemar_test(
    y_true: Union[List[int], np.ndarray],
    y_pred_a: Union[List[int], np.ndarray],
    y_pred_b: Union[List[int], np.ndarray],
    model_a_name: str = "Model_A",
    model_b_name: str = "Model_B",
    alpha: float = 0.05,
    continuity_correction: bool = True,
) -> McNemarResult:
    """Compute McNemar's test for paired classification outputs on the exact same test instances."""
    t = np.array(y_true, dtype=int)
    pa = np.array(y_pred_a, dtype=int)
    pb = np.array(y_pred_b, dtype=int)

    if not (len(t) == len(pa) == len(pb)):
        raise ValueError("y_true, y_pred_a, and y_pred_b must have identical lengths.")
    if len(t) == 0:
        raise ValueError("Cannot perform statistical test on empty prediction arrays.")

    correct_a = (pa == t)
    correct_b = (pb == t)

    both_correct = int(np.sum(correct_a & correct_b))
    model_a_only = int(np.sum(correct_a & ~correct_b))  # b
    model_b_only = int(np.sum(~correct_a & correct_b))  # c
    both_incorrect = int(np.sum(~correct_a & ~correct_b))

    b = model_a_only
    c = model_b_only
    discordant = b + c

    if discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        if continuity_correction:
            # Edwards continuity correction: (|b - c| - 1)^2 / (b + c)
            numerator = max(0.0, abs(b - c) - 1.0) ** 2
            stat = float(numerator / discordant)
        else:
            stat = float(((b - c) ** 2) / discordant)

        # 1 degree of freedom
        p_val = float(stats.chi2.sf(stat, df=1))

    if b > c:
        superior = model_a_name
    elif c > b:
        superior = model_b_name
    else:
        superior = "tied"

    return McNemarResult(
        model_a=model_a_name,
        model_b=model_b_name,
        sample_count=len(t),
        both_correct=both_correct,
        model_a_only=model_a_only,
        model_b_only=model_b_only,
        both_incorrect=both_incorrect,
        statistic=stat,
        p_value=p_val,
        alpha=alpha,
        is_significant=bool(p_val < alpha),
        superior_model=superior,
    )


class StatisticalComparator:
    """Orchestrates statistical significance testing against reference model."""

    def __init__(self, alpha: float = 0.05, output_dir: Union[str, Path] = "results") -> None:
        self.alpha = float(alpha)
        self.output_dir = Path(output_dir)

    def compare_models(
        self,
        y_true: List[int],
        reference_preds: List[int],
        comparator_preds_map: Dict[str, List[int]],
        reference_name: str = "Dynamic_Fusion",
    ) -> List[McNemarResult]:
        """Perform McNemar comparison of reference model against multiple alternatives."""
        results: List[McNemarResult] = []
        for comp_name, comp_preds in comparator_preds_map.items():
            res = mcnemar_test(
                y_true=y_true,
                y_pred_a=reference_preds,
                y_pred_b=comp_preds,
                model_a_name=reference_name,
                model_b_name=comp_name,
                alpha=self.alpha,
            )
            results.append(res)
        return results

    def save_artifacts(self, results: List[McNemarResult]) -> Dict[str, Path]:
        """Save statistical test outputs to results/statistical_tests/."""
        target_dir = self.output_dir / "statistical_tests"
        target_dir.mkdir(parents=True, exist_ok=True)

        json_path = target_dir / "mcnemar_tests.json"
        csv_path = target_dir / "mcnemar_tests.csv"

        fieldnames = [
            "model_a", "model_b", "samples", "both_correct",
            "model_a_only", "model_b_only", "both_incorrect",
            "statistic", "p_value", "alpha", "significant_at_005", "superior_model",
        ]
        rows = [r.to_csv_row() for r in results]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "significance_threshold_alpha": self.alpha,
            "test_type": "McNemar's test with Edwards' continuity correction (df=1)",
            "comparisons": [r.to_dict() for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "json": json_path,
            "csv": csv_path,
        }
