"""Reproducible Ablation Study Command.

Evaluates four architectural variants of the Dynamic Hybrid Sentiment Fusion
system against a test CSV and writes results/ablation_results.csv.

Variants evaluated
------------------
A  Full Dynamic Fusion (proposed system — reference)
B  Static Fusion / no dynamic routing (constant alpha = 0.02)
C  Dynamic routing without length contribution (w1 = 0)
D  Dynamic routing without the N <= 0.20 gated threshold

Usage (from project root)::

    python ablation.py --test_csv data/test.csv

All flags::

    python ablation.py \\
        --test_csv data/test.csv \\
        --text_col text \\
        --label_col label \\
        --output_dir results \\
        --threshold 0.5 \\
        --distilbert_checkpoint saved_models/distilbert

Arguments
---------
--test_csv               Path to held-out test CSV (MUST be test split only).
--text_col               Text column name (default: text).
--label_col              Label column name (default: label).
--output_dir             Output directory (default: results).
--threshold              Decision threshold (default: 0.5).
--distilbert_checkpoint  Path to fine-tuned DistilBERT directory (optional).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ablation import (
    AblationRunner,
    ALL_VARIANTS,
    VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D,
)
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproducible ablation study for Dynamic Hybrid Sentiment Fusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--test_csv", required=True, help="Path to test CSV.")
    p.add_argument("--text_col", default="text", help="Text column name.")
    p.add_argument("--label_col", default="label", help="Label column name.")
    p.add_argument("--output_dir", default="results", help="Output directory.")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument(
        "--distilbert_checkpoint",
        default=None,
        help="Path to fine-tuned DistilBERT directory.",
    )
    return p


def load_test_data(
    csv_path: str,
    text_col: str,
    label_col: str,
) -> tuple[list[str], list[int], list[int]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found: {path}")

    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in {path}. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not in {path}. Available: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = [int(v) for v in df[label_col].tolist()]
    sample_ids = list(range(len(texts)))
    return texts, labels, sample_ids


def main() -> None:
    args = build_parser().parse_args()

    print(f"[INFO] Loading test data from: {args.test_csv}")
    texts, labels, sample_ids = load_test_data(
        csv_path=args.test_csv,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    print(f"[INFO] {len(texts)} samples loaded. "
          f"Positives: {sum(labels)}, Negatives: {len(labels) - sum(labels)}")

    # Build shared model instances (loaded once for all variants)
    print("[INFO] Loading models ...")
    vader = VaderSentimentModel()
    if args.distilbert_checkpoint:
        distilbert = DistilBertSentimentModel(model_name_or_path=args.distilbert_checkpoint)
    else:
        distilbert = DistilBertSentimentModel()

    runner = AblationRunner(
        vader_model=vader,
        distilbert_model=distilbert,
        output_dir=args.output_dir,
        threshold=args.threshold,
        variants=ALL_VARIANTS,
    )

    print("[INFO] Running ablation study ...")
    results = runner.run(texts, labels, sample_ids)
    csv_path = runner.save_csv(results)

    # Print summary table
    ref_f1 = results[0].binary_f1
    print("\n========== ABLATION RESULTS ==========")
    print(f"{'Variant':<35} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ΔF1':>9}")
    print("-" * 75)
    for r in results:
        delta_str = "  (ref)" if r.f1_delta is None else f"{r.f1_delta:+.4f}"
        print(
            f"{r.variant_name:<35} "
            f"{r.accuracy:>6.4f} "
            f"{r.precision:>7.4f} "
            f"{r.recall:>7.4f} "
            f"{r.binary_f1:>7.4f} "
            f"{delta_str:>9}"
        )
    print("======================================\n")
    print(f"[INFO] Results written to: {csv_path}")


if __name__ == "__main__":
    main()
