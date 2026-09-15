"""Reproducible Evaluation Command.

Runs all registered baseline models against a supplied test CSV and writes
four output artifacts to results/:

  results/metrics.csv
  results/predictions.csv
  results/confusion_matrices/<model>.csv
  results/experiment_summary.json

Usage (from project root):

    python evaluate.py --test_csv data/test.csv \\
                       --text_col text \\
                       --label_col label \\
                       --output_dir results \\
                       --experiment baseline_run_v1

    # Skip expensive BERTweet (disabled by default):
    python evaluate.py --test_csv data/test.csv

    # Evaluate only VADER and LR (comma-separated subset):
    python evaluate.py --test_csv data/test.csv --models vader,lr

Arguments:
    --test_csv      Path to CSV file (MUST be the held-out test split only).
    --text_col      Column name containing raw text. Default: "text".
    --label_col     Column name containing binary labels (0 / 1). Default: "label".
    --output_dir    Directory for output artifacts. Default: "results".
    --experiment    Human-readable experiment name recorded in summary. Default: "evaluation".
    --threshold     Decision threshold for binary classification. Default: 0.5.
    --models        Comma-separated subset of model names to evaluate.
                    Available: vader, lr, static_fusion, dynamic_fusion.
                    Omit to evaluate all non-BERTweet models.
    --bertweet      Enable optional BERTweet baseline (requires model download).
    --lr_checkpoint Path to a pre-fitted LogisticRegression checkpoint (joblib).
                    If omitted, LR is fitted on the test set for demo purposes only.
    --distilbert_checkpoint  Path to fine-tuned DistilBERT checkpoint directory.
                    If omitted, uses the default HuggingFace pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.engine import EvaluationEngine
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel
from src.models.dynamic_fusion import DynamicFusionFramework
from src.models.baselines import (
    LogisticRegressionBaseline,
    StaticFusionBaseline,
    BERTweetBaseline,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproducible sentiment-model evaluation script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--test_csv", required=True, help="Path to test CSV file.")
    p.add_argument("--text_col", default="text", help="Text column name.")
    p.add_argument("--label_col", default="label", help="Label column name.")
    p.add_argument("--output_dir", default="results", help="Output directory.")
    p.add_argument("--experiment", default="evaluation", help="Experiment name.")
    p.add_argument("--threshold", type=float, default=0.50, help="Decision threshold.")
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to evaluate (vader, lr, static_fusion, dynamic_fusion).",
    )
    p.add_argument(
        "--bertweet",
        action="store_true",
        default=False,
        help="Enable optional BERTweet baseline.",
    )
    p.add_argument(
        "--lr_checkpoint",
        default=None,
        help="Path to joblib-serialised LogisticRegressionBaseline checkpoint.",
    )
    p.add_argument(
        "--distilbert_checkpoint",
        default=None,
        help="Path to fine-tuned DistilBERT checkpoint directory.",
    )
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(
    csv_path: str,
    text_col: str,
    label_col: str,
) -> tuple[list[str], list[int], list[int]]:
    """Load test CSV and return (texts, labels, sample_ids).

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if required columns are missing or labels are invalid.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found: {path}")

    df = pd.read_csv(path)

    if text_col not in df.columns:
        raise ValueError(
            f"Text column '{text_col}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_col].fillna("").astype(str).tolist()
    labels_raw = df[label_col].tolist()

    # Validate labels are binary
    valid = {0, 1}
    invalid = set(int(l) for l in labels_raw if int(l) not in valid)
    if invalid:
        raise ValueError(
            f"Label column '{label_col}' contains non-binary values: {invalid}. "
            "Labels must be 0 (Negative) or 1 (Positive)."
        )

    labels = [int(l) for l in labels_raw]
    sample_ids = list(range(len(texts)))

    return texts, labels, sample_ids


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(
    args: argparse.Namespace,
    texts: list[str],
    labels: list[int],
) -> dict:
    """Instantiate and return registered model objects."""
    vader = VaderSentimentModel()

    # DistilBERT — use custom checkpoint if provided
    if args.distilbert_checkpoint:
        distilbert = DistilBertSentimentModel(
            model_name_or_path=args.distilbert_checkpoint
        )
    else:
        distilbert = DistilBertSentimentModel()

    # Logistic Regression — load or fit-on-test (demo only)
    if args.lr_checkpoint:
        try:
            import joblib
            lr = joblib.load(args.lr_checkpoint)
            print(f"[INFO] Loaded LR checkpoint from {args.lr_checkpoint}")
        except Exception as exc:
            print(f"[WARN] Failed to load LR checkpoint: {exc}. Fitting on test set (demo mode).")
            lr = LogisticRegressionBaseline()
            lr.fit(texts, labels)
    else:
        # Fit on test data only for reproducibility demo.
        # In a real experiment this MUST be fitted on the training split.
        print(
            "[WARN] No LR checkpoint provided. "
            "Fitting LogisticRegressionBaseline on test data for demo only. "
            "Accuracy will be optimistic — provide --lr_checkpoint for real evaluation."
        )
        lr = LogisticRegressionBaseline()
        lr.fit(texts, labels)

    static_fusion = StaticFusionBaseline(
        vader_model=vader,
        distilbert_model=distilbert,
        fixed_alpha=0.15,
    )
    dynamic_fusion = DynamicFusionFramework(
        vader_model=vader,
        distilbert_model=distilbert,
    )

    available: dict = {
        "vader": vader,
        "lr": lr,
        "static_fusion": static_fusion,
        "dynamic_fusion": dynamic_fusion,
    }

    if args.bertweet:
        bertweet = BERTweetBaseline()
        bertweet.enable()
        bertweet.load_model()
        available["bertweet"] = bertweet

    return available


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load test data
    print(f"[INFO] Loading test data from: {args.test_csv}")
    texts, labels, sample_ids = load_test_data(
        csv_path=args.test_csv,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    print(f"[INFO] Loaded {len(texts)} samples. "
          f"Positives: {sum(labels)}, Negatives: {len(labels) - sum(labels)}")

    # 2. Build models
    all_models = build_models(args, texts, labels)

    # 3. Filter to requested subset
    if args.models:
        requested = {m.strip().lower() for m in args.models.split(",")}
        unknown = requested - set(all_models)
        if unknown:
            print(f"[WARN] Unknown model names ignored: {unknown}")
        selected = {k: v for k, v in all_models.items() if k in requested}
    else:
        selected = all_models

    if not selected:
        print("[ERROR] No models selected for evaluation. Exiting.")
        sys.exit(1)

    print(f"[INFO] Evaluating models: {list(selected.keys())}")

    # 4. Build engine and register models
    engine = EvaluationEngine(
        output_dir=args.output_dir,
        threshold=args.threshold,
        experiment_name=args.experiment,
    )
    for name, model in selected.items():
        engine.register(name, model)

    # 5. Run evaluation
    print("[INFO] Running evaluation ...")
    results = engine.run(texts=texts, labels=labels, sample_ids=sample_ids)

    # 6. Save all artifacts
    extra_config = {
        "test_csv": args.test_csv,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "threshold": args.threshold,
        "distilbert_checkpoint": args.distilbert_checkpoint,
        "lr_checkpoint": args.lr_checkpoint,
        "bertweet_enabled": args.bertweet,
    }
    paths = engine.save_all(results, extra_config=extra_config)

    # 7. Print summary table
    print("\n========== EVALUATION RESULTS ==========")
    print(f"{'Model':<22} {'Acc':>6} {'Mac-F1':>8} {'Bin-F1':>8} {'AUC':>8} {'Lat(ms)':>10}")
    print("-" * 66)
    for r in results:
        auc_str = f"{r.roc_auc:.4f}" if r.roc_auc is not None else "  N/A  "
        print(
            f"{r.model_name:<22} "
            f"{r.accuracy:>6.4f} "
            f"{r.macro_f1:>8.4f} "
            f"{r.binary_f1:>8.4f} "
            f"{auc_str:>8} "
            f"{r.mean_latency_ms:>10.2f}"
        )
    print("=========================================\n")

    # 8. Report artifact paths
    print("[INFO] Output artifacts:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
