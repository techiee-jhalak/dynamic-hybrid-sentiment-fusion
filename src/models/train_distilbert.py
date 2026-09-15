"""DistilBERT Fine-Tuning Pipeline for Binary Sentiment Classification.

Research-grade training pipeline strictly conforming to PROJECT_SPEC.md:
- Epochs: 3
- Batch Size: 16
- Learning Rate: 2e-5 (0.00002)
- Max Sequence Length: <= 128
- Strict data partitioning: Train split only for training, Val split only for model selection.
- Test split is NEVER touched during training.
- Checkpoint saving & best validation model selection (F1 / Accuracy).
- Deterministic random seed and resume-from-checkpoint support.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)
from datasets import Dataset

from configs.config import config, ModelTrainingConfig


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute validation metrics: Accuracy, Macro F1, Binary F1, Precision, and Recall."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    acc = float(accuracy_score(labels, preds))
    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1_binary = float(f1_score(labels, preds, average="binary", zero_division=0))
    prec = float(precision_score(labels, preds, average="binary", zero_division=0))
    rec = float(recall_score(labels, preds, average="binary", zero_division=0))

    return {
        "accuracy": acc,
        "f1": f1_macro,
        "f1_binary": f1_binary,
        "precision": prec,
        "recall": rec,
    }


class DistilBertTrainingPipeline:
    """Manages fine-tuning, evaluation, checkpointing, and saving of DistilBERT sentiment model."""

    def __init__(
        self,
        model_name_or_path: str = config.training.model_name,
        output_dir: Optional[Union[str, Path]] = None,
        training_config: ModelTrainingConfig = config.training,
    ) -> None:
        self.model_name = model_name_or_path
        self.cfg = training_config
        self.output_dir = Path(output_dir) if output_dir else Path(config.paths.saved_models_dir) / "distilbert_best"
        self.checkpoint_dir = Path(config.paths.saved_models_dir) / "checkpoints"

    def prepare_hf_dataset(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        text_col: str = "text",
        label_col: str = "label",
        max_length: int = 128,
    ) -> Dataset:
        """Tokenize a pandas DataFrame and convert to Hugging Face Dataset format."""
        if text_col not in df.columns or label_col not in df.columns:
            raise KeyError(f"Missing required columns '{text_col}' or '{label_col}' in DataFrame.")

        # Ensure valid types
        df_clean = df[[text_col, label_col]].copy()
        df_clean[text_col] = df_clean[text_col].astype(str)
        df_clean[label_col] = df_clean[label_col].astype(int)

        hf_ds = Dataset.from_pandas(df_clean, preserve_index=False)

        def tokenize_function(examples):
            return tokenizer(
                examples[text_col],
                max_length=min(max_length, 128),
                truncation=True,
                padding="max_length",
            )

        tokenized_ds = hf_ds.map(tokenize_function, batched=True)
        # Rename label column if needed to 'label' or 'labels'
        if label_col != "label":
            tokenized_ds = tokenized_ds.rename_column(label_col, "label")

        return tokenized_ds

    def build_training_args(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        seed: int = 42,
    ) -> TrainingArguments:
        """Create exact Hugging Face TrainingArguments based on PROJECT_SPEC.md."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        use_fp16 = torch.cuda.is_available()

        # Handle transformer versions supporting eval_strategy vs evaluation_strategy
        kwargs: Dict[str, Any] = {
            "output_dir": str(self.checkpoint_dir),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "seed": seed,
            "fp16": use_fp16,
            "save_total_limit": 2,
            "logging_steps": 50,
            "report_to": "none",  # Avoid external third-party loggers (e.g. wandb)
        }

        try:
            return TrainingArguments(eval_strategy="epoch", **kwargs)
        except (TypeError, ValueError):
            return TrainingArguments(evaluation_strategy="epoch", **kwargs)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 128,
        seed: int = 42,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Execute full 3-epoch fine-tuning on train split and model selection on val split."""
        set_seed(seed)

        # 1. Initialize Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )

        # 2. Prepare Datasets (Train and Val ONLY)
        train_dataset = self.prepare_hf_dataset(
            train_df,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        val_dataset = self.prepare_hf_dataset(
            val_df,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        # 3. Setup Trainer
        training_args = self.build_training_args(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # 4. Train
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # 5. Final Evaluation on Validation Split
        eval_metrics = trainer.evaluate()

        # 6. Save Best Model and Tokenizer to output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        # 7. Save Training Logs and Validation Metrics
        summary = {
            "model_name": self.model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "seed": seed,
            "train_loss": train_result.training_loss,
            "validation_metrics": eval_metrics,
        }
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def main():
    """CLI entrypoint for running DistilBERT fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment analysis.")
    parser.add_argument("--train_path", type=str, default="data/splits/train.csv", help="Path to train CSV")
    parser.add_argument("--val_path", type=str, default="data/splits/val.csv", help="Path to validation CSV")
    parser.add_argument("--output_dir", type=str, default="saved_models/distilbert_best", help="Save directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default 3)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (default 2e-5)")
    parser.add_argument("--max_length", type=int, default=128, help="Max token sequence length (default 128)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path or True")

    args = parser.parse_args()

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Dataset splits not found at {train_path} or {val_path}. "
            "Please run the dataset pipeline to generate data/splits/ first."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    pipeline = DistilBertTrainingPipeline(output_dir=args.output_dir)
    print(f"Starting DistilBERT fine-tuning: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    summary = pipeline.train(
        train_df=train_df,
        val_df=val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        resume_from_checkpoint=args.resume,
    )
    print("Training finished successfully. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
