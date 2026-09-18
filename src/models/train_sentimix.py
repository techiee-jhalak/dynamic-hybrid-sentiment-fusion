"""Training pipeline for DistilBERT 3-class sentiment model on SentiMix Hinglish.

Strictly follows research specifications:
- Dataset: SemEval-2020 Task 9 / SentiMix Hindi-English
- Splits: Train (14,000) for training, Dev (3,000) for validation and model selection.
- Test split (3,000) is NEVER touched during training.
- num_labels: 3 (0: positive, 1: negative, 2: neutral)
- Epochs: 3
- Batch Size: 16
- Learning Rate: 2e-5 (0.00002)
- Max Sequence Length: <= 128
- Model architecture: distilbert-base-uncased
- Checkpointing: saves best model based on validation macro-F1 to saved_models/
- Training metadata: saves comprehensive run_metadata.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)
from datasets import Dataset

from configs.config import config
from src.data.sentimix_loader import (
    SentimixDataLoader,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
)


def compute_eval_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute validation metrics for 3-class classification: Macro-F1, Accuracy, etc."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    macro_prec = float(precision_score(labels, preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(labels, preds, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }


class SentimixDistilBertTrainer:
    """Trainer orchestrator for DistilBERT 3-class model on SentiMix."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else Path(config.paths.saved_models_dir) / "sentimix_distilbert_best"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(config.paths.saved_models_dir) / "checkpoints" / "sentimix_distilbert_3class"
        self.loader = SentimixDataLoader()

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> Dataset:
        """Tokenize DataFrame and format for Hugging Face Trainer."""
        df_clean = pd.DataFrame({
            "text": df["text"].astype(str),
            "label": df["label"].astype(int),
        })

        hf_ds = Dataset.from_pandas(df_clean, preserve_index=False)

        def tokenize_batch(examples):
            return tokenizer(
                examples["text"],
                max_length=min(max_length, 128),
                truncation=True,
            )

        return hf_ds.map(tokenize_batch, batched=True, desc="Tokenizing")

    def run_training(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 128,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Execute 3-class training pipeline."""
        start_time = datetime.now(timezone.utc)
        set_seed(seed)

        # 1. Verify and Load Dataset
        train_df = self.loader.load_train()
        dev_df = self.loader.load_dev()

        if len(train_df) != 14000:
            raise ValueError(f"Expected 14,000 train samples, got {len(train_df)}")
        if len(dev_df) != 3000:
            raise ValueError(f"Expected 3,000 dev samples, got {len(dev_df)}")

        # 2. Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label={0: "positive", 1: "negative", 2: "neutral"},
            label2id={"positive": 0, "negative": 1, "neutral": 2},
        )

        # 3. Prepare Tokenized Datasets
        train_dataset = self.prepare_dataset(train_df, tokenizer, max_length=max_length)
        dev_dataset = self.prepare_dataset(dev_df, tokenizer, max_length=max_length)

        # 4. Data Collator (Dynamic padding per batch for optimal CPU performance)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # 5. Training Arguments
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        args_dict = {
            "output_dir": str(self.checkpoint_dir),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size * 2,
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "seed": seed,
            "save_total_limit": 2,
            "logging_steps": 100,
            "report_to": "none",
        }

        try:
            training_args = TrainingArguments(eval_strategy="epoch", **args_dict)
        except (TypeError, ValueError):
            training_args = TrainingArguments(evaluation_strategy="epoch", **args_dict)

        # 6. Trainer
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": dev_dataset,
            "data_collator": data_collator,
            "compute_metrics": compute_eval_metrics,
        }
        try:
            trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
        except TypeError:
            trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

        # 7. Train
        print(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
        train_output = trainer.train()

        # 8. Final Validation Evaluation
        dev_eval = trainer.evaluate()
        print(f"Dev evaluation results: {dev_eval}")

        # 9. Save Best Model and Tokenizer
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        # Also copy to checkpoint dir
        trainer.save_model(str(self.checkpoint_dir))
        tokenizer.save_pretrained(str(self.checkpoint_dir))

        end_time = datetime.now(timezone.utc)
        duration_sec = (end_time - start_time).total_seconds()

        # 10. Training Metadata
        metadata = {
            "dataset_identity": "SemEval-2020 Task 9 / SentiMix — Hinglish",
            "train_count": len(train_df),
            "dev_count": len(dev_df),
            "test_count": 3000,
            "model_architecture": "DistilBertForSequenceClassification",
            "model_name": self.model_name,
            "num_labels": 3,
            "class_order": {"positive": 0, "negative": 1, "neutral": 2},
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "platform": platform.platform(),
            "random_seed": seed,
            "training_start_utc": start_time.isoformat(),
            "training_end_utc": end_time.isoformat(),
            "duration_seconds": round(duration_sec, 2),
            "best_checkpoint_path": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "dev_metrics": {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v
                for k, v in dev_eval.items()
            },
        }

        meta_path = self.checkpoint_dir / "training_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        meta_path_best = self.output_dir / "training_metadata.json"
        with open(meta_path_best, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        return metadata


if __name__ == "__main__":
    trainer = SentimixDistilBertTrainer()
    trainer.run_training()
