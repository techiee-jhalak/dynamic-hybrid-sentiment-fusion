"""Dataset pipeline module for dynamic noise-aware sentiment fusion.

Handles:
- Loading CSV datasets from data/raw/ or custom paths
- Configurable text and label column mapping
- Label normalization (negative -> 0, positive -> 1, handling multiple formats)
- Missing value validation (dropping/flagging invalid records)
- Exact duplicate removal
- Raw text preservation
- Deterministic stratified splitting (70% train, 10% val, 20% test)
- Configurable random seed for full reproducibility
- Exporting splits to data/splits/
- Generating comprehensive dataset statistics
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import config


LABEL_MAPPING = {
    # Negative representations -> 0
    "0": 0,
    0: 0,
    0.0: 0,
    "negative": 0,
    "neg": 0,
    "n": 0,
    False: 0,
    "false": 0,
    # Positive representations -> 1
    "1": 1,
    1: 1,
    1.0: 1,
    "positive": 1,
    "pos": 1,
    "p": 1,
    True: 1,
    "true": 1,
}


class DatasetPipeline:
    """End-to-end dataset ingestion, validation, normalization, and splitting pipeline."""

    def __init__(
        self,
        raw_data_dir: Optional[Union[str, Path]] = None,
        splits_dir: Optional[Union[str, Path]] = None,
        text_column: str = "text",
        label_column: str = "label",
        random_seed: int = 42,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else Path(config.paths.data_dir) / "raw"
        self.splits_dir = Path(splits_dir) if splits_dir else Path(config.paths.data_dir) / "splits"
        self.text_column = text_column
        self.label_column = label_column
        self.random_seed = random_seed

    @staticmethod
    def normalize_label(val: Any) -> int:
        """Normalize various label formats to integer 0 (Negative) or 1 (Positive).

        Raises ValueError for unrecognized or ambiguous labels.
        """
        if pd.isna(val):
            raise ValueError("Label is NaN or missing.")

        if isinstance(val, (int, np.integer)):
            if val in (0, 1):
                return int(val)
            raise ValueError(f"Numeric label {val} is not 0 or 1.")

        if isinstance(val, (float, np.floating)):
            if int(val) == val and int(val) in (0, 1):
                return int(val)
            raise ValueError(f"Floating label {val} cannot be cast to binary 0 or 1.")

        if isinstance(val, bool):
            return 1 if val else 0

        str_val = str(val).strip().lower()
        if str_val in LABEL_MAPPING:
            return LABEL_MAPPING[str_val]

        raise ValueError(f"Unrecognized sentiment label value: {val!r}")

    def clean_and_validate(
        self,
        df: pd.DataFrame,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Validate, normalize labels, clean missing records, and optionally remove duplicates.

        Preserves original raw text.
        """
        text_c = text_col or self.text_column
        label_c = label_col or self.label_column

        if text_c not in df.columns:
            raise KeyError(f"Text column '{text_c}' not found in DataFrame columns: {list(df.columns)}")
        if label_c not in df.columns:
            raise KeyError(f"Label column '{label_c}' not found in DataFrame columns: {list(df.columns)}")

        # Copy to avoid side effects
        cleaned_df = df.copy()

        # 1. Validate missing text & label values
        # Drop rows where text or label is null
        cleaned_df = cleaned_df.dropna(subset=[text_c, label_c])

        # Filter out empty string or whitespace-only text
        cleaned_df[text_c] = cleaned_df[text_c].astype(str)
        cleaned_df = cleaned_df[cleaned_df[text_c].str.strip() != ""]

        # 2. Normalize labels
        def _safe_normalize(x):
            try:
                return self.normalize_label(x)
            except ValueError:
                return np.nan

        cleaned_df[label_c] = cleaned_df[label_c].apply(_safe_normalize)
        cleaned_df = cleaned_df.dropna(subset=[label_c])
        cleaned_df[label_c] = cleaned_df[label_c].astype(int)

        # 3. Remove exact duplicate samples based on text and label
        if drop_duplicates:
            cleaned_df = cleaned_df.drop_duplicates(subset=[text_c, label_c])

        # Standardize column names if needed or keep both text and raw_text
        cleaned_df = cleaned_df.reset_index(drop=True)
        return cleaned_df

    def load_csv(
        self,
        filepath_or_name: Union[str, Path],
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load and clean a CSV file from data/raw/ or a direct filepath."""
        text_c = text_col or self.text_column
        label_c = label_col or self.label_column

        file_path = Path(filepath_or_name)
        if not file_path.is_absolute() and not file_path.exists():
            file_path = self.raw_data_dir / filepath_or_name

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset CSV file not found at: {file_path}")

        df = pd.read_csv(file_path, **kwargs)
        return self.clean_and_validate(df, text_col=text_c, label_col=label_c)

    def stratified_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
        label_col: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Deterministically split dataset into stratified train, val, and test partitions."""
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0. Got: {total_ratio}")

        label_c = label_col or self.label_column
        seed = random_seed if random_seed is not None else self.random_seed
        n_total = len(df)

        if n_total == 0:
            raise ValueError("Cannot split an empty DataFrame.")

        # Calculate exact target integer counts
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = n_total - n_test - n_val

        # Stage 1: Split off the test set with exact integer count
        train_val_df, test_df = train_test_split(
            df,
            test_size=n_test,
            stratify=df[label_c],
            random_state=seed,
            shuffle=True,
        )

        # Stage 2: Split train_val into train and validation with exact integer count
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=n_val,
            stratify=train_val_df[label_c],
            random_state=seed,
            shuffle=True,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, val_df, test_df


    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        splits_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Save train, validation, and test splits into CSV files in splits_dir."""
        target_dir = Path(splits_dir) if splits_dir else self.splits_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        train_path = target_dir / "train.csv"
        val_path = target_dir / "val.csv"
        test_path = target_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        return {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }

    @staticmethod
    def generate_dataset_statistics(
        df_or_splits: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        text_col: str = "text",
        label_col: str = "label",
    ) -> Dict[str, Any]:
        """Generate comprehensive statistics for a single DataFrame or split dictionary."""
        if isinstance(df_or_splits, pd.DataFrame):
            splits_map = {"dataset": df_or_splits}
        else:
            splits_map = df_or_splits

        stats = {}
        for split_name, split_df in splits_map.items():
            if split_df.empty:
                stats[split_name] = {"total_samples": 0}
                continue

            total_samples = len(split_df)
            label_counts = split_df[label_col].value_counts().to_dict()
            num_pos = int(label_counts.get(1, 0))
            num_neg = int(label_counts.get(0, 0))

            char_lengths = split_df[text_col].astype(str).str.len()
            word_counts = split_df[text_col].astype(str).str.split().str.len()

            stats[split_name] = {
                "total_samples": total_samples,
                "positive_samples": num_pos,
                "negative_samples": num_neg,
                "positive_ratio": round(num_pos / total_samples, 4) if total_samples > 0 else 0.0,
                "negative_ratio": round(num_neg / total_samples, 4) if total_samples > 0 else 0.0,
                "char_length": {
                    "mean": round(float(char_lengths.mean()), 2),
                    "std": round(float(char_lengths.std()), 2) if total_samples > 1 else 0.0,
                    "min": int(char_lengths.min()),
                    "max": int(char_lengths.max()),
                },
                "word_count": {
                    "mean": round(float(word_counts.mean()), 2),
                    "std": round(float(word_counts.std()), 2) if total_samples > 1 else 0.0,
                    "min": int(word_counts.min()),
                    "max": int(word_counts.max()),
                },
            }

        return stats
