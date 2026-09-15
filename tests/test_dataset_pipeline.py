import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np

from src.data.dataset import DatasetPipeline, LABEL_MAPPING



class TestDatasetPipeline(unittest.TestCase):
    """Comprehensive test suite for DatasetPipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.splits_dir = Path(self.temp_dir) / "splits"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = DatasetPipeline(
            raw_data_dir=self.raw_dir,
            splits_dir=self.splits_dir,
            text_column="text",
            label_column="label",
            random_seed=42,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_label_conversion(self):
        """Test normalization of multiple label representations to 0 and 1."""
        # Positive variations
        self.assertEqual(DatasetPipeline.normalize_label(1), 1)
        self.assertEqual(DatasetPipeline.normalize_label(1.0), 1)
        self.assertEqual(DatasetPipeline.normalize_label("1"), 1)
        self.assertEqual(DatasetPipeline.normalize_label("positive"), 1)
        self.assertEqual(DatasetPipeline.normalize_label("Positive"), 1)
        self.assertEqual(DatasetPipeline.normalize_label(" POSITIVE "), 1)
        self.assertEqual(DatasetPipeline.normalize_label("pos"), 1)
        self.assertEqual(DatasetPipeline.normalize_label("p"), 1)
        self.assertEqual(DatasetPipeline.normalize_label(True), 1)
        self.assertEqual(DatasetPipeline.normalize_label("true"), 1)

        # Negative variations
        self.assertEqual(DatasetPipeline.normalize_label(0), 0)
        self.assertEqual(DatasetPipeline.normalize_label(0.0), 0)
        self.assertEqual(DatasetPipeline.normalize_label("0"), 0)
        self.assertEqual(DatasetPipeline.normalize_label("negative"), 0)
        self.assertEqual(DatasetPipeline.normalize_label("Negative"), 0)
        self.assertEqual(DatasetPipeline.normalize_label(" NEGATIVE "), 0)
        self.assertEqual(DatasetPipeline.normalize_label("neg"), 0)
        self.assertEqual(DatasetPipeline.normalize_label("n"), 0)
        self.assertEqual(DatasetPipeline.normalize_label(False), 0)
        self.assertEqual(DatasetPipeline.normalize_label("false"), 0)

        # Invalid labels
        with self.assertRaises(ValueError):
            DatasetPipeline.normalize_label(None)
        with self.assertRaises(ValueError):
            DatasetPipeline.normalize_label(np.nan)
        with self.assertRaises(ValueError):
            DatasetPipeline.normalize_label("neutral")
        with self.assertRaises(ValueError):
            DatasetPipeline.normalize_label(2)
        with self.assertRaises(ValueError):
            DatasetPipeline.normalize_label("unknown")

    def test_duplicate_handling(self):
        """Test removal of exact duplicate samples."""
        df = pd.DataFrame({
            "text": [
                "ye movie bahut achhi thi!! 😍",
                "ye movie bahut achhi thi!! 😍",  # duplicate
                "worst experience ever 😡",
                "worst experience ever 😡",       # duplicate
                "different sentence",
            ],
            "label": ["positive", "positive", "negative", "negative", "positive"],
        })

        cleaned = self.pipeline.clean_and_validate(df)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned["label"].tolist(), [1, 0, 1])

    def test_missing_values(self):
        """Test handling and dropping of missing/empty texts and labels."""
        df = pd.DataFrame({
            "text": [
                "valid positive text",
                None,                          # null text
                "",                            # empty text
                "   ",                         # whitespace text
                "valid negative text",
                "text with missing label",
                "text with invalid label",
            ],
            "label": [
                "positive",
                "positive",
                "positive",
                "negative",
                "negative",
                None,                          # missing label
                "invalid_category",            # unparseable label
            ],
        })

        cleaned = self.pipeline.clean_and_validate(df)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["text"].tolist(), ["valid positive text", "valid negative text"])
        self.assertEqual(cleaned["label"].tolist(), [1, 0])

    def test_raw_text_preservation(self):
        """Test that emojis, Hinglish words, punctuation, and casing are preserved exactly."""
        raw_samples = [
            "Kya baat hai brooo!!! 🔥🔥 Full paisa vasool!",
            "Bhai bilkul bakwaas movie thi... time waste :((",
            "100% recommended!! Sach me maza aa gaya ❤️❤️",
        ]
        df = pd.DataFrame({
            "text": raw_samples,
            "label": [1, 0, 1],
        })

        cleaned = self.pipeline.clean_and_validate(df)
        self.assertEqual(cleaned["text"].tolist(), raw_samples)

    def test_stratification(self):
        """Test that train (70%), val (10%), and test (20%) maintain identical class distributions."""
        # Create a synthetic dataset of 100 samples with 60% positive and 40% negative
        n_pos = 60
        n_neg = 40
        texts = [f"pos_text_{i}" for i in range(n_pos)] + [f"neg_text_{i}" for i in range(n_neg)]
        labels = [1] * n_pos + [0] * n_neg
        df = pd.DataFrame({"text": texts, "label": labels})

        train_df, val_df, test_df = self.pipeline.stratified_split(
            df,
            train_ratio=0.70,
            val_ratio=0.10,
            test_ratio=0.20,
            random_seed=42,
        )

        # Check total lengths
        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(val_df), 10)
        self.assertEqual(len(test_df), 20)

        # Check class ratios across splits
        # 60% pos -> train should have ~42 pos (70 * 0.6 = 42), val ~6 pos, test ~12 pos
        self.assertEqual((train_df["label"] == 1).sum(), 42)
        self.assertEqual((train_df["label"] == 0).sum(), 28)

        self.assertEqual((val_df["label"] == 1).sum(), 6)
        self.assertEqual((val_df["label"] == 0).sum(), 4)

        self.assertEqual((test_df["label"] == 1).sum(), 12)
        self.assertEqual((test_df["label"] == 0).sum(), 8)

    def test_reproducibility(self):
        """Test that identical random seeds yield identical splits, and different seeds differ."""
        n_samples = 100
        df = pd.DataFrame({
            "text": [f"sample_{i}" for i in range(n_samples)],
            "label": [i % 2 for i in range(n_samples)],
        })

        # Run 1 with seed 42
        tr1, val1, te1 = self.pipeline.stratified_split(df, random_seed=42)
        # Run 2 with seed 42
        tr2, val2, te2 = self.pipeline.stratified_split(df, random_seed=42)
        # Run 3 with seed 999
        tr3, val3, te3 = self.pipeline.stratified_split(df, random_seed=999)

        # Verify exact equality for same seed
        pd.testing.assert_frame_equal(tr1, tr2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(te1, te2)

        # Verify difference for distinct seed
        self.assertFalse(tr1["text"].equals(tr3["text"]))

    def test_save_splits_and_statistics(
        self,
    ):
        """Test writing split CSVs to disk and generating dataset statistics."""
        texts = [f"good movie review number {i}" for i in range(25)] + [f"bad movie review number {i}" for i in range(25)]
        labels = [1] * 25 + [0] * 25
        df = pd.DataFrame({
            "text": texts,
            "label": labels,
        })
        cleaned = self.pipeline.clean_and_validate(df)
        train_df, val_df, test_df = self.pipeline.stratified_split(cleaned, random_seed=42)


        saved_paths = self.pipeline.save_splits(train_df, val_df, test_df)
        self.assertTrue(saved_paths["train"].exists())
        self.assertTrue(saved_paths["val"].exists())
        self.assertTrue(saved_paths["test"].exists())

        # Verify loaded content
        loaded_train = pd.read_csv(saved_paths["train"])
        self.assertEqual(len(loaded_train), len(train_df))

        # Test statistics generation
        stats = DatasetPipeline.generate_dataset_statistics({
            "train": train_df,
            "val": val_df,
            "test": test_df,
        })
        self.assertIn("train", stats)
        self.assertIn("val", stats)
        self.assertIn("test", stats)
        self.assertEqual(stats["train"]["total_samples"], len(train_df))
        self.assertIn("positive_ratio", stats["train"])
        self.assertIn("char_length", stats["train"])
        self.assertIn("word_count", stats["train"])


if __name__ == "__main__":
    unittest.main()
