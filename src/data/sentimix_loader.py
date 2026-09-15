"""SentiMix CoNLL-format dataset loader for SemEval-2020 Task 9 (Hindi-English).

This module is SEPARATE from the existing binary CSV pipeline (dataset.py).

Dataset: SemEval-2020 Task 9 / SentiMix — Hinglish (Hindi-English code-mixed)
Source:  https://ritual.uh.edu/semeval-2020-task9/
Format:  CoNLL — each sentence block headed by a "meta" line:
         meta  <uid>  <label>
         <token>  <lang_tag>
         ...
         <blank line>

Labels:  positive / negative / neutral  (3-class; NOT binary)

IMPORTANT:
- This dataset is NOT SAIL 2017.
- Do NOT map Neutral to Positive or Negative silently.
- This loader preserves raw tokens (emojis, repeated chars, punctuation, symbols).
- Preprocessing for noise features (E/R/C/S) must happen at the feature stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

SENTIMIX_LABEL_STR2INT: Dict[str, int] = {
    "positive": 0,
    "negative": 1,
    "neutral":  2,
}

SENTIMIX_LABEL_INT2STR: Dict[int, str] = {v: k for k, v in SENTIMIX_LABEL_STR2INT.items()}

# Integer label aliases (the final int encoding used in DataFrames)
LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1
LABEL_NEUTRAL  = 2

# Language tags used in the CoNLL files
LANG_TAG_ENG = "Eng"
LANG_TAG_HIN = "Hin"
LANG_TAG_OTHER = "O"  # symbols, other tokens


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SentimixSample:
    """A single SentiMix sentence sample."""
    uid: str
    text: str               # Reconstructed space-joined surface form (raw, no normalization)
    tokens: List[str]       # Individual tokens preserving emojis/symbols/repeated chars
    lang_tags: List[str]    # Per-token language tags (Eng / Hin / O)
    label: Optional[int]    # LABEL_POSITIVE/NEGATIVE/NEUTRAL, or None for unlabelled test
    label_str: Optional[str] = None  # "positive" / "negative" / "neutral" / None


# ---------------------------------------------------------------------------
# CoNLL parser
# ---------------------------------------------------------------------------

class SentimixCoNLLParser:
    """Parse SemEval-2020 Task 9 SentiMix CoNLL files into SentimixSample objects.

    File format (train / dev):
        meta\t<uid>\t<label>
        <token>\t<lang_tag>
        ...
        <blank line>

    File format (unlabelled test):
        meta\t<uid>
        <token>\t<lang_tag>
        ...
        <blank line>
    """

    def parse_file(
        self,
        filepath: Path,
        label_file: Optional[Path] = None,
    ) -> List[SentimixSample]:
        """Parse a SentiMix CoNLL file.

        Args:
            filepath: Path to the CoNLL file.
            label_file: Optional path to a separate label CSV (uid, label) for test set.

        Returns:
            List of SentimixSample objects.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"SentiMix file not found: {filepath}")

        # Load external label mapping if provided (for the test set)
        label_map: Dict[str, int] = {}
        if label_file is not None:
            label_map = self._load_label_file(Path(label_file))

        samples: List[SentimixSample] = []
        current_uid: Optional[str] = None
        current_label: Optional[int] = None
        current_label_str: Optional[str] = None
        current_tokens: List[str] = []
        current_langs: List[str] = []

        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n").rstrip("\r")

                if line.startswith("meta\t"):
                    # Flush previous sentence if exists
                    if current_uid is not None:
                        samples.append(self._build_sample(
                            current_uid, current_label, current_label_str,
                            current_tokens, current_langs,
                        ))

                    parts = line.split("\t")
                    current_uid = parts[1].strip() if len(parts) > 1 else ""
                    current_label = None
                    current_label_str = None

                    if len(parts) >= 3:
                        # Labelled (train / dev)
                        raw_label = parts[2].strip().lower()
                        current_label_str = raw_label
                        current_label = SENTIMIX_LABEL_STR2INT.get(raw_label)
                    elif current_uid in label_map:
                        # Unlabelled test + external label file
                        current_label = label_map[current_uid]
                        current_label_str = SENTIMIX_LABEL_INT2STR.get(current_label)

                    current_tokens = []
                    current_langs = []

                elif line.strip() == "":
                    # Blank line — end of sentence block; will be flushed on next meta or EOF
                    pass

                else:
                    # Token line: <token>\t<lang_tag>
                    parts = line.split("\t")
                    token = parts[0] if parts else ""
                    lang = parts[1].strip() if len(parts) > 1 else LANG_TAG_OTHER
                    if token:  # ignore truly empty token lines
                        current_tokens.append(token)
                        current_langs.append(lang)

        # Flush final sentence
        if current_uid is not None and current_tokens:
            samples.append(self._build_sample(
                current_uid, current_label, current_label_str,
                current_tokens, current_langs,
            ))

        return samples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sample(
        uid: str,
        label: Optional[int],
        label_str: Optional[str],
        tokens: List[str],
        langs: List[str],
    ) -> SentimixSample:
        text = " ".join(tokens)
        return SentimixSample(
            uid=uid,
            text=text,
            tokens=list(tokens),
            lang_tags=list(langs),
            label=label,
            label_str=label_str,
        )

    @staticmethod
    def _load_label_file(label_file: Path) -> Dict[str, int]:
        """Parse the test label CSV (Uid,Sentiment) into {uid: int_label}."""
        label_map: Dict[str, int] = {}
        with open(label_file, encoding="utf-8") as fh:
            for i, raw_line in enumerate(fh):
                line = raw_line.strip()
                if i == 0 and line.lower().startswith("uid"):
                    continue  # skip header
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                uid = parts[0].strip()
                sentiment_str = parts[1].strip().lower()
                int_label = SENTIMIX_LABEL_STR2INT.get(sentiment_str)
                if int_label is not None:
                    label_map[uid] = int_label
        return label_map


# ---------------------------------------------------------------------------
# DataFrame loader
# ---------------------------------------------------------------------------

class SentimixDataLoader:
    """High-level loader that returns pandas DataFrames from SentiMix files.

    NOTE: This loader does NOT apply noise-feature preprocessing.
    Raw text is preserved exactly for downstream E/R/C/S computation.
    """

    # Paths relative to project root
    _DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "sentimix"

    TRAIN_FILE = "Hinglish_train_14k_split_conll.txt"
    DEV_FILE   = "Hinglish_dev_3k_split_conll.txt"
    TEST_FILE  = "Hinglish_test_unlabelled_conll_updated.txt"
    LABEL_FILE = "Hinglish_test_labels.txt"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else self._DATA_DIR
        self._parser = SentimixCoNLLParser()

    def _samples_to_df(self, samples: List[SentimixSample]) -> pd.DataFrame:
        """Convert a list of SentimixSample objects to a DataFrame."""
        rows = []
        for s in samples:
            rows.append({
                "uid":       s.uid,
                "text":      s.text,
                "tokens":    s.tokens,
                "lang_tags": s.lang_tags,
                "label":     s.label,       # int or None
                "label_str": s.label_str,   # "positive"/"negative"/"neutral"/None
            })
        return pd.DataFrame(rows)

    def load_train(self) -> pd.DataFrame:
        """Load training split (14,000 samples, labelled)."""
        samples = self._parser.parse_file(self.data_dir / self.TRAIN_FILE)
        return self._samples_to_df(samples)

    def load_dev(self) -> pd.DataFrame:
        """Load development/validation split (3,000 samples, labelled)."""
        samples = self._parser.parse_file(self.data_dir / self.DEV_FILE)
        return self._samples_to_df(samples)

    def load_test(self) -> pd.DataFrame:
        """Load test split (3,000 samples) merged with label file.

        Test sentence IDs are matched by UID to the label file.
        """
        label_file = self.data_dir / self.LABEL_FILE
        samples = self._parser.parse_file(
            self.data_dir / self.TEST_FILE,
            label_file=label_file if label_file.exists() else None,
        )
        return self._samples_to_df(samples)

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train_df, dev_df, test_df).

        IMPORTANT: The SentiMix-provided train/dev/test splits are preserved as-is.
        No re-splitting is performed. Use these splits for all experiments.
        """
        return self.load_train(), self.load_dev(), self.load_test()

    def dataset_statistics(self, df: pd.DataFrame, name: str = "split") -> dict:
        """Generate basic statistics for a loaded DataFrame."""
        if df.empty:
            return {"name": name, "total": 0}

        label_counts = df["label_str"].value_counts().to_dict()
        return {
            "name": name,
            "total": len(df),
            "positive": int(label_counts.get("positive", 0)),
            "negative": int(label_counts.get("negative", 0)),
            "neutral":  int(label_counts.get("neutral",  0)),
            "unlabelled": int(df["label"].isna().sum()),
        }
