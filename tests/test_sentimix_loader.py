"""Tests for the SentiMix CoNLL parser and data loader.

These tests verify:
- CoNLL parser correctly reads meta/token/lang_tag structure
- Label parsing for 3-class (positive / negative / neutral)
- Raw text is preserved (emojis, repeated chars, punctuation, symbols)
- train/dev/test loading from actual files (when available)
- Test-label alignment by UID matching
- Duplicate/leakage detection
- Dataset statistics
- Existing noise formulas are unaffected (smoke test)
- Existing router behavior is unaffected (smoke test)
"""

from __future__ import annotations

import textwrap
from io import StringIO
from pathlib import Path
from typing import List
from unittest.mock import mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Import the new loader
# ---------------------------------------------------------------------------
from src.data.sentimix_loader import (
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    SENTIMIX_LABEL_INT2STR,
    SENTIMIX_LABEL_STR2INT,
    SentimixCoNLLParser,
    SentimixDataLoader,
    SentimixSample,
)

# ---------------------------------------------------------------------------
# Fixtures: in-memory mock CoNLL content
# ---------------------------------------------------------------------------

TRAIN_MOCK = textwrap.dedent("""\
    meta\t101\tpositive
    yaar\tHin
    kya\tHin
    baat\tHin
    hai\tHin
    !\tO

    meta\t102\tnegative
    this\tEng
    is\tEng
    pathetic\tEng
    😠\tO

    meta\t103\tneutral
    okay\tEng
    theek\tHin
    hai\tHin

""")

DEV_MOCK = textwrap.dedent("""\
    meta\t201\tpositive
    amazing\tEng
    yaar\tHin

    meta\t202\tneutral
    shayad\tHin
    maybe\tEng

""")

TEST_MOCK_UNLABELLED = textwrap.dedent("""\
    meta\t301
    great\tEng
    !!!!\tO

    meta\t302
    bahut\tHin
    bura\tHin

""")

TEST_LABELS_MOCK = textwrap.dedent("""\
    Uid,Sentiment
    301,positive
    302,negative
""")


# ---------------------------------------------------------------------------
# Unit tests: label constants
# ---------------------------------------------------------------------------

class TestLabelConstants:
    def test_label_str2int_keys(self):
        assert set(SENTIMIX_LABEL_STR2INT.keys()) == {"positive", "negative", "neutral"}

    def test_label_int2str_values(self):
        assert set(SENTIMIX_LABEL_INT2STR.values()) == {"positive", "negative", "neutral"}

    def test_positive_constant(self):
        assert LABEL_POSITIVE == SENTIMIX_LABEL_STR2INT["positive"]

    def test_negative_constant(self):
        assert LABEL_NEGATIVE == SENTIMIX_LABEL_STR2INT["negative"]

    def test_neutral_constant(self):
        assert LABEL_NEUTRAL == SENTIMIX_LABEL_STR2INT["neutral"]

    def test_no_silent_neutral_mapping(self):
        """Neutral must be a distinct third class — NOT mapped to 0 or 1."""
        assert LABEL_NEUTRAL not in (0, 1), (
            "Neutral must not be silently mapped to Positive or Negative"
        )

    def test_all_three_labels_distinct(self):
        assert len({LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL}) == 3


# ---------------------------------------------------------------------------
# Unit tests: CoNLL parser on mock data
# ---------------------------------------------------------------------------

class TestSentimixCoNLLParserMock:
    """Parse in-memory mock data without touching the filesystem."""

    def _parse_string(self, content: str, label_str: str = "") -> List[SentimixSample]:
        parser = SentimixCoNLLParser()
        tmp = Path("_mock_sentimix_test.txt")
        tmp.write_text(content, encoding="utf-8")
        try:
            samples = parser.parse_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return samples

    def test_train_sample_count(self):
        samples = self._parse_string(TRAIN_MOCK)
        assert len(samples) == 3

    def test_positive_label_parsed(self):
        samples = self._parse_string(TRAIN_MOCK)
        assert samples[0].label == LABEL_POSITIVE
        assert samples[0].label_str == "positive"

    def test_negative_label_parsed(self):
        samples = self._parse_string(TRAIN_MOCK)
        assert samples[1].label == LABEL_NEGATIVE
        assert samples[1].label_str == "negative"

    def test_neutral_label_parsed(self):
        samples = self._parse_string(TRAIN_MOCK)
        assert samples[2].label == LABEL_NEUTRAL
        assert samples[2].label_str == "neutral"

    def test_uid_parsed(self):
        samples = self._parse_string(TRAIN_MOCK)
        assert samples[0].uid == "101"
        assert samples[1].uid == "102"
        assert samples[2].uid == "103"

    def test_tokens_preserved(self):
        samples = self._parse_string(TRAIN_MOCK)
        assert "yaar" in samples[0].tokens
        assert "baat" in samples[0].tokens

    def test_emoji_preserved_in_tokens(self):
        """Raw emoji characters must be preserved in tokens."""
        samples = self._parse_string(TRAIN_MOCK)
        # sample 102 (negative) contains emoji 😠
        assert "😠" in samples[1].tokens, "Emoji must be preserved in token list"

    def test_emoji_preserved_in_text(self):
        samples = self._parse_string(TRAIN_MOCK)
        assert "😠" in samples[1].text

    def test_punctuation_preserved(self):
        samples = self._parse_string(TRAIN_MOCK)
        assert "!" in samples[0].tokens

    def test_repeated_chars_preserved(self):
        """Repeated-char tokens like '!!!!' must not be collapsed."""
        tmp = Path("_mock_repeat.txt")
        tmp.write_text(
            "meta\t999\tpositive\nwooooow\tEng\n!!!!\tO\n\n",
            encoding="utf-8"
        )
        try:
            parser = SentimixCoNLLParser()
            samples = parser.parse_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        assert "wooooow" in samples[0].tokens
        assert "!!!!" in samples[0].tokens

    def test_lang_tags_parsed(self):
        samples = self._parse_string(TRAIN_MOCK)
        # sample 0: yaar Hin, kya Hin, baat Hin, hai Hin, ! O
        assert "Hin" in samples[0].lang_tags
        assert "O" in samples[0].lang_tags

    def test_text_is_space_joined_tokens(self):
        samples = self._parse_string(TRAIN_MOCK)
        reconstructed = " ".join(samples[0].tokens)
        assert samples[0].text == reconstructed

    def test_code_mixed_sample_has_both_tags(self):
        """A code-mixed sentence has both Eng and Hin tags."""
        samples = self._parse_string(TRAIN_MOCK)
        # sample 2 (neutral): okay Eng, theek Hin, hai Hin
        combined_langs = set(samples[2].lang_tags)
        assert "Eng" in combined_langs
        assert "Hin" in combined_langs

    def test_all_three_label_classes_present(self):
        samples = self._parse_string(TRAIN_MOCK)
        labels = {s.label for s in samples}
        assert LABEL_POSITIVE in labels
        assert LABEL_NEGATIVE in labels
        assert LABEL_NEUTRAL in labels


class TestSentimixTestSetAlignment:
    """Test the UID-based label alignment between test file and label file."""

    def test_aligned_labels_loaded(self):
        tmp_test   = Path("_mock_test_unlabelled.txt")
        tmp_labels = Path("_mock_test_labels.txt")
        tmp_test.write_text(TEST_MOCK_UNLABELLED, encoding="utf-8")
        tmp_labels.write_text(TEST_LABELS_MOCK, encoding="utf-8")
        try:
            parser = SentimixCoNLLParser()
            samples = parser.parse_file(tmp_test, label_file=tmp_labels)
        finally:
            tmp_test.unlink(missing_ok=True)
            tmp_labels.unlink(missing_ok=True)

        assert len(samples) == 2
        uid_to_label = {s.uid: s.label for s in samples}
        assert uid_to_label["301"] == LABEL_POSITIVE
        assert uid_to_label["302"] == LABEL_NEGATIVE

    def test_unlabelled_without_label_file_yields_none(self):
        tmp_test = Path("_mock_test_nolabels.txt")
        tmp_test.write_text(TEST_MOCK_UNLABELLED, encoding="utf-8")
        try:
            parser = SentimixCoNLLParser()
            samples = parser.parse_file(tmp_test)
        finally:
            tmp_test.unlink(missing_ok=True)

        for s in samples:
            assert s.label is None


# ---------------------------------------------------------------------------
# Integration tests: real files (skipped when files are absent)
# ---------------------------------------------------------------------------

SENTIMIX_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "sentimix"
)

requires_sentimix = pytest.mark.skipif(
    not (SENTIMIX_DATA_DIR / "Hinglish_train_14k_split_conll.txt").exists(),
    reason="SentiMix data files not available at data/raw/sentimix/",
)


class TestSentimixRealFiles:
    """Integration tests that run only when data/raw/sentimix/ is populated."""

    @requires_sentimix
    def test_train_count(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_train()
        assert len(df) == 14000, f"Expected 14000 train samples, got {len(df)}"

    @requires_sentimix
    def test_dev_count(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_dev()
        assert len(df) == 3000, f"Expected 3000 dev samples, got {len(df)}"

    @requires_sentimix
    def test_test_count(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_test()
        assert len(df) == 3000, f"Expected 3000 test samples, got {len(df)}"

    @requires_sentimix
    def test_train_label_distribution(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_train()
        counts = df["label_str"].value_counts()
        assert counts.get("positive", 0) == 4634
        assert counts.get("negative", 0) == 4102
        assert counts.get("neutral",  0) == 5264

    @requires_sentimix
    def test_dev_label_distribution(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_dev()
        counts = df["label_str"].value_counts()
        assert counts.get("positive", 0) == 982
        assert counts.get("negative", 0) == 890
        assert counts.get("neutral",  0) == 1128

    @requires_sentimix
    def test_test_label_distribution(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_test()
        counts = df["label_str"].value_counts()
        assert counts.get("positive", 0) == 1000
        assert counts.get("negative", 0) == 900
        assert counts.get("neutral",  0) == 1100

    @requires_sentimix
    def test_no_null_labels_in_train(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_train()
        assert df["label"].isna().sum() == 0

    @requires_sentimix
    def test_test_label_alignment_by_uid(self):
        """All test UIDs must have a corresponding label (no None labels)."""
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_test()
        assert df["label"].isna().sum() == 0, "All test UIDs should be covered by label file"

    @requires_sentimix
    def test_no_duplicates_within_train(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_train()
        assert df["uid"].duplicated().sum() == 0

    @requires_sentimix
    def test_no_duplicates_within_dev(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_dev()
        assert df["uid"].duplicated().sum() == 0

    @requires_sentimix
    def test_no_train_dev_leakage(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        train_df = loader.load_train()
        dev_df = loader.load_dev()
        overlap = set(train_df["uid"]) & set(dev_df["uid"])
        assert len(overlap) == 0, f"Train-Dev UID overlap: {len(overlap)}"

    @requires_sentimix
    def test_no_train_test_leakage(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        train_df = loader.load_train()
        test_df = loader.load_test()
        overlap = set(train_df["uid"]) & set(test_df["uid"])
        assert len(overlap) == 0, f"Train-Test UID overlap: {len(overlap)}"

    @requires_sentimix
    def test_raw_text_not_empty(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_train()
        assert (df["text"].str.strip() == "").sum() == 0

    @requires_sentimix
    def test_three_class_labels_only(self):
        """Verify no label outside {0, 1, 2} exists."""
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        train_df = loader.load_train()
        valid_labels = {LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL}
        bad = train_df[~train_df["label"].isin(valid_labels)]
        assert len(bad) == 0, f"Unexpected labels found: {bad['label'].unique()}"

    @requires_sentimix
    def test_dataset_statistics_structure(self):
        loader = SentimixDataLoader(data_dir=SENTIMIX_DATA_DIR)
        df = loader.load_train()
        stats = loader.dataset_statistics(df, name="train")
        assert stats["total"] == 14000
        assert "positive" in stats
        assert "negative" in stats
        assert "neutral" in stats


# ---------------------------------------------------------------------------
# Smoke tests: existing noise formulas remain unaffected
# ---------------------------------------------------------------------------

class TestExistingNoiseFormulas:
    """Verify core noise-quantification formulas are unchanged."""

    def test_noise_formula_structure(self):
        from src.features.noise_quantifier import NoiseQuantifier
        nq = NoiseQuantifier()
        # Minimal smoke test — does not require real model
        result = nq.extract_features("hello world yaar 😊 !!!")
        assert hasattr(result, "noise_score"), "NoiseQuantifier output must have noise_score"
        assert 0.0 <= result.noise_score <= 1.0, "N must be in [0, 1]"

    def test_noise_weights_unchanged(self):
        from configs.config import config
        assert config.noise.w_emoji    == 0.25
        assert config.noise.w_repetition == 0.25
        assert config.noise.w_code_mix == 0.30
        assert config.noise.w_symbol   == 0.20

    def test_router_thresholds_unchanged(self):
        from configs.config import config
        assert config.routing.reference_length == 20
        assert config.routing.noise_threshold  == 0.20
        assert config.routing.w1              == 0.05
        assert config.routing.w2              == 12.0
        assert config.routing.alpha_min       == 0.02
        assert config.routing.alpha_max       == 0.25
        assert config.routing.decision_threshold == 0.50

    def test_router_low_noise_gives_alpha_min(self):
        """When N <= 0.20, alpha must equal alpha_min (0.02)."""
        from src.models.adaptive_router import AdaptiveRouter
        router = AdaptiveRouter()
        alpha = router.compute_alpha(noise_score=0.10, token_length=15)
        assert alpha == pytest.approx(0.02), f"Expected alpha_min=0.02, got {alpha}"

    def test_router_high_noise_alpha_clamped(self):
        """When N is high, alpha must stay <= alpha_max (0.25)."""
        from src.models.adaptive_router import AdaptiveRouter
        router = AdaptiveRouter()
        alpha = router.compute_alpha(noise_score=0.99, token_length=5)
        assert alpha <= 0.25, f"Alpha must be clamped to 0.25, got {alpha}"


# ---------------------------------------------------------------------------
# 3-class model smoke test (no weights required)
# ---------------------------------------------------------------------------

class TestDistilBert3ClassSmoke:
    def test_output_structure(self):
        from src.models.distilbert_3class import DistilBert3ClassOutput, LABEL_NEUTRAL
        out = DistilBert3ClassOutput(
            positive_prob=0.3,
            negative_prob=0.3,
            neutral_prob=0.4,
            predicted_label=LABEL_NEUTRAL,
            logits=[0.1, 0.1, 0.2],
        )
        d = out.to_dict()
        assert "positive_prob" in d
        assert "negative_prob" in d
        assert "neutral_prob" in d
        assert "predicted_label" in d

    def test_three_labels_distinct_in_output(self):
        from src.models.distilbert_3class import LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL
        assert len({LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL}) == 3
