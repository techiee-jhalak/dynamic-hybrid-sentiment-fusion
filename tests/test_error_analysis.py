"""Unit tests for the Deterministic Error Analysis Module.

Tests cover:
- categorize_error: each of the five categories (cross-script, dual-sarcasm,
  implicit irony, slang drift, unavailable) and the not_an_error short-circuit
- ErrorRecord.to_dict: field presence and type correctness
- ErrorAnalyzer.analyze_errors: error detection, correct-sample skipping
- ErrorAnalyzer.save_artifacts: CSV/JSON structure, category distribution,
  no fabricated categories
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import (
    ErrorAnalyzer,
    ErrorRecord,
    categorize_error,
    ALL_ERROR_CATEGORIES,
    CAT_CROSS_SCRIPT,
    CAT_DUAL_SARCASM,
    CAT_IMPLICIT_IRONY,
    CAT_SLANG_DRIFT,
    CAT_UNAVAILABLE,
)


# ---------------------------------------------------------------------------
# categorize_error function tests
# ---------------------------------------------------------------------------

class TestCategorizeError(unittest.TestCase):

    def test_not_an_error_when_correct(self):
        category, rationale = categorize_error("great movie", true_label=1, predicted_label=1)
        self.assertEqual(category, "not_an_error")

    def test_cross_script_with_devanagari(self):
        # Devanagari + Latin → cross-script
        text = "यह movie bahut acchi hai"
        category, rationale = categorize_error(text, true_label=1, predicted_label=0)
        self.assertEqual(category, CAT_CROSS_SCRIPT)

    def test_dual_sarcasm_positive_emoji_on_negative_sample(self):
        # Negative label (0) with laughing/positive emoji → dual-sarcasm
        text = "the movie was terrible 😂"
        category, rationale = categorize_error(text, true_label=0, predicted_label=1)
        self.assertEqual(category, CAT_DUAL_SARCASM)

    def test_dual_sarcasm_negative_emoji_on_positive_sample(self):
        # Positive label (1) with sad/negative emoji → dual-sarcasm
        text = "absolutely loved this! 😢"
        category, rationale = categorize_error(text, true_label=1, predicted_label=0)
        self.assertEqual(category, CAT_DUAL_SARCASM)

    def test_implicit_irony_contrastive_marker(self):
        # Contains "but" and enough words → irony
        text = "The acting was brilliant but the story was completely awful and boring"
        category, rationale = categorize_error(text, true_label=0, predicted_label=1)
        self.assertEqual(category, CAT_IMPLICIT_IRONY)

    def test_implicit_irony_marker_pattern(self):
        # "yeah right" → irony marker pattern
        text = "yeah right this is the best movie ever"
        category, rationale = categorize_error(text, true_label=0, predicted_label=1)
        self.assertEqual(category, CAT_IMPLICIT_IRONY)

    def test_slang_drift_with_known_slang(self):
        text = "yeh toh ek dum bakwaas film hai"
        category, rationale = categorize_error(text, true_label=0, predicted_label=1)
        self.assertEqual(category, CAT_SLANG_DRIFT)

    def test_unavailable_for_clean_plain_text(self):
        # Clean English text with no special features → unavailable
        text = "The film was okay"
        category, rationale = categorize_error(text, true_label=1, predicted_label=0)
        self.assertEqual(category, CAT_UNAVAILABLE)

    def test_rationale_is_non_empty_string(self):
        text = "great movie"
        _, rationale = categorize_error(text, true_label=1, predicted_label=0)
        self.assertIsInstance(rationale, str)
        self.assertGreater(len(rationale), 0)

    def test_returns_tuple_of_two_strings(self):
        result = categorize_error("ok film", true_label=1, predicted_label=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], str)

    def test_category_is_always_in_known_set(self):
        """All returned categories should be in the known set."""
        test_cases = [
            ("यह movie hai", 1, 0),
            ("terrible 😂", 0, 1),
            ("great but the story was messy and long and terrible", 0, 1),
            ("yeah right this film", 0, 1),
            ("bakwaas film yaar", 0, 1),
            ("a simple film", 1, 0),
        ]
        valid = set(ALL_ERROR_CATEGORIES) | {"not_an_error"}
        for text, true, pred in test_cases:
            cat, _ = categorize_error(text, true_label=true, predicted_label=pred)
            self.assertIn(cat, valid, msg=f"Unexpected category '{cat}' for: {text!r}")


# ---------------------------------------------------------------------------
# ErrorRecord tests
# ---------------------------------------------------------------------------

class TestErrorRecord(unittest.TestCase):

    def _make_record(self, **kwargs) -> ErrorRecord:
        defaults = dict(
            sample_id=0,
            model_name="TestModel",
            raw_text="bad film 😂",
            true_label=0,
            predicted_label=1,
            confidence=0.85,
            noise_score=0.30,
            error_category=CAT_DUAL_SARCASM,
            rationale="Mismatch between emoji and label.",
        )
        defaults.update(kwargs)
        return ErrorRecord(**defaults)

    def test_to_dict_has_required_fields(self):
        rec = self._make_record()
        d = rec.to_dict()
        required = {
            "sample_id", "model", "text", "true_label", "predicted_label",
            "confidence", "noise_score", "error_category", "rationale",
        }
        self.assertTrue(required.issubset(set(d.keys())))

    def test_to_dict_confidence_rounded(self):
        rec = self._make_record(confidence=0.123456789)
        d = rec.to_dict()
        self.assertEqual(d["confidence"], round(0.123456789, 4))

    def test_to_dict_noise_score_rounded(self):
        rec = self._make_record(noise_score=0.987654321)
        d = rec.to_dict()
        self.assertEqual(d["noise_score"], round(0.987654321, 4))

    def test_to_dict_text_matches_raw_text(self):
        rec = self._make_record(raw_text="hello world")
        d = rec.to_dict()
        self.assertEqual(d["text"], "hello world")


# ---------------------------------------------------------------------------
# ErrorAnalyzer.analyze_errors tests
# ---------------------------------------------------------------------------

class TestErrorAnalyzerAnalyzeErrors(unittest.TestCase):

    def _make_analyzer(self) -> ErrorAnalyzer:
        return ErrorAnalyzer()

    def test_no_errors_returns_empty_list(self):
        analyzer = self._make_analyzer()
        texts = ["good film", "nice movie"]
        true_labels = [1, 1]
        pred_labels = [1, 1]
        scores = [0.9, 0.8]
        errors = analyzer.analyze_errors("M", texts, true_labels, pred_labels, scores)
        self.assertEqual(errors, [])

    def test_all_wrong_returns_all_as_errors(self):
        analyzer = self._make_analyzer()
        texts = ["bad film", "horrible story"]
        true_labels = [0, 0]
        pred_labels = [1, 1]
        scores = [0.8, 0.7]
        errors = analyzer.analyze_errors("M", texts, true_labels, pred_labels, scores)
        self.assertEqual(len(errors), 2)

    def test_partial_errors(self):
        analyzer = self._make_analyzer()
        texts = ["great movie", "terrible film"]
        true_labels = [1, 0]
        pred_labels = [1, 1]   # second is wrong
        scores = [0.9, 0.8]
        errors = analyzer.analyze_errors("M", texts, true_labels, pred_labels, scores)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].true_label, 0)
        self.assertEqual(errors[0].predicted_label, 1)

    def test_error_record_model_name(self):
        analyzer = self._make_analyzer()
        texts = ["ok film"]
        errors = analyzer.analyze_errors("MyModel", texts, [0], [1], [0.8])
        self.assertEqual(errors[0].model_name, "MyModel")

    def test_custom_sample_ids_preserved(self):
        analyzer = self._make_analyzer()
        texts = ["bad"]
        errors = analyzer.analyze_errors(
            "M", texts, [0], [1], [0.9], sample_ids=["custom_id"]
        )
        self.assertEqual(errors[0].sample_id, "custom_id")

    def test_default_sample_ids_are_integers(self):
        analyzer = self._make_analyzer()
        texts = ["a", "b"]
        errors = analyzer.analyze_errors("M", texts, [0, 0], [1, 1], [0.8, 0.7])
        for e in errors:
            self.assertIsInstance(e.sample_id, int)

    def test_confidence_is_positive_score_when_pred_positive(self):
        """When predicted label is 1, confidence should equal the raw score."""
        analyzer = self._make_analyzer()
        texts = ["test text"]
        errors = analyzer.analyze_errors("M", texts, [0], [1], [0.85])
        self.assertAlmostEqual(errors[0].confidence, 0.85)

    def test_confidence_is_1_minus_score_when_pred_negative(self):
        """When predicted label is 0, confidence = 1 - score."""
        analyzer = self._make_analyzer()
        texts = ["test text"]
        errors = analyzer.analyze_errors("M", texts, [1], [0], [0.20])
        self.assertAlmostEqual(errors[0].confidence, 0.80)

    def test_error_category_is_valid(self):
        analyzer = self._make_analyzer()
        texts = ["The film was okay but the script was mediocre and long and boring"]
        errors = analyzer.analyze_errors("M", texts, [0], [1], [0.7])
        if errors:
            valid = set(ALL_ERROR_CATEGORIES)
            self.assertIn(errors[0].error_category, valid)


# ---------------------------------------------------------------------------
# ErrorAnalyzer.save_artifacts tests
# ---------------------------------------------------------------------------

class TestErrorAnalyzerSaveArtifacts(unittest.TestCase):

    def _make_errors(self) -> List[ErrorRecord]:
        return [
            ErrorRecord(
                sample_id=i,
                model_name="M",
                raw_text="bad film",
                true_label=0,
                predicted_label=1,
                confidence=0.8,
                noise_score=0.3,
                error_category=CAT_UNAVAILABLE,
                rationale="No pattern matched.",
            )
            for i in range(5)
        ]

    def test_csv_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            errors = self._make_errors()
            paths = analyzer.save_artifacts(errors, total_evaluated_samples=20)
            self.assertTrue(paths["csv"].exists())

    def test_json_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            errors = self._make_errors()
            paths = analyzer.save_artifacts(errors, total_evaluated_samples=20)
            self.assertTrue(paths["json"].exists())

    def test_csv_header_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            errors = self._make_errors()
            paths = analyzer.save_artifacts(errors, total_evaluated_samples=20)
            with open(paths["csv"], newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = set(reader.fieldnames)
            expected = {"sample_id", "model", "true_label", "predicted_label",
                        "confidence", "noise_score", "error_category", "rationale", "text"}
            self.assertTrue(expected.issubset(header))

    def test_csv_row_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            errors = self._make_errors()
            paths = analyzer.save_artifacts(errors, total_evaluated_samples=20)
            with open(paths["csv"], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)

    def test_json_summary_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            errors = self._make_errors()
            paths = analyzer.save_artifacts(errors, total_evaluated_samples=20)
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            for key in ("model_name", "total_samples", "total_errors", "error_rate",
                        "category_distribution"):
                self.assertIn(key, data)

    def test_error_rate_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            errors = self._make_errors()  # 5 errors
            paths = analyzer.save_artifacts(errors, total_evaluated_samples=20)
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            self.assertAlmostEqual(data["error_rate"], 5 / 20, places=3)

    def test_zero_errors_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            paths = analyzer.save_artifacts([], total_evaluated_samples=10)
            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["json"].exists())

    def test_zero_error_rate_when_no_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            paths = analyzer.save_artifacts([], total_evaluated_samples=10)
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            self.assertAlmostEqual(data["error_rate"], 0.0)
            self.assertEqual(data["total_errors"], 0)

    def test_category_distribution_keys_are_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            errors = self._make_errors()
            paths = analyzer.save_artifacts(errors, total_evaluated_samples=20,
                                            model_name="M")
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            for key in data["category_distribution"]:
                self.assertIn(key, ALL_ERROR_CATEGORIES)

    def test_model_name_safe_filename(self):
        """Spaces and slashes in model name should not break filename creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ErrorAnalyzer(output_dir=tmpdir)
            errors = self._make_errors()
            paths = analyzer.save_artifacts(
                errors, total_evaluated_samples=20, model_name="Dynamic Fusion/v2"
            )
            self.assertTrue(paths["csv"].exists())


if __name__ == "__main__":
    unittest.main()
