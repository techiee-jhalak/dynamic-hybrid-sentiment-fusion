"""Unit tests for Statistical Hypothesis Testing (McNemar's Test).

Tests cover:
- McNemarResult dataclass: field types, to_dict(), to_csv_row()
- mcnemar_test: perfect agreement (p=1), one-sided dominance, empty arrays,
  length mismatch, no continuity correction, significance determination
- StatisticalComparator.compare_models: multi-model comparison
- save_artifacts: CSV header, row count, JSON structure, alpha stored correctly
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistical_tests import (
    McNemarResult,
    StatisticalComparator,
    mcnemar_test,
)


# ---------------------------------------------------------------------------
# mcnemar_test unit tests
# ---------------------------------------------------------------------------

class TestMcNemarTest(unittest.TestCase):

    # --- error guards ---

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mcnemar_test([0, 1], [0], [0, 1])

    def test_empty_arrays_raise(self):
        with self.assertRaises(ValueError):
            mcnemar_test([], [], [])

    # --- perfect agreement ---

    def test_both_always_correct(self):
        """When both models make identical correct predictions, p=1."""
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        result = mcnemar_test(y_true, y_pred, y_pred)
        self.assertEqual(result.both_correct, 5)
        self.assertEqual(result.model_a_only, 0)
        self.assertEqual(result.model_b_only, 0)
        self.assertAlmostEqual(result.p_value, 1.0)
        self.assertFalse(result.is_significant)

    def test_both_always_wrong(self):
        """When both models are always wrong, no discordant pairs → p=1."""
        y_true = [1, 1, 1]
        y_a = [0, 0, 0]
        y_b = [0, 0, 0]
        result = mcnemar_test(y_true, y_a, y_b)
        self.assertEqual(result.both_incorrect, 3)
        self.assertEqual(result.model_a_only, 0)
        self.assertAlmostEqual(result.p_value, 1.0)

    # --- contingency cell counting ---

    def test_model_a_dominates(self):
        """Model A correct on all, B wrong on all → b > c → A is superior."""
        y_true = [1, 1, 1, 1, 1, 1, 1, 1]
        y_pred_a = [1, 1, 1, 1, 1, 1, 1, 1]  # all correct
        y_pred_b = [0, 0, 0, 0, 0, 0, 0, 0]  # all wrong
        result = mcnemar_test(
            y_true, y_pred_a, y_pred_b,
            model_a_name="ModelA", model_b_name="ModelB"
        )
        self.assertEqual(result.model_a_only, 8)
        self.assertEqual(result.model_b_only, 0)
        self.assertEqual(result.superior_model, "ModelA")

    def test_model_b_dominates(self):
        y_true = [1, 1, 1, 1]
        y_pred_a = [0, 0, 0, 0]  # all wrong
        y_pred_b = [1, 1, 1, 1]  # all correct
        result = mcnemar_test(
            y_true, y_pred_a, y_pred_b,
            model_a_name="A", model_b_name="B"
        )
        self.assertEqual(result.superior_model, "B")
        self.assertEqual(result.model_b_only, 4)

    def test_tied_models(self):
        """Identical predictions → tied."""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 0, 0]
        result = mcnemar_test(y_true, y_pred, y_pred)
        self.assertEqual(result.superior_model, "tied")

    # --- statistical properties ---

    def test_significance_with_large_discordance(self):
        """Strong asymmetry with many samples should produce significant p-value."""
        n = 100
        y_true = [1] * n
        y_pred_a = [1] * n           # all correct
        y_pred_b = [0] * 80 + [1] * 20  # 80 wrong, 20 correct
        result = mcnemar_test(y_true, y_pred_a, y_pred_b, alpha=0.05)
        self.assertTrue(result.is_significant)
        self.assertLess(result.p_value, 0.05)

    def test_p_value_in_range(self):
        y_true = [1, 0, 1, 0, 1, 0]
        y_a = [1, 0, 1, 0, 0, 1]
        y_b = [0, 1, 1, 0, 1, 0]
        result = mcnemar_test(y_true, y_a, y_b)
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)

    def test_statistic_nonnegative(self):
        y_true = [1, 0, 1, 0, 1]
        y_a = [1, 0, 0, 1, 1]
        y_b = [1, 1, 0, 0, 0]
        result = mcnemar_test(y_true, y_a, y_b)
        self.assertGreaterEqual(result.statistic, 0.0)

    def test_no_continuity_correction_gives_higher_or_equal_stat(self):
        """Without continuity correction, statistic should be >= corrected version."""
        y_true = [1, 0, 1, 0, 1, 0]
        y_a = [1, 0, 1, 1, 0, 0]
        y_b = [0, 1, 1, 0, 1, 0]
        corrected = mcnemar_test(y_true, y_a, y_b, continuity_correction=True)
        uncorrected = mcnemar_test(y_true, y_a, y_b, continuity_correction=False)
        self.assertGreaterEqual(uncorrected.statistic, corrected.statistic)

    def test_alpha_stored_correctly(self):
        y_true = [1, 0, 1, 0]
        y_a = [1, 0, 0, 1]
        y_b = [0, 1, 1, 0]
        result = mcnemar_test(y_true, y_a, y_b, alpha=0.01)
        self.assertAlmostEqual(result.alpha, 0.01)

    def test_sample_count_matches_input(self):
        y_true = [1, 0, 1, 0, 1, 0, 1]
        y_a = [1, 0, 0, 1, 1, 0, 1]
        y_b = [0, 1, 1, 0, 1, 0, 1]
        result = mcnemar_test(y_true, y_a, y_b)
        self.assertEqual(result.sample_count, 7)

    # --- to_dict / to_csv_row ---

    def test_to_dict_has_required_keys(self):
        y_true = [1, 0, 1, 0]
        y_a = [1, 0, 0, 1]
        y_b = [0, 1, 1, 0]
        result = mcnemar_test(y_true, y_a, y_b)
        d = result.to_dict()
        for key in ("model_a", "model_b", "sample_count", "statistic", "p_value",
                    "alpha", "is_significant", "superior_model", "contingency_table"):
            self.assertIn(key, d, msg=f"Missing key: {key}")

    def test_to_csv_row_has_required_keys(self):
        y_true = [1, 0, 1, 0]
        y_a = [1, 0, 0, 1]
        y_b = [0, 1, 1, 0]
        result = mcnemar_test(y_true, y_a, y_b)
        row = result.to_csv_row()
        for key in ("model_a", "model_b", "samples", "statistic", "p_value",
                    "alpha", "significant_at_005", "superior_model"):
            self.assertIn(key, row, msg=f"Missing key: {key}")

    def test_contingency_sums_to_sample_count(self):
        y_true = [1, 0, 1, 0, 1]
        y_a = [1, 0, 1, 1, 0]
        y_b = [1, 1, 0, 0, 1]
        result = mcnemar_test(y_true, y_a, y_b)
        total = (result.both_correct + result.model_a_only
                 + result.model_b_only + result.both_incorrect)
        self.assertEqual(total, result.sample_count)

    def test_numpy_arrays_accepted(self):
        """Input as numpy arrays should work without error."""
        y_true = np.array([1, 0, 1, 0])
        y_a = np.array([1, 0, 0, 1])
        y_b = np.array([0, 1, 1, 0])
        result = mcnemar_test(y_true, y_a, y_b)
        self.assertIsInstance(result, McNemarResult)


# ---------------------------------------------------------------------------
# StatisticalComparator tests
# ---------------------------------------------------------------------------

class TestStatisticalComparator(unittest.TestCase):

    def test_compare_models_returns_one_result_per_comparator(self):
        comp = StatisticalComparator(alpha=0.05)
        y_true = [1, 0, 1, 0, 1, 0]
        ref_preds = [1, 0, 1, 0, 0, 1]
        comp_map = {
            "Baseline_A": [1, 0, 0, 1, 1, 0],
            "Baseline_B": [0, 1, 1, 0, 1, 0],
        }
        results = comp.compare_models(y_true, ref_preds, comp_map)
        self.assertEqual(len(results), 2)

    def test_compare_models_reference_name_is_model_a(self):
        comp = StatisticalComparator(alpha=0.05)
        y_true = [1, 0, 1, 0]
        ref_preds = [1, 0, 0, 1]
        comp_map = {"Other": [0, 1, 1, 0]}
        results = comp.compare_models(y_true, ref_preds, comp_map,
                                      reference_name="MyRef")
        self.assertEqual(results[0].model_a, "MyRef")

    def test_compare_models_comparator_name_is_model_b(self):
        comp = StatisticalComparator(alpha=0.05)
        y_true = [1, 0, 1, 0]
        ref_preds = [1, 0, 0, 1]
        comp_map = {"CompX": [0, 1, 1, 0]}
        results = comp.compare_models(y_true, ref_preds, comp_map)
        self.assertEqual(results[0].model_b, "CompX")

    def test_alpha_passed_to_test(self):
        comp = StatisticalComparator(alpha=0.01)
        y_true = [1, 0, 1, 0]
        ref_preds = [1, 0, 0, 1]
        comp_map = {"X": [0, 1, 1, 0]}
        results = comp.compare_models(y_true, ref_preds, comp_map)
        self.assertAlmostEqual(results[0].alpha, 0.01)


# ---------------------------------------------------------------------------
# save_artifacts tests
# ---------------------------------------------------------------------------

class TestStatisticalTestsSaveArtifacts(unittest.TestCase):

    def _make_results(self, n: int = 2):
        y_true = [1, 0, 1, 0, 1]
        y_a = [1, 0, 1, 0, 0]
        y_b = [0, 1, 1, 0, 1]
        return [
            mcnemar_test(y_true, y_a, y_b, model_a_name=f"A{i}", model_b_name=f"B{i}")
            for i in range(n)
        ]

    def test_csv_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = StatisticalComparator(output_dir=tmpdir)
            results = self._make_results()
            paths = comp.save_artifacts(results)
            self.assertTrue(paths["csv"].exists())

    def test_json_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = StatisticalComparator(output_dir=tmpdir)
            results = self._make_results()
            paths = comp.save_artifacts(results)
            self.assertTrue(paths["json"].exists())

    def test_csv_header_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = StatisticalComparator(output_dir=tmpdir)
            results = self._make_results(2)
            paths = comp.save_artifacts(results)
            with open(paths["csv"], newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = set(reader.fieldnames)
            expected = {"model_a", "model_b", "samples", "statistic", "p_value",
                        "alpha", "significant_at_005", "superior_model"}
            self.assertTrue(expected.issubset(header))

    def test_csv_row_count(self):
        n = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = StatisticalComparator(output_dir=tmpdir)
            results = self._make_results(n)
            paths = comp.save_artifacts(results)
            with open(paths["csv"], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), n)

    def test_json_has_comparisons_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = StatisticalComparator(output_dir=tmpdir)
            results = self._make_results(2)
            paths = comp.save_artifacts(results)
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("comparisons", data)

    def test_json_alpha_matches_comparator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = StatisticalComparator(alpha=0.05, output_dir=tmpdir)
            results = self._make_results(1)
            paths = comp.save_artifacts(results)
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            self.assertAlmostEqual(data["significance_threshold_alpha"], 0.05)


if __name__ == "__main__":
    unittest.main()
