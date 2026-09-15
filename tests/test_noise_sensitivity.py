"""Unit tests for NoiseSensitivityAnalyzer.

Tests cover:
- assign_noise_group: boundary conditions for all four bands
- NoiseSensitivityAnalyzer.partition_samples: routing logic
- NoiseSensitivityAnalyzer.evaluate_model_on_groups: metric computation and
  insufficient-sample guard
- NoiseSensitivityAnalyzer.analyze: empty-input guard, multi-model aggregation
- save_artifacts: CSV header presence, row count, JSON structure
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.noise_sensitivity import (
    NoiseSensitivityAnalyzer,
    NoiseGroupMetrics,
    assign_noise_group,
    ALL_NOISE_GROUPS,
    NOISE_GROUP_LOW,
    NOISE_GROUP_MODERATE,
    NOISE_GROUP_HIGH,
    NOISE_GROUP_EXTREME,
)
from src.models.base import BaseSentimentModel


# ---------------------------------------------------------------------------
# Minimal stub model
# ---------------------------------------------------------------------------

class _ConstantModel(BaseSentimentModel):
    """Always returns a fixed score regardless of input."""

    def __init__(self, score: float = 0.8) -> None:
        self._s = float(score)

    def predict_score(self, text: str) -> float:  # type: ignore[override]
        return self._s

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        return [self._s] * len(texts)


# ---------------------------------------------------------------------------
# assign_noise_group boundary tests
# ---------------------------------------------------------------------------

class TestAssignNoiseGroup(unittest.TestCase):

    def test_zero_is_low(self):
        self.assertEqual(assign_noise_group(0.0), NOISE_GROUP_LOW)

    def test_boundary_low(self):
        self.assertEqual(assign_noise_group(0.20), NOISE_GROUP_LOW)

    def test_just_above_low_is_moderate(self):
        self.assertEqual(assign_noise_group(0.21), NOISE_GROUP_MODERATE)

    def test_boundary_moderate(self):
        self.assertEqual(assign_noise_group(0.50), NOISE_GROUP_MODERATE)

    def test_just_above_moderate_is_high(self):
        self.assertEqual(assign_noise_group(0.51), NOISE_GROUP_HIGH)

    def test_boundary_high(self):
        self.assertEqual(assign_noise_group(0.80), NOISE_GROUP_HIGH)

    def test_just_above_high_is_extreme(self):
        self.assertEqual(assign_noise_group(0.81), NOISE_GROUP_EXTREME)

    def test_one_is_extreme(self):
        self.assertEqual(assign_noise_group(1.0), NOISE_GROUP_EXTREME)

    def test_clamps_below_zero(self):
        self.assertEqual(assign_noise_group(-0.5), NOISE_GROUP_LOW)

    def test_clamps_above_one(self):
        self.assertEqual(assign_noise_group(2.0), NOISE_GROUP_EXTREME)

    def test_all_groups_covered(self):
        mapping = {
            NOISE_GROUP_LOW: 0.10,
            NOISE_GROUP_MODERATE: 0.35,
            NOISE_GROUP_HIGH: 0.65,
            NOISE_GROUP_EXTREME: 0.90,
        }
        for expected, score in mapping.items():
            self.assertEqual(assign_noise_group(score), expected, msg=f"score={score}")


# ---------------------------------------------------------------------------
# partition_samples tests
# ---------------------------------------------------------------------------

class TestPartitionSamples(unittest.TestCase):

    def _make_analyzer(self) -> NoiseSensitivityAnalyzer:
        return NoiseSensitivityAnalyzer(min_samples_per_group=1)

    def test_returns_all_groups(self):
        analyzer = self._make_analyzer()
        texts = ["hello world"]
        labels = [1]
        partitions = analyzer.partition_samples(texts, labels)
        self.assertEqual(set(partitions.keys()), set(ALL_NOISE_GROUPS))

    def test_partition_structure(self):
        analyzer = self._make_analyzer()
        partitions = analyzer.partition_samples(["good"], [1])
        for group in ALL_NOISE_GROUPS:
            self.assertIn("texts", partitions[group])
            self.assertIn("labels", partitions[group])
            self.assertIn("ids", partitions[group])
            self.assertIn("noise_scores", partitions[group])

    def test_total_sample_count_preserved(self):
        analyzer = self._make_analyzer()
        n = 8
        texts = ["hello"] * n
        labels = [0, 1] * (n // 2)
        partitions = analyzer.partition_samples(texts, labels)
        total = sum(len(partitions[g]["texts"]) for g in ALL_NOISE_GROUPS)
        self.assertEqual(total, n)

    def test_label_alignment(self):
        """Each partition must preserve label alignment with texts."""
        analyzer = self._make_analyzer()
        texts = ["good film", "bad film"] * 3
        labels = [1, 0] * 3
        partitions = analyzer.partition_samples(texts, labels)
        for group in ALL_NOISE_GROUPS:
            grp_texts = partitions[group]["texts"]
            grp_labels = partitions[group]["labels"]
            self.assertEqual(len(grp_texts), len(grp_labels))

    def test_custom_sample_ids(self):
        analyzer = self._make_analyzer()
        texts = ["hello", "world"]
        labels = [0, 1]
        ids = ["id_a", "id_b"]
        partitions = analyzer.partition_samples(texts, labels, sample_ids=ids)
        all_ids = []
        for g in ALL_NOISE_GROUPS:
            all_ids.extend(partitions[g]["ids"])
        self.assertIn("id_a", all_ids)
        self.assertIn("id_b", all_ids)

    def test_default_ids_are_integers(self):
        analyzer = self._make_analyzer()
        texts = ["a", "b", "c"]
        labels = [0, 1, 0]
        partitions = analyzer.partition_samples(texts, labels)
        all_ids = []
        for g in ALL_NOISE_GROUPS:
            all_ids.extend(partitions[g]["ids"])
        for idx in all_ids:
            self.assertIsInstance(idx, int)
            self.assertIn(idx, [0, 1, 2])


# ---------------------------------------------------------------------------
# evaluate_model_on_groups tests
# ---------------------------------------------------------------------------

class TestEvaluateModelOnGroups(unittest.TestCase):

    def _make_analyzer(self, min_samples: int = 1) -> NoiseSensitivityAnalyzer:
        return NoiseSensitivityAnalyzer(min_samples_per_group=min_samples)

    def _make_partitions_with_samples(self, n_per_group: int = 4):
        partitions = {
            g: {"texts": [], "labels": [], "ids": [], "noise_scores": []}
            for g in ALL_NOISE_GROUPS
        }
        for group in ALL_NOISE_GROUPS:
            for i in range(n_per_group):
                partitions[group]["texts"].append("test text")
                partitions[group]["labels"].append(i % 2)
                partitions[group]["ids"].append(i)
                partitions[group]["noise_scores"].append(0.1)
        return partitions

    def test_returns_four_results(self):
        analyzer = self._make_analyzer()
        partitions = self._make_partitions_with_samples(4)
        model = _ConstantModel(score=0.8)
        results = analyzer.evaluate_model_on_groups("TestModel", model, partitions)
        self.assertEqual(len(results), 4)

    def test_each_result_has_correct_model_name(self):
        analyzer = self._make_analyzer()
        partitions = self._make_partitions_with_samples(4)
        model = _ConstantModel(score=0.8)
        results = analyzer.evaluate_model_on_groups("MyModel", model, partitions)
        for r in results:
            self.assertEqual(r.model_name, "MyModel")

    def test_each_result_has_valid_noise_group(self):
        analyzer = self._make_analyzer()
        partitions = self._make_partitions_with_samples(4)
        model = _ConstantModel(score=0.8)
        results = analyzer.evaluate_model_on_groups("M", model, partitions)
        groups = {r.noise_group for r in results}
        self.assertEqual(groups, set(ALL_NOISE_GROUPS))

    def test_insufficient_samples_sets_status(self):
        analyzer = self._make_analyzer(min_samples=10)
        partitions = self._make_partitions_with_samples(4)
        model = _ConstantModel(score=0.8)
        results = analyzer.evaluate_model_on_groups("M", model, partitions)
        for r in results:
            self.assertEqual(r.status, "insufficient_samples")
            self.assertIsNone(r.accuracy)
            self.assertIsNone(r.macro_f1)

    def test_evaluated_status_when_sufficient(self):
        analyzer = self._make_analyzer(min_samples=1)
        partitions = self._make_partitions_with_samples(4)
        model = _ConstantModel(score=0.8)
        results = analyzer.evaluate_model_on_groups("M", model, partitions)
        for r in results:
            self.assertEqual(r.status, "evaluated")
            self.assertIsNotNone(r.accuracy)

    def test_metrics_in_valid_range(self):
        analyzer = self._make_analyzer(min_samples=1)
        partitions = self._make_partitions_with_samples(4)
        model = _ConstantModel(score=0.8)
        results = analyzer.evaluate_model_on_groups("M", model, partitions)
        for r in results:
            if r.status == "evaluated":
                self.assertGreaterEqual(r.accuracy, 0.0)
                self.assertLessEqual(r.accuracy, 1.0)

    def test_sample_count_matches_partition(self):
        analyzer = self._make_analyzer(min_samples=1)
        partitions = self._make_partitions_with_samples(4)
        model = _ConstantModel(score=0.8)
        results = analyzer.evaluate_model_on_groups("M", model, partitions)
        for r in results:
            self.assertEqual(r.sample_count, 4)


# ---------------------------------------------------------------------------
# analyze() integration tests
# ---------------------------------------------------------------------------

class TestNoiseSensitivityAnalyzerAnalyze(unittest.TestCase):

    def test_empty_texts_raises_value_error(self):
        analyzer = NoiseSensitivityAnalyzer()
        with self.assertRaises(ValueError):
            analyzer.analyze({"m": _ConstantModel()}, [], [])

    def test_multi_model_returns_4_results_per_model(self):
        analyzer = NoiseSensitivityAnalyzer(min_samples_per_group=1)
        texts = ["good film"] * 6
        labels = [1, 0, 1, 0, 1, 0]
        models = {"A": _ConstantModel(0.8), "B": _ConstantModel(0.2)}
        results = analyzer.analyze(models, texts, labels)
        self.assertEqual(len(results), 8)

    def test_all_model_names_present(self):
        analyzer = NoiseSensitivityAnalyzer(min_samples_per_group=1)
        texts = ["test"] * 4
        labels = [0, 1, 0, 1]
        models = {"ModelX": _ConstantModel(0.7), "ModelY": _ConstantModel(0.3)}
        results = analyzer.analyze(models, texts, labels)
        names = {r.model_name for r in results}
        self.assertIn("ModelX", names)
        self.assertIn("ModelY", names)

    def test_single_model_single_sample(self):
        analyzer = NoiseSensitivityAnalyzer(min_samples_per_group=1)
        texts = ["good"]
        labels = [1]
        models = {"M": _ConstantModel(0.9)}
        results = analyzer.analyze(models, texts, labels)
        self.assertEqual(len(results), 4)


# ---------------------------------------------------------------------------
# save_artifacts tests
# ---------------------------------------------------------------------------

class TestNoiseSensitivitySaveArtifacts(unittest.TestCase):

    def _make_fake_results(self) -> list:
        return [
            NoiseGroupMetrics(
                model_name="M",
                noise_group=g,
                sample_count=5,
                accuracy=0.80,
                macro_f1=0.79,
                binary_f1=0.80,
                precision=0.81,
                recall=0.78,
                status="evaluated",
            )
            for g in ALL_NOISE_GROUPS
        ]

    def test_csv_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = NoiseSensitivityAnalyzer(output_dir=tmpdir)
            results = self._make_fake_results()
            paths = analyzer.save_artifacts(results)
            self.assertTrue(paths["csv"].exists())

    def test_json_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = NoiseSensitivityAnalyzer(output_dir=tmpdir)
            results = self._make_fake_results()
            paths = analyzer.save_artifacts(results)
            self.assertTrue(paths["json"].exists())

    def test_csv_has_correct_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = NoiseSensitivityAnalyzer(output_dir=tmpdir)
            results = self._make_fake_results()
            paths = analyzer.save_artifacts(results)
            with open(paths["csv"], newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
            expected_cols = {"model", "noise_group", "samples", "accuracy", "macro_f1", "status"}
            self.assertTrue(expected_cols.issubset(set(header)))

    def test_csv_row_count_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = NoiseSensitivityAnalyzer(output_dir=tmpdir)
            results = self._make_fake_results()
            paths = analyzer.save_artifacts(results)
            with open(paths["csv"], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), len(results))

    def test_json_contains_definitions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = NoiseSensitivityAnalyzer(output_dir=tmpdir)
            results = self._make_fake_results()
            paths = analyzer.save_artifacts(results)
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("noise_group_definitions", data)
            self.assertIn("results", data)

    def test_no_fabricated_metrics_in_insufficient_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = NoiseSensitivityAnalyzer(output_dir=tmpdir)
            results = [
                NoiseGroupMetrics(
                    model_name="M",
                    noise_group=g,
                    sample_count=0,
                    accuracy=None,
                    macro_f1=None,
                    binary_f1=None,
                    precision=None,
                    recall=None,
                    status="insufficient_samples",
                )
                for g in ALL_NOISE_GROUPS
            ]
            paths = analyzer.save_artifacts(results)
            with open(paths["csv"], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                self.assertEqual(row["accuracy"], "")
                self.assertEqual(row["macro_f1"], "")


if __name__ == "__main__":
    unittest.main()
