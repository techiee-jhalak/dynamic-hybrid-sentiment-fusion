"""Unit tests for the Evaluation Engine.

Covers:
- ModelResult structure and scalar dict
- EvaluationEngine.evaluate_single: metrics, latency, per-sample records
- EvaluationEngine.run: multi-model evaluation
- File artifact writing: metrics.csv, predictions.csv, confusion_matrices/, experiment_summary.json
- Edge cases: empty test set, single-class labels, threshold boundary
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

from src.evaluation.engine import (
    EvaluationEngine,
    ModelResult,
    SamplePrediction,
    _measure_latency,
)
from src.models.base import BaseSentimentModel


# ---------------------------------------------------------------------------
# Minimal stub model — avoids any real ML dependency
# ---------------------------------------------------------------------------

class _FixedScoreModel(BaseSentimentModel):
    """Returns a constant score for every input. Deterministic and dependency-free."""

    def __init__(self, score: float = 0.75) -> None:
        self._score = float(score)

    def predict_score(self, text) -> float:
        return self._score

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        return [self._score] * len(texts)


class _VariableScoreModel(BaseSentimentModel):
    """Returns the i-th score from a pre-set list, cycling if exhausted."""

    def __init__(self, scores: List[float]) -> None:
        self._scores = list(scores)
        self._idx = 0

    def predict_score(self, text) -> float:
        s = self._scores[self._idx % len(self._scores)]
        self._idx += 1
        return s

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        return [self.predict_score(t) for t in texts]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEXTS_6 = ["good", "great", "bad", "terrible", "nice", "awful"]
LABELS_6 = [1, 1, 0, 0, 1, 0]
IDS_6 = list(range(6))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMeasureLatency(unittest.TestCase):
    """Test the latency measurement helper."""

    def test_returns_correct_number_of_scores_and_latencies(self):
        model = _FixedScoreModel(0.6)
        scores, latencies = _measure_latency(model, TEXTS_6)
        self.assertEqual(len(scores), 6)
        self.assertEqual(len(latencies), 6)

    def test_scores_match_model_output(self):
        model = _FixedScoreModel(0.9)
        scores, _ = _measure_latency(model, TEXTS_6)
        for s in scores:
            self.assertAlmostEqual(s, 0.9, places=9)

    def test_latencies_are_non_negative(self):
        model = _FixedScoreModel(0.5)
        _, latencies = _measure_latency(model, TEXTS_6)
        for lat in latencies:
            self.assertGreaterEqual(lat, 0.0)

    def test_empty_texts_returns_empty(self):
        model = _FixedScoreModel(0.5)
        scores, latencies = _measure_latency(model, [])
        self.assertEqual(scores, [])
        self.assertEqual(latencies, [])


class TestModelResult(unittest.TestCase):
    """Test ModelResult structure and serialization."""

    def _make_result(self) -> ModelResult:
        return ModelResult(
            model_name="test_model",
            total_samples=6,
            accuracy=0.8333,
            macro_f1=0.8,
            binary_f1=0.8571,
            precision=0.75,
            recall=1.0,
            roc_auc=0.9,
            confusion_matrix={"tn": 2, "fp": 1, "fn": 0, "tp": 3},
            mean_latency_ms=1.5,
            p50_latency_ms=1.2,
            p95_latency_ms=2.1,
        )

    def test_scalar_dict_has_required_keys(self):
        result = self._make_result()
        d = result.to_scalar_dict()
        for key in ["model", "accuracy", "macro_f1", "binary_f1", "precision",
                    "recall", "roc_auc", "tn", "fp", "fn", "tp",
                    "mean_latency_ms", "p50_latency_ms", "p95_latency_ms"]:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_scalar_dict_values_match(self):
        result = self._make_result()
        d = result.to_scalar_dict()
        self.assertEqual(d["model"], "test_model")
        self.assertEqual(d["accuracy"], 0.8333)
        self.assertEqual(d["tn"], 2)
        self.assertEqual(d["tp"], 3)
        self.assertAlmostEqual(d["mean_latency_ms"], 1.5, places=4)

    def test_roc_auc_none_serializes_as_empty_string(self):
        result = ModelResult(
            model_name="m",
            total_samples=4,
            accuracy=0.5,
            macro_f1=0.5,
            binary_f1=0.5,
            precision=0.5,
            recall=0.5,
            roc_auc=None,
            confusion_matrix={"tn": 1, "fp": 1, "fn": 1, "tp": 1},
            mean_latency_ms=0.5,
            p50_latency_ms=0.4,
            p95_latency_ms=0.9,
        )
        d = result.to_scalar_dict()
        self.assertEqual(d["roc_auc"], "")


class TestEvaluationEngineCore(unittest.TestCase):
    """Test EvaluationEngine.evaluate_single and .run."""

    def setUp(self):
        self.texts = TEXTS_6
        self.labels = LABELS_6
        self.ids = IDS_6

    # --- Registration ---

    def test_register_returns_self(self):
        engine = EvaluationEngine()
        model = _FixedScoreModel(0.9)
        ret = engine.register("m1", model)
        self.assertIs(ret, engine)

    def test_register_stores_model(self):
        engine = EvaluationEngine()
        engine.register("m1", _FixedScoreModel(0.9))
        self.assertIn("m1", engine.registered_models())

    def test_register_invalid_name_raises(self):
        engine = EvaluationEngine()
        with self.assertRaises(ValueError):
            engine.register("", _FixedScoreModel())

    def test_register_non_base_model_raises(self):
        engine = EvaluationEngine()
        with self.assertRaises(TypeError):
            engine.register("bad", object())  # type: ignore

    # --- evaluate_single ---

    def test_evaluate_single_returns_model_result(self):
        engine = EvaluationEngine(threshold=0.50)
        # All positive predictions (score = 0.9 >= 0.50 -> label 1)
        result = engine.evaluate_single(
            "always_positive",
            _FixedScoreModel(0.9),
            self.texts,
            self.labels,
            self.ids,
        )
        self.assertIsInstance(result, ModelResult)
        self.assertEqual(result.model_name, "always_positive")
        self.assertEqual(result.total_samples, 6)

    def test_evaluate_single_accuracy_all_positive(self):
        engine = EvaluationEngine(threshold=0.50)
        # Score 0.9 -> always predict Positive (1)
        # True labels: [1, 1, 0, 0, 1, 0] -> 3 correct, 3 wrong -> acc = 0.5
        result = engine.evaluate_single(
            "m", _FixedScoreModel(0.9), self.texts, self.labels, self.ids,
        )
        self.assertAlmostEqual(result.accuracy, 0.5, places=4)

    def test_evaluate_single_accuracy_all_negative(self):
        engine = EvaluationEngine(threshold=0.50)
        # Score 0.1 -> always predict Negative (0)
        # True labels: [1, 1, 0, 0, 1, 0] -> 3 correct, 3 wrong -> acc = 0.5
        result = engine.evaluate_single(
            "m", _FixedScoreModel(0.1), self.texts, self.labels, self.ids,
        )
        self.assertAlmostEqual(result.accuracy, 0.5, places=4)

    def test_evaluate_single_per_sample_predictions_count(self):
        engine = EvaluationEngine()
        result = engine.evaluate_single(
            "m", _FixedScoreModel(0.8), self.texts, self.labels, self.ids,
        )
        self.assertEqual(len(result.per_sample_predictions), 6)

    def test_evaluate_single_per_sample_ids_preserved(self):
        engine = EvaluationEngine()
        custom_ids = ["a", "b", "c", "d", "e", "f"]
        result = engine.evaluate_single(
            "m", _FixedScoreModel(0.8), self.texts, self.labels, custom_ids,
        )
        ids_in_result = [sp.sample_id for sp in result.per_sample_predictions]
        self.assertEqual(ids_in_result, custom_ids)

    def test_evaluate_single_default_ids_are_indices(self):
        engine = EvaluationEngine()
        result = engine.evaluate_single(
            "m", _FixedScoreModel(0.8), self.texts, self.labels,
        )
        ids_in_result = [sp.sample_id for sp in result.per_sample_predictions]
        self.assertEqual(ids_in_result, list(range(6)))

    def test_evaluate_single_latency_fields_populated(self):
        engine = EvaluationEngine()
        result = engine.evaluate_single(
            "m", _FixedScoreModel(0.7), self.texts, self.labels,
        )
        self.assertGreaterEqual(result.mean_latency_ms, 0.0)
        self.assertGreaterEqual(result.p50_latency_ms, 0.0)
        self.assertGreaterEqual(result.p95_latency_ms, 0.0)

    def test_evaluate_single_empty_texts_raises(self):
        engine = EvaluationEngine()
        with self.assertRaises(ValueError):
            engine.evaluate_single("m", _FixedScoreModel(), [], [])

    def test_evaluate_single_mismatched_lengths_raises(self):
        engine = EvaluationEngine()
        with self.assertRaises(ValueError):
            engine.evaluate_single("m", _FixedScoreModel(), ["a", "b"], [1])

    def test_evaluate_single_threshold_boundary_exactly_05(self):
        engine = EvaluationEngine(threshold=0.50)
        # Score == threshold -> Positive (>=)
        result = engine.evaluate_single(
            "m", _FixedScoreModel(0.50), self.texts, self.labels,
        )
        for sp in result.per_sample_predictions:
            self.assertEqual(sp.predicted_label, 1)

    def test_evaluate_single_threshold_boundary_just_below(self):
        engine = EvaluationEngine(threshold=0.50)
        result = engine.evaluate_single(
            "m", _FixedScoreModel(0.4999), self.texts, self.labels,
        )
        for sp in result.per_sample_predictions:
            self.assertEqual(sp.predicted_label, 0)

    # --- run ---

    def test_run_no_models_raises(self):
        engine = EvaluationEngine()
        with self.assertRaises(RuntimeError):
            engine.run(self.texts, self.labels)

    def test_run_returns_one_result_per_model(self):
        engine = EvaluationEngine()
        engine.register("m1", _FixedScoreModel(0.9))
        engine.register("m2", _FixedScoreModel(0.1))
        results = engine.run(self.texts, self.labels)
        self.assertEqual(len(results), 2)

    def test_run_result_model_names_match_registration(self):
        engine = EvaluationEngine()
        engine.register("vader_stub", _FixedScoreModel(0.9))
        engine.register("lr_stub", _FixedScoreModel(0.2))
        results = engine.run(self.texts, self.labels)
        names = {r.model_name for r in results}
        self.assertIn("vader_stub", names)
        self.assertIn("lr_stub", names)

    def test_run_with_sample_ids(self):
        engine = EvaluationEngine()
        engine.register("m", _FixedScoreModel(0.9))
        results = engine.run(self.texts, self.labels, sample_ids=self.ids)
        self.assertEqual(len(results[0].per_sample_predictions), 6)


class TestEvaluationEngineArtifacts(unittest.TestCase):
    """Test that all four output artifacts are written correctly."""

    def setUp(self):
        self.texts = TEXTS_6
        self.labels = LABELS_6
        self.ids = IDS_6
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run_and_save(self, model_name="model_A", score=0.8):
        engine = EvaluationEngine(
            output_dir=self.output_dir,
            threshold=0.50,
            experiment_name="test_exp",
        )
        engine.register(model_name, _FixedScoreModel(score))
        results = engine.run(self.texts, self.labels, sample_ids=self.ids)
        paths = engine.save_all(results, extra_config={"run": "test"})
        return results, paths

    # --- metrics.csv ---

    def test_metrics_csv_exists(self):
        _, paths = self._run_and_save()
        self.assertTrue(paths["metrics_csv"].exists())

    def test_metrics_csv_has_header_row(self):
        _, paths = self._run_and_save()
        with open(paths["metrics_csv"], newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
        for col in ["model", "accuracy", "macro_f1", "binary_f1", "precision", "recall"]:
            self.assertIn(col, headers, f"Missing column: {col}")

    def test_metrics_csv_has_one_data_row_per_model(self):
        engine = EvaluationEngine(output_dir=self.output_dir)
        engine.register("m1", _FixedScoreModel(0.9))
        engine.register("m2", _FixedScoreModel(0.1))
        results = engine.run(self.texts, self.labels)
        paths = engine.save_all(results)
        with open(paths["metrics_csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)

    def test_metrics_csv_accuracy_is_correct(self):
        _, paths = self._run_and_save(score=0.9)
        with open(paths["metrics_csv"], newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        # Score 0.9 -> always predict Positive -> acc = 3/6 = 0.5
        self.assertAlmostEqual(float(row["accuracy"]), 0.5, places=3)

    # --- predictions.csv ---

    def test_predictions_csv_exists(self):
        _, paths = self._run_and_save()
        self.assertTrue(paths["predictions_csv"].exists())

    def test_predictions_csv_row_count(self):
        _, paths = self._run_and_save()
        with open(paths["predictions_csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # 6 texts × 1 model = 6 rows
        self.assertEqual(len(rows), 6)

    def test_predictions_csv_required_columns(self):
        _, paths = self._run_and_save()
        with open(paths["predictions_csv"], newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
        for col in ["model", "sample_id", "true_label", "predicted_label", "positive_score", "text"]:
            self.assertIn(col, headers)

    def test_predictions_csv_sample_ids_are_correct(self):
        _, paths = self._run_and_save()
        with open(paths["predictions_csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids_in_file = [int(r["sample_id"]) for r in rows]
        self.assertEqual(ids_in_file, list(range(6)))

    def test_predictions_csv_multi_model(self):
        engine = EvaluationEngine(output_dir=self.output_dir)
        engine.register("m1", _FixedScoreModel(0.9))
        engine.register("m2", _FixedScoreModel(0.1))
        results = engine.run(self.texts, self.labels)
        paths = engine.save_all(results)
        with open(paths["predictions_csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # 6 texts × 2 models = 12 rows
        self.assertEqual(len(rows), 12)

    # --- confusion matrices ---

    def test_confusion_dir_exists(self):
        _, paths = self._run_and_save()
        self.assertTrue(paths["confusion_dir"].is_dir())

    def test_confusion_csv_created_for_each_model(self):
        engine = EvaluationEngine(output_dir=self.output_dir)
        engine.register("model_X", _FixedScoreModel(0.9))
        engine.register("model_Y", _FixedScoreModel(0.1))
        results = engine.run(self.texts, self.labels)
        paths = engine.save_all(results)
        cm_files = list(paths["confusion_dir"].glob("*.csv"))
        self.assertEqual(len(cm_files), 2)

    def test_confusion_csv_structure(self):
        _, paths = self._run_and_save(model_name="model_A")
        cm_file = paths["confusion_dir"] / "model_A.csv"
        self.assertTrue(cm_file.exists())
        with open(cm_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # 3 rows: header + 2 data rows
        self.assertEqual(len(rows), 3)
        # Header should have 3 columns
        self.assertEqual(len(rows[0]), 3)

    def test_confusion_csv_values_sum_to_n_samples(self):
        _, paths = self._run_and_save(model_name="model_A", score=0.9)
        cm_file = paths["confusion_dir"] / "model_A.csv"
        with open(cm_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # rows[1] = ["Actual_Negative", tn, fp]
        # rows[2] = ["Actual_Positive", fn, tp]
        tn, fp = int(rows[1][1]), int(rows[1][2])
        fn, tp = int(rows[2][1]), int(rows[2][2])
        self.assertEqual(tn + fp + fn + tp, 6)

    # --- experiment_summary.json ---

    def test_summary_json_exists(self):
        _, paths = self._run_and_save()
        self.assertTrue(paths["summary_json"].exists())

    def test_summary_json_structure(self):
        _, paths = self._run_and_save()
        with open(paths["summary_json"], encoding="utf-8") as f:
            summary = json.load(f)
        for key in ["experiment_name", "threshold", "num_models", "config", "models"]:
            self.assertIn(key, summary)

    def test_summary_json_experiment_name(self):
        _, paths = self._run_and_save()
        with open(paths["summary_json"], encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["experiment_name"], "test_exp")

    def test_summary_json_models_list_length(self):
        _, paths = self._run_and_save()
        with open(paths["summary_json"], encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["num_models"], 1)
        self.assertEqual(len(summary["models"]), 1)

    def test_summary_json_model_entry_keys(self):
        _, paths = self._run_and_save()
        with open(paths["summary_json"], encoding="utf-8") as f:
            summary = json.load(f)
        entry = summary["models"][0]
        for key in ["model_name", "total_samples", "accuracy", "macro_f1",
                    "binary_f1", "precision", "recall", "roc_auc",
                    "confusion_matrix", "latency"]:
            self.assertIn(key, entry, f"Missing key: {key}")

    def test_summary_json_latency_subkeys(self):
        _, paths = self._run_and_save()
        with open(paths["summary_json"], encoding="utf-8") as f:
            summary = json.load(f)
        lat = summary["models"][0]["latency"]
        for key in ["mean_ms", "p50_ms", "p95_ms"]:
            self.assertIn(key, lat)

    def test_summary_json_extra_config_stored(self):
        engine = EvaluationEngine(output_dir=self.output_dir, experiment_name="cfg_test")
        engine.register("m", _FixedScoreModel(0.8))
        results = engine.run(self.texts, self.labels)
        paths = engine.save_all(results, extra_config={"custom_key": "custom_value"})
        with open(paths["summary_json"], encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["config"].get("custom_key"), "custom_value")

    def test_summary_json_accuracy_is_not_hardcoded(self):
        """Verify metrics are derived from actual predictions, not hardcoded."""
        _, paths = self._run_and_save(score=0.9)
        with open(paths["summary_json"], encoding="utf-8") as f:
            summary = json.load(f)
        acc = summary["models"][0]["accuracy"]
        # score=0.9 -> always predict 1 -> acc = 3/6 = 0.5
        self.assertAlmostEqual(acc, 0.5, places=3)


class TestEvaluationEngineEdgeCases(unittest.TestCase):
    """Edge cases for robustness."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_single_sample(self):
        engine = EvaluationEngine(output_dir=self.output_dir)
        engine.register("m", _FixedScoreModel(0.9))
        results = engine.run(["hello"], [1])
        self.assertEqual(results[0].total_samples, 1)
        self.assertAlmostEqual(results[0].accuracy, 1.0, places=4)

    def test_spaces_in_model_name_sanitized_in_cm_filename(self):
        engine = EvaluationEngine(output_dir=self.output_dir)
        engine.register("Model With Spaces", _FixedScoreModel(0.8))
        results = engine.run(TEXTS_6, LABELS_6)
        engine.save_confusion_matrices(results)
        cm_file = self.output_dir / "confusion_matrices" / "Model_With_Spaces.csv"
        self.assertTrue(cm_file.exists())

    def test_all_true_labels_same_class_roc_auc_handled(self):
        """ROC AUC should be None (or absent) when only one class is present."""
        engine = EvaluationEngine(output_dir=self.output_dir)
        engine.register("m", _FixedScoreModel(0.9))
        all_positive_labels = [1, 1, 1, 1]
        results = engine.run(["a", "b", "c", "d"], all_positive_labels)
        # roc_auc should be None (cannot compute with single class)
        self.assertIsNone(results[0].roc_auc)

    def test_threshold_zero_always_predicts_positive(self):
        engine = EvaluationEngine(threshold=0.0)
        engine.register("m", _FixedScoreModel(0.0))
        results = engine.run(TEXTS_6, LABELS_6)
        for sp in results[0].per_sample_predictions:
            self.assertEqual(sp.predicted_label, 1)

    def test_threshold_one_only_score_one_predicts_positive(self):
        engine = EvaluationEngine(threshold=1.0)
        engine.register("below", _FixedScoreModel(0.999))
        results = engine.run(TEXTS_6, LABELS_6)
        for sp in results[0].per_sample_predictions:
            self.assertEqual(sp.predicted_label, 0)


if __name__ == "__main__":
    unittest.main()
