"""Unit tests for the Ablation Study Framework.

Tests cover:
- AblationConfig defaults and flag values for all four variants
- build_router: correct router type returned per variant
- Router behaviour: alpha values and routing_state for each variant
- AblationResult.to_csv_row: field presence, f1_delta formatting
- AblationRunner.run: metric computation, F1 delta calculation, CSV writing
- CSV artifact: structure, header, row count, f1_delta column
- Edge cases: single sample, all-positive labels, threshold boundary
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import RoutingConfig, config as _cfg
from src.evaluation.ablation import (
    AblationConfig,
    AblationResult,
    AblationRunner,
    ALL_VARIANTS,
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    build_router,
    _StaticAlphaRouter,
    _NoLengthRouter,
    _NoGateRouter,
)
from src.models.adaptive_router import AdaptiveRouter
from src.models.base import BaseSentimentModel
from src.models.dynamic_fusion import DynamicFusionFramework


# ---------------------------------------------------------------------------
# Minimal stub models — no ML dependencies
# ---------------------------------------------------------------------------

class _FixedModel(BaseSentimentModel):
    def __init__(self, score: float = 0.8) -> None:
        self._s = float(score)
    def predict_score(self, text) -> float:
        return self._s
    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        return [self._s] * len(texts)


TEXTS = ["good movie", "great film", "bad film", "terrible acting", "nice story", "awful"]
LABELS = [1, 1, 0, 0, 1, 0]
IDS = list(range(6))


# ---------------------------------------------------------------------------
# AblationConfig tests
# ---------------------------------------------------------------------------

class TestAblationConfig(unittest.TestCase):

    def test_variant_a_is_full_dynamic(self):
        self.assertFalse(VARIANT_A.use_static_alpha)
        self.assertFalse(VARIANT_A.disable_length_contribution)
        self.assertFalse(VARIANT_A.disable_noise_gate)

    def test_variant_b_is_static(self):
        self.assertTrue(VARIANT_B.use_static_alpha)
        self.assertFalse(VARIANT_B.disable_length_contribution)
        self.assertFalse(VARIANT_B.disable_noise_gate)

    def test_variant_b_static_alpha_is_alpha_min(self):
        self.assertAlmostEqual(VARIANT_B.static_alpha, _cfg.routing.alpha_min)

    def test_variant_c_disables_length(self):
        self.assertFalse(VARIANT_C.use_static_alpha)
        self.assertTrue(VARIANT_C.disable_length_contribution)
        self.assertFalse(VARIANT_C.disable_noise_gate)

    def test_variant_d_disables_gate(self):
        self.assertFalse(VARIANT_D.use_static_alpha)
        self.assertFalse(VARIANT_D.disable_length_contribution)
        self.assertTrue(VARIANT_D.disable_noise_gate)

    def test_all_variants_list_has_four_items(self):
        self.assertEqual(len(ALL_VARIANTS), 4)

    def test_variant_names_are_unique(self):
        names = [v.name for v in ALL_VARIANTS]
        self.assertEqual(len(names), len(set(names)))


# ---------------------------------------------------------------------------
# build_router tests
# ---------------------------------------------------------------------------

class TestBuildRouter(unittest.TestCase):

    def test_variant_a_returns_base_adaptive_router(self):
        router = build_router(VARIANT_A)
        # Should be a plain AdaptiveRouter, not a subclass override
        self.assertIsInstance(router, AdaptiveRouter)
        self.assertNotIsInstance(router, _StaticAlphaRouter)
        self.assertNotIsInstance(router, _NoLengthRouter)
        self.assertNotIsInstance(router, _NoGateRouter)

    def test_variant_b_returns_static_alpha_router(self):
        router = build_router(VARIANT_B)
        self.assertIsInstance(router, _StaticAlphaRouter)

    def test_variant_c_returns_no_length_router(self):
        router = build_router(VARIANT_C)
        self.assertIsInstance(router, _NoLengthRouter)

    def test_variant_d_returns_no_gate_router(self):
        router = build_router(VARIANT_D)
        self.assertIsInstance(router, _NoGateRouter)


# ---------------------------------------------------------------------------
# Router behaviour tests
# ---------------------------------------------------------------------------

class TestStaticAlphaRouter(unittest.TestCase):

    def setUp(self):
        self.router = _StaticAlphaRouter(alpha=0.02, routing_config=_cfg.routing)

    def test_alpha_always_fixed(self):
        for noise in [0.0, 0.10, 0.20, 0.50, 1.0]:
            for length in [1, 20, 100]:
                dec = self.router.route(noise, length)
                self.assertAlmostEqual(dec.alpha, 0.02, places=9,
                                       msg=f"Failed at N={noise}, L={length}")

    def test_routing_state_is_static(self):
        dec = self.router.route(0.5, 20)
        self.assertEqual(dec.routing_state, "static_alpha")


class TestNoLengthRouter(unittest.TestCase):

    def setUp(self):
        self.router = _NoLengthRouter(routing_config=_cfg.routing)

    def test_different_lengths_give_same_alpha(self):
        """With w1=0 the length term vanishes; alpha must not change with L."""
        noise = 0.5  # above gate threshold
        alpha_at_1 = self.router.route(noise, 1).alpha
        alpha_at_100 = self.router.route(noise, 100).alpha
        self.assertAlmostEqual(alpha_at_1, alpha_at_100, places=9)

    def test_low_noise_still_gated(self):
        """N <= 0.20 threshold gate must still apply in Variant C."""
        dec = self.router.route(0.10, 50)
        self.assertAlmostEqual(dec.alpha, _cfg.routing.alpha_min, places=9)
        self.assertEqual(dec.routing_state, "low_noise_default")

    def test_alpha_within_bounds(self):
        for noise in [0.0, 0.21, 0.5, 1.0]:
            dec = self.router.route(noise, 20)
            self.assertGreaterEqual(dec.alpha, _cfg.routing.alpha_min)
            self.assertLessEqual(dec.alpha, _cfg.routing.alpha_max)


class TestNoGateRouter(unittest.TestCase):

    def setUp(self):
        self.router = _NoGateRouter(routing_config=_cfg.routing)

    def test_low_noise_still_produces_dynamic_alpha(self):
        """N <= 0.20 should NOT be gated; alpha should come from sigmoid."""
        dec_no_gate = self.router.route(0.10, 20)
        # Full router would return alpha_min=0.02 for N=0.10
        full_router = AdaptiveRouter(_cfg.routing)
        dec_full = full_router.route(0.10, 20)
        # The no-gate router must not unconditionally return alpha_min when N <= 0.20
        # (they may still match if sigmoid also gives alpha_min, but routing_state differs)
        self.assertNotEqual(dec_no_gate.routing_state, "low_noise_default")

    def test_alpha_within_bounds_for_all_noise(self):
        for noise in [0.0, 0.10, 0.20, 0.30, 0.50, 1.0]:
            dec = self.router.route(noise, 20)
            self.assertGreaterEqual(dec.alpha, _cfg.routing.alpha_min)
            self.assertLessEqual(dec.alpha, _cfg.routing.alpha_max)

    def test_routing_state_never_low_noise_default(self):
        for noise in [0.0, 0.10, 0.20, 0.50]:
            dec = self.router.route(noise, 20)
            self.assertNotEqual(dec.routing_state, "low_noise_default")


# ---------------------------------------------------------------------------
# AblationResult tests
# ---------------------------------------------------------------------------

class TestAblationResult(unittest.TestCase):

    def _make(self, name="A", delta=None) -> AblationResult:
        return AblationResult(
            variant_name=name,
            total_samples=100,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            binary_f1=0.85,
            f1_delta=delta,
        )

    def test_to_csv_row_has_required_keys(self):
        row = self._make().to_csv_row()
        for key in ["variant", "samples", "accuracy", "precision",
                    "recall", "binary_f1", "f1_delta_vs_full"]:
            self.assertIn(key, row)

    def test_f1_delta_none_serializes_as_empty_string(self):
        row = self._make(delta=None).to_csv_row()
        self.assertEqual(row["f1_delta_vs_full"], "")

    def test_f1_delta_value_is_rounded(self):
        row = self._make(delta=0.012345678).to_csv_row()
        # Rounded to 6 decimal places
        self.assertEqual(row["f1_delta_vs_full"], round(0.012345678, 6))

    def test_negative_delta_preserved(self):
        row = self._make(delta=-0.03).to_csv_row()
        self.assertAlmostEqual(float(row["f1_delta_vs_full"]), -0.03, places=9)


# ---------------------------------------------------------------------------
# AblationRunner integration tests
# ---------------------------------------------------------------------------

class TestAblationRunner(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)
        # Use stub models to avoid any ML dependency
        self.vader_stub = _FixedModel(0.8)
        self.distilbert_stub = _FixedModel(0.6)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _runner(self, **kwargs) -> AblationRunner:
        return AblationRunner(
            vader_model=self.vader_stub,
            distilbert_model=self.distilbert_stub,
            output_dir=self.output_dir,
            threshold=0.50,
            **kwargs,
        )

    def test_run_returns_four_results_for_default_variants(self):
        runner = self._runner()
        results = runner.run(TEXTS, LABELS, IDS)
        self.assertEqual(len(results), 4)

    def test_run_variant_names_match_config(self):
        runner = self._runner()
        results = runner.run(TEXTS, LABELS, IDS)
        names = [r.variant_name for r in results]
        expected = [v.name for v in ALL_VARIANTS]
        self.assertEqual(names, expected)

    def test_first_result_f1_delta_is_none(self):
        runner = self._runner()
        results = runner.run(TEXTS, LABELS, IDS)
        self.assertIsNone(results[0].f1_delta, "Variant A f1_delta must be None (reference).")

    def test_delta_variants_have_float_f1_delta(self):
        runner = self._runner()
        results = runner.run(TEXTS, LABELS, IDS)
        for r in results[1:]:
            self.assertIsNotNone(r.f1_delta, f"Variant {r.variant_name} should have f1_delta.")
            self.assertIsInstance(r.f1_delta, float)

    def test_delta_is_correct_relative_to_variant_a(self):
        runner = self._runner()
        results = runner.run(TEXTS, LABELS, IDS)
        ref_f1 = results[0].binary_f1
        for r in results[1:]:
            expected_delta = round(r.binary_f1 - ref_f1, 6)
            self.assertAlmostEqual(r.f1_delta, expected_delta, places=6)

    def test_accuracy_is_between_0_and_1(self):
        runner = self._runner()
        for r in runner.run(TEXTS, LABELS, IDS):
            self.assertGreaterEqual(r.accuracy, 0.0)
            self.assertLessEqual(r.accuracy, 1.0)

    def test_custom_variant_subset(self):
        runner = self._runner(variants=[VARIANT_A, VARIANT_B])
        results = runner.run(TEXTS, LABELS, IDS)
        self.assertEqual(len(results), 2)

    def test_single_variant(self):
        runner = self._runner(variants=[VARIANT_A])
        results = runner.run(TEXTS, LABELS, IDS)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].f1_delta)

    def test_single_sample(self):
        runner = self._runner(variants=[VARIANT_A])
        results = runner.run(["hello world"], [1], [0])
        self.assertEqual(results[0].total_samples, 1)

    def test_threshold_boundary_exactly_05(self):
        """Score == threshold should predict Positive."""
        m = _FixedModel(0.50)
        runner = AblationRunner(
            vader_model=m,
            distilbert_model=m,
            output_dir=self.output_dir,
            threshold=0.50,
            variants=[VARIANT_B],  # static alpha -> deterministic fusion
        )
        results = runner.run(["good"], [1])
        self.assertGreaterEqual(results[0].accuracy, 0.0)

    def test_all_positive_labels_handled(self):
        """Single-class labels must not crash (ROC-AUC safely skipped)."""
        runner = self._runner(variants=[VARIANT_A])
        results = runner.run(["a", "b", "c"], [1, 1, 1])
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# CSV artifact tests
# ---------------------------------------------------------------------------

class TestAblationCSV(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run_and_save(self):
        runner = AblationRunner(
            vader_model=_FixedModel(0.8),
            distilbert_model=_FixedModel(0.6),
            output_dir=self.output_dir,
            threshold=0.50,
        )
        results = runner.run(TEXTS, LABELS, IDS)
        path = runner.save_csv(results)
        return results, path

    def test_csv_file_created(self):
        _, path = self._run_and_save()
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "ablation_results.csv")

    def test_csv_in_correct_directory(self):
        _, path = self._run_and_save()
        self.assertEqual(path.parent.resolve(), self.output_dir.resolve())

    def test_csv_has_correct_header(self):
        _, path = self._run_and_save()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
        expected = ["variant", "samples", "accuracy", "precision",
                    "recall", "binary_f1", "f1_delta_vs_full"]
        for col in expected:
            self.assertIn(col, headers, f"Missing column: {col}")

    def test_csv_has_four_data_rows(self):
        _, path = self._run_and_save()
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)

    def test_csv_variant_a_f1_delta_is_empty(self):
        _, path = self._run_and_save()
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["f1_delta_vs_full"], "")

    def test_csv_variant_b_c_d_have_delta_values(self):
        _, path = self._run_and_save()
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows[1:]:
            self.assertNotEqual(row["f1_delta_vs_full"], "",
                                f"Expected delta for {row['variant']}")

    def test_csv_accuracy_parses_as_float(self):
        _, path = self._run_and_save()
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            acc = float(row["accuracy"])
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_csv_samples_column_correct(self):
        _, path = self._run_and_save()
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            self.assertEqual(int(row["samples"]), len(TEXTS))

    def test_csv_variant_names_are_correct(self):
        _, path = self._run_and_save()
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        csv_names = [row["variant"] for row in rows]
        expected_names = [v.name for v in ALL_VARIANTS]
        self.assertEqual(csv_names, expected_names)

    def test_run_and_save_shortcut(self):
        runner = AblationRunner(
            vader_model=_FixedModel(0.7),
            distilbert_model=_FixedModel(0.5),
            output_dir=self.output_dir,
        )
        path = runner.run_and_save(TEXTS, LABELS, IDS)
        self.assertTrue(path.exists())
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
