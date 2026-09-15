"""Focused test suite for SentiMix 3-class methodology adaptation.

Verifies all 14 requirements specified in the methodology adaptation task:
1. Class order consistency across all modules.
2. SentiMix label parsing.
3. VADER3ClassAdapter (deterministic, non-learned, no fitted parameters).
4. Probability/score vector shape and simplex property.
5. Vector fusion on Delta^2.
6. Alpha dynamic weighting contribution.
7. Argmax prediction resolution.
8. Neutral preservation (never dropped or silently mapped).
9. Existing binary pipeline unchanged.
10. Existing noise formulas unchanged.
11. Existing adaptive router unchanged.
12. Static fusion ablation.
13. DistilBERT-only prediction ablation.
14. VADER-only prediction ablation.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.data.sentimix_loader import (
    SENTIMIX_LABEL_STR2INT,
    SENTIMIX_LABEL_INT2STR,
    LABEL_POSITIVE as DATA_POS,
    LABEL_NEGATIVE as DATA_NEG,
    LABEL_NEUTRAL as DATA_NEU,
    SentimixDataLoader,
    SentimixCoNLLParser,
)
from src.models.vader_3class import (
    VADER3ClassAdapter,
    Vader3ClassOutput,
    LABEL_POSITIVE as VADER_POS,
    LABEL_NEGATIVE as VADER_NEG,
    LABEL_NEUTRAL as VADER_NEU,
)
from src.models.distilbert_3class import (
    DistilBert3ClassModel,
    DistilBert3ClassOutput,
    LABEL_POSITIVE as BERT_POS,
    LABEL_NEGATIVE as BERT_NEG,
    LABEL_NEUTRAL as BERT_NEU,
)
from src.models.fusion_3class import (
    fuse_3class_vectors,
    DynamicFusion3ClassFramework,
    Fusion3ClassResult,
    LABEL_POSITIVE as FUSION_POS,
    LABEL_NEGATIVE as FUSION_NEG,
    LABEL_NEUTRAL as FUSION_NEU,
)
from src.evaluation.metrics_3class import (
    compute_3class_metrics,
    compute_binary_subset_metrics,
    LABEL_POSITIVE as METRIC_POS,
    LABEL_NEGATIVE as METRIC_NEG,
    LABEL_NEUTRAL as METRIC_NEU,
)


# ---------------------------------------------------------------------------
# 1. Class Order Consistency
# ---------------------------------------------------------------------------

class TestClassOrderConsistency:
    """Verify that positive=0, negative=1, neutral=2 everywhere across modules."""

    def test_canonical_integer_mapping(self):
        assert DATA_POS == 0 and DATA_NEG == 1 and DATA_NEU == 2
        assert VADER_POS == 0 and VADER_NEG == 1 and VADER_NEU == 2
        assert BERT_POS == 0 and BERT_NEG == 1 and BERT_NEU == 2
        assert FUSION_POS == 0 and FUSION_NEG == 1 and FUSION_NEU == 2
        assert METRIC_POS == 0 and METRIC_NEG == 1 and METRIC_NEU == 2

    def test_string_to_int_mapping(self):
        assert SENTIMIX_LABEL_STR2INT["positive"] == 0
        assert SENTIMIX_LABEL_STR2INT["negative"] == 1
        assert SENTIMIX_LABEL_STR2INT["neutral"] == 2

    def test_int_to_string_mapping(self):
        assert SENTIMIX_LABEL_INT2STR[0] == "positive"
        assert SENTIMIX_LABEL_INT2STR[1] == "negative"
        assert SENTIMIX_LABEL_INT2STR[2] == "neutral"


# ---------------------------------------------------------------------------
# 2. SentiMix Label Parsing
# ---------------------------------------------------------------------------

class TestSentiMixLabelParsing:
    def test_parser_parses_three_distinct_labels(self, tmp_path):
        conll_content = (
            "meta\t101\tpositive\n"
            "great\tEng\n"
            "film\tEng\n\n"
            "meta\t102\tnegative\n"
            "worst\tEng\n"
            "acting\tEng\n\n"
            "meta\t103\tneutral\n"
            "kal\tHin\n"
            "milte\tHin\n"
            "hain\tHin\n\n"
        )
        p = tmp_path / "sample.conll"
        p.write_text(conll_content, encoding="utf-8")

        parser = SentimixCoNLLParser()
        samples = parser.parse_file(p)
        assert len(samples) == 3
        assert samples[0].label == 0 and samples[0].label_str == "positive"
        assert samples[1].label == 1 and samples[1].label_str == "negative"
        assert samples[2].label == 2 and samples[2].label_str == "neutral"


# ---------------------------------------------------------------------------
# 3. VADER3ClassAdapter
# ---------------------------------------------------------------------------

class TestVADER3ClassAdapter:
    @pytest.fixture
    def adapter(self):
        return VADER3ClassAdapter()

    def test_deterministic_output(self, adapter):
        res1 = adapter.predict("ye film bohot achhi hai!")
        res2 = adapter.predict("ye film bohot achhi hai!")
        assert res1.scores_vector == res2.scores_vector
        assert res1.predicted_label == res2.predicted_label

    def test_scores_sum_to_one(self, adapter):
        texts = [
            "excellent movie loved it!",
            "terrible waste of time horrible",
            "today is Wednesday",
            "kya baat hai yaar",
            "",
            "   ",
        ]
        for t in texts:
            out = adapter.predict(t)
            assert sum(out.scores_vector) == pytest.approx(1.0, rel=1e-5)
            assert 0.0 <= out.positive_prob <= 1.0
            assert 0.0 <= out.negative_prob <= 1.0
            assert 0.0 <= out.neutral_prob <= 1.0

    def test_neutral_fallback_on_empty(self, adapter):
        out = adapter.predict("")
        assert out.predicted_label == 2
        assert out.neutral_prob == 1.0
        assert out.positive_prob == 0.0
        assert out.negative_prob == 0.0

    def test_positive_text_detection(self, adapter):
        out = adapter.predict("This is great and wonderful and amazing!")
        assert out.positive_prob > out.negative_prob
        assert out.predicted_label == 0


# ---------------------------------------------------------------------------
# 4. Probability / Score Vector Shape and Simplex Property
# ---------------------------------------------------------------------------

class TestVectorShapeAndSimplex:
    def test_vader_vector_shape(self):
        adapter = VADER3ClassAdapter()
        vec = adapter.predict_scores("test text")
        assert len(vec) == 3
        assert isinstance(vec, list)
        assert sum(vec) == pytest.approx(1.0, rel=1e-5)

    def test_fusion_vector_shape(self):
        res = fuse_3class_vectors([0.7, 0.1, 0.2], [0.8, 0.1, 0.1], alpha=0.10)
        assert len(res.p_final) == 3
        assert sum(res.p_final) == pytest.approx(1.0, rel=1e-5)
        assert 0.0 <= res.confidence <= 1.0


# ---------------------------------------------------------------------------
# 5. Vector Fusion on Delta^2
# ---------------------------------------------------------------------------

class TestVectorFusionMath:
    def test_exact_convex_combination(self):
        p_v = [0.6, 0.2, 0.2]
        p_d = [0.1, 0.8, 0.1]
        alpha = 0.20  # 20% VADER, 80% DistilBERT

        # Expected:
        # pos: 0.20*0.6 + 0.80*0.1 = 0.12 + 0.08 = 0.20
        # neg: 0.20*0.2 + 0.80*0.8 = 0.04 + 0.64 = 0.68
        # neu: 0.20*0.2 + 0.80*0.1 = 0.04 + 0.08 = 0.12
        res = fuse_3class_vectors(p_v, p_d, alpha=alpha)
        assert res.p_final[0] == pytest.approx(0.20, abs=1e-4)
        assert res.p_final[1] == pytest.approx(0.68, abs=1e-4)
        assert res.p_final[2] == pytest.approx(0.12, abs=1e-4)
        assert res.predicted_label == 1  # Negative is argmax


# ---------------------------------------------------------------------------
# 6. Alpha Contribution
# ---------------------------------------------------------------------------

class TestAlphaContribution:
    def test_alpha_zero_gives_pure_distilbert(self):
        p_v = [1.0, 0.0, 0.0]
        p_d = [0.0, 1.0, 0.0]
        res = fuse_3class_vectors(p_v, p_d, alpha=0.0)
        assert res.p_final[0] == pytest.approx(0.0)
        assert res.p_final[1] == pytest.approx(1.0)
        assert res.p_final[2] == pytest.approx(0.0)

    def test_alpha_one_gives_pure_vader(self):
        p_v = [1.0, 0.0, 0.0]
        p_d = [0.0, 1.0, 0.0]
        res = fuse_3class_vectors(p_v, p_d, alpha=1.0)
        assert res.p_final[0] == pytest.approx(1.0)
        assert res.p_final[1] == pytest.approx(0.0)
        assert res.p_final[2] == pytest.approx(0.0)

    def test_higher_alpha_increases_vader_influence(self):
        p_v = [0.9, 0.05, 0.05]
        p_d = [0.1, 0.80, 0.10]
        res_low = fuse_3class_vectors(p_v, p_d, alpha=0.02)
        res_high = fuse_3class_vectors(p_v, p_d, alpha=0.25)
        assert res_high.p_final[0] > res_low.p_final[0]


# ---------------------------------------------------------------------------
# 7. Argmax Prediction Resolution
# ---------------------------------------------------------------------------

class TestArgmaxPrediction:
    def test_argmax_positive(self):
        res = fuse_3class_vectors([0.8, 0.1, 0.1], [0.7, 0.2, 0.1], alpha=0.1)
        assert res.predicted_label == 0
        assert res.sentiment_label == "positive"

    def test_argmax_negative(self):
        res = fuse_3class_vectors([0.1, 0.8, 0.1], [0.1, 0.7, 0.2], alpha=0.1)
        assert res.predicted_label == 1
        assert res.sentiment_label == "negative"

    def test_argmax_neutral(self):
        res = fuse_3class_vectors([0.1, 0.1, 0.8], [0.1, 0.1, 0.8], alpha=0.1)
        assert res.predicted_label == 2
        assert res.sentiment_label == "neutral"


# ---------------------------------------------------------------------------
# 8. Neutral Preservation
# ---------------------------------------------------------------------------

class TestNeutralPreservation:
    def test_neutral_is_retained_in_metrics(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 2]
        metrics = compute_3class_metrics(y_true, y_pred)
        assert "neutral" in metrics["per_class"]
        assert metrics["per_class"]["neutral"]["support"] == 2
        assert len(metrics["confusion_matrix"]["matrix"]) == 3

    def test_secondary_binary_subset_filters_neutral(self):
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 1, 1]
        sec_res = compute_binary_subset_metrics(y_true, y_pred)
        assert sec_res["is_secondary_analysis"] is True
        # Only samples where both true and pred are non-neutral (0 or 1)
        assert sec_res["total_samples"] == 4


# ---------------------------------------------------------------------------
# 9. Existing Binary Pipeline Unchanged
# ---------------------------------------------------------------------------

class TestExistingBinaryPipelineUnchanged:
    def test_fuse_scores_still_binary(self):
        from src.models.dynamic_fusion import fuse_scores
        res = fuse_scores(s_vader=0.8, s_distilbert=0.9, alpha=0.10)
        assert hasattr(res, "prediction")
        assert res.prediction in (0, 1)
        assert res.sentiment_label in ("Positive", "Negative")

    def test_binary_vader_model_unchanged(self):
        from src.models.vader_model import VaderSentimentModel
        vm = VaderSentimentModel()
        out = vm.predict("good movie")
        assert hasattr(out, "positive_prob")
        assert hasattr(out, "negative_prob")
        assert not hasattr(out, "neutral_prob")


# ---------------------------------------------------------------------------
# 10. Existing Noise Formulas Unchanged
# ---------------------------------------------------------------------------

class TestExistingNoiseFormulasUnchanged:
    def test_weights_and_constants(self):
        from configs.config import config
        assert config.noise.w_emoji == 0.25
        assert config.noise.w_repetition == 0.25
        assert config.noise.w_code_mix == 0.30
        assert config.noise.w_symbol == 0.20

    def test_composite_formula_calculation(self):
        from src.features.noise_quantifier import calculate_noise_index
        n = calculate_noise_index(0.2, 0.4, 0.1, 0.5)
        # N = 0.25*0.2 + 0.25*0.4 + 0.30*0.1 + 0.20*0.5 = 0.05 + 0.10 + 0.03 + 0.10 = 0.28
        assert n == pytest.approx(0.28)


# ---------------------------------------------------------------------------
# 11. Existing Adaptive Router Unchanged
# ---------------------------------------------------------------------------

class TestExistingAdaptiveRouterUnchanged:
    def test_router_threshold_and_bounds(self):
        from configs.config import config
        from src.models.adaptive_router import AdaptiveRouter
        r = AdaptiveRouter()
        assert config.routing.reference_length == 20
        assert config.routing.noise_threshold == 0.20
        assert config.routing.w1 == 0.05
        assert config.routing.w2 == 12.0
        assert config.routing.alpha_min == 0.02
        assert config.routing.alpha_max == 0.25

        # Low noise test: alpha must be alpha_min
        dec_low = r.route(noise_score=0.15, token_length=18)
        assert dec_low.alpha == pytest.approx(0.02)
        assert dec_low.routing_state == "low_noise_default"

        # High noise test: alpha must be clamped <= 0.25
        dec_high = r.route(noise_score=0.90, token_length=5)
        assert dec_high.alpha <= 0.25
        assert dec_high.alpha >= 0.02


# ---------------------------------------------------------------------------
# 12-14. Ablation Configurations in 3-Class Framework
# ---------------------------------------------------------------------------

class TestAblationConfigurations3Class:
    @pytest.fixture
    def framework(self):
        from unittest.mock import MagicMock
        mock_bert = MagicMock()
        mock_bert.predict.return_value = DistilBert3ClassOutput(
            positive_prob=0.7,
            negative_prob=0.2,
            neutral_prob=0.1,
            predicted_label=0,
            logits=[1.5, -0.5, -1.0],
        )
        return DynamicFusion3ClassFramework(distilbert_model=mock_bert)

    def test_static_fusion_ablation(self, framework):
        """Static fusion uses fixed alpha (e.g. 0.02)."""
        res = framework.predict("mast movie bhai", mode="static", fixed_alpha=0.02)
        assert res.alpha == pytest.approx(0.02)
        assert res.mode == "static"

    def test_distilbert_only_ablation(self, framework):
        """DistilBERT-only sets alpha to 0.0."""
        res = framework.predict("mast movie bhai", mode="distilbert_only")
        assert res.alpha == pytest.approx(0.0)
        assert res.mode == "distilbert_only"
        # p_final must equal p_distilbert
        assert res.p_final == pytest.approx(res.p_distilbert)

    def test_vader_only_ablation(self, framework):
        """VADER-only sets alpha to 1.0."""
        res = framework.predict("mast movie bhai", mode="vader_only")
        assert res.alpha == pytest.approx(1.0)
        assert res.mode == "vader_only"
        # p_final must equal p_vader
        assert res.p_final == pytest.approx(res.p_vader)

    def test_dynamic_fusion_full_system(self, framework):
        """Dynamic fusion invokes adaptive router."""
        res = framework.predict("mast movie bhai 😊😊😊 !!!", mode="dynamic")
        assert 0.02 <= res.alpha <= 0.25
        assert res.mode == "dynamic"
