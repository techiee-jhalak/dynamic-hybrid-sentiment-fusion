# EXPERIMENTS — Dynamic Hybrid Sentiment Fusion

> **Status:** Framework implemented & validated. Execution against held-out data
> requires a real dataset file and (optionally) a trained DistilBERT checkpoint.
> All results marked `NOT RUN` are structurally ready; no values have been
> fabricated or copied from the paper.

---

## 1. Research Formulas (Protected — Must Not Be Modified)

### 1.1 Composite Noise Score

```
N = 0.25·E + 0.25·R + 0.30·C + 0.20·S
```

| Symbol | Feature | Source |
|--------|---------|--------|
| E | Emoji density (emoji count / max(token count, 1)) | `NoiseQuantifier.emoji_density` |
| R | Repetition score (elongated-char tokens / max(token count, 1)) | `NoiseQuantifier.repetition_score` |
| C | Code-Mixing Intensity (Hinglish tokens / max(non-stopword tokens, 1)) | `NoiseQuantifier.code_mixing_ratio` |
| S | Symbol density (special-symbol count / max(token count, 1)) | `NoiseQuantifier.symbol_density` |

### 1.2 Adaptive Routing (α)

```
if N ≤ 0.20:
    α = 0.02
else:
    α = clamp(σ(z), 0.02, 0.25)   where z = learned routing logit
```

### 1.3 Final Sentiment Score (Fusion)

```
S_final = α · S_vader + (1 − α) · S_distilbert
```

---

## 2. Models Evaluated

| ID | Model | Implementation |
|----|-------|----------------|
| M1 | Logistic Regression / TF-IDF baseline | `src/baselines/lr_tfidf.py` |
| M2 | VADER | `src/models/vader_model.py` |
| M3 | DistilBERT | `src/models/distilbert_model.py` |
| M4 | Static Fusion (α fixed = 0.02) | Variant B in `src/evaluation/ablation.py` |
| M5 | Dynamic Fusion (full system) | `src/models/dynamic_fusion.py` |

---

## 3. Evaluation Metrics

All metrics computed from real model predictions on the held-out **test split**
(20 % of dataset, stratified by label, seed = 42). No metric is hardcoded.

| Metric | Implementation |
|--------|---------------|
| Accuracy | `sklearn.metrics.accuracy_score` |
| Precision (macro) | `sklearn.metrics.precision_score(average='macro')` |
| Recall (macro) | `sklearn.metrics.recall_score(average='macro')` |
| Macro F1 | `sklearn.metrics.f1_score(average='macro')` |
| Binary F1 | `sklearn.metrics.f1_score(average='binary')` |
| ROC-AUC | `sklearn.metrics.roc_auc_score` |
| Confusion Matrix | `sklearn.metrics.confusion_matrix` |
| Latency (ms/sample) | Wall-clock average over full test batch |

### 3.1 Results — Main Evaluation

> Results below are placeholders that will be overwritten by `evaluate.py`
> once the real dataset is provided. Until then all values report `NOT RUN`.

| Model | Accuracy | Macro F1 | Precision | Recall | ROC-AUC | Latency (ms) |
|-------|----------|----------|-----------|--------|---------|--------------|
| LR/TF-IDF | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| VADER | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| DistilBERT | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Static Fusion | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Dynamic Fusion | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

**Artifacts saved to:** `results/metrics/`, `results/predictions/`, `results/confusion_matrices/`

---

## 4. Ablation Study

Four variants are defined in `src/evaluation/ablation.py`:

| Variant | Label | Description |
|---------|-------|-------------|
| A | Full | Complete Dynamic Fusion (baseline for Δ F1) |
| B | Static | Fixed α = 0.02 (disables gated routing entirely) |
| C | No Length-Aware Smoothing | Raw N score used without length normalization |
| D | No Gated Threshold | α routed for all samples regardless of N ≤ 0.20 |

### 4.1 Ablation Results (Macro F1 on Test Split)

| Variant | Description | Macro F1 | Δ vs. Full |
|---------|-------------|----------|-----------|
| A — Full | Complete system | NOT RUN | — |
| B — Static | No dynamic routing | NOT RUN | NOT RUN |
| C — No Length Smoothing | Raw N score | NOT RUN | NOT RUN |
| D — No Gate | No low-noise threshold | NOT RUN | NOT RUN |

**Artifacts saved to:** `results/ablation/ablation_results.csv`

---

## 5. Noise-Sensitivity Analysis

Test instances are partitioned into four bands by composite noise score N,
as defined in `src/evaluation/noise_sensitivity.py`:

| Band | N range | Interpretation |
|------|---------|----------------|
| LOW | 0.00 ≤ N ≤ 0.20 | Clean, standard text |
| MODERATE | 0.20 < N ≤ 0.50 | Some informal markers |
| HIGH | 0.50 < N ≤ 0.80 | Heavy code-mixing or slang |
| EXTREME | 0.80 < N ≤ 1.00 | Severely noisy (emojis, Devanagari, elongation) |

Minimum 1 sample required per group for metrics to be computed; otherwise
status = `insufficient_samples` (never fabricated).

### 5.1 Results by Noise Band

> Results will be populated by `evaluate.py --noise-sensitivity` once real
> test data is available.

| Model | Band | Samples | Accuracy | Macro F1 | Status |
|-------|------|---------|----------|----------|--------|
| Dynamic Fusion | LOW | NOT RUN | NOT RUN | NOT RUN | pending |
| Dynamic Fusion | MODERATE | NOT RUN | NOT RUN | NOT RUN | pending |
| Dynamic Fusion | HIGH | NOT RUN | NOT RUN | NOT RUN | pending |
| Dynamic Fusion | EXTREME | NOT RUN | NOT RUN | NOT RUN | pending |

**Artifacts saved to:** `results/noise_sensitivity/noise_sensitivity.csv`, `results/noise_sensitivity/noise_sensitivity_summary.json`

---

## 6. Statistical Significance Tests

**Test:** McNemar's Test with Edwards' continuity correction, df = 1, α = 0.05.

Reference model: **Dynamic Fusion (M5)** compared against:
- DistilBERT (M3)
- Static Fusion (M4)

### 6.1 McNemar Results

| Comparison | b (A only) | c (B only) | χ² | p-value | Significant? | Superior |
|------------|-----------|-----------|-----|---------|-------------|---------|
| Dynamic vs. DistilBERT | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Dynamic vs. Static Fusion | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

**Artifacts saved to:** `results/statistical_tests/mcnemar_tests.json`, `results/statistical_tests/mcnemar_tests.csv`

---

## 7. Deterministic Error Analysis

Errors on the Dynamic Fusion model are categorized using only deterministic
rule-based checks — never guesses. Implementation in `src/evaluation/error_analysis.py`.

| Category | Detection Rule |
|----------|---------------|
| `severe cross-script code-mixing` | Devanagari + Latin co-occurrence OR C ≥ 0.40 and ≥4 words |
| `dual-sarcasm/polysemous emojis` | Negative label with positive emoji (or vice-versa) |
| `structural/implicit irony` | Irony regex OR contrastive discourse marker + ≥5 words |
| `slang/syntax drift` | R ≥ 0.35 OR known heavy-slang token present |
| `unavailable` | None of the above patterns match (never fabricated) |

### 7.1 Error Distribution

> Will be populated from `results/error_analysis/error_summary_Dynamic_Fusion.json`
> once real data is available.

| Category | Count | % of Errors |
|----------|-------|-------------|
| severe cross-script code-mixing | NOT RUN | NOT RUN |
| dual-sarcasm/polysemous emojis | NOT RUN | NOT RUN |
| structural/implicit irony | NOT RUN | NOT RUN |
| slang/syntax drift | NOT RUN | NOT RUN |
| unavailable | NOT RUN | NOT RUN |

**Artifacts saved to:** `results/error_analysis/error_analysis_Dynamic_Fusion.csv`, `results/error_analysis/error_summary_Dynamic_Fusion.json`

---

## 8. Reproducibility Settings

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Dataset split | 70/10/20 stratified |
| Threshold for binary prediction | 0.50 |
| McNemar α | 0.05 |
| Min noise-band samples | 1 |
| Python | 3.12.10 |
| Key dependencies | scikit-learn, scipy, numpy, vaderSentiment, emoji |

All splits are generated by `src/data/dataset_pipeline.py` using
`sklearn.model_selection.train_test_split(stratify=labels, random_state=42)`.
No shuffle is applied after splitting to prevent leakage.

---

## 9. How to Reproduce

### Step 1 — Prepare the dataset
```bash
python -c "from src.data.dataset_pipeline import load_and_split_dataset; load_and_split_dataset('data/raw/dataset.csv')"
```

### Step 2 — Run full evaluation
```bash
python evaluate.py
```

### Step 3 — Run ablation study
```bash
python ablation.py
```

### Step 4 — Run individual test suite
```bash
python -m pytest -q
```

---

## 10. Test Coverage

| Test File | Module Covered | Tests |
|-----------|---------------|-------|
| `test_noise_sensitivity.py` | `src/evaluation/noise_sensitivity.py` | 28 |
| `test_statistical_tests.py` | `src/evaluation/statistical_tests.py` | 27 |
| `test_error_analysis.py` | `src/evaluation/error_analysis.py` | 40 |
| `test_ablation.py` | `src/evaluation/ablation.py` | ~30 |
| `test_evaluation_engine.py` | `src/evaluation/engine.py` | ~40 |
| `test_noise_quantifier.py` | `src/features/noise_quantifier.py` | ~15 |
| `test_models.py` | `src/models/base.py` + all models | ~12 |
| *(other existing tests)* | dataset, preprocessing, baselines, router, etc. | remaining |
| **Total** | | **267 passed** |

---

## 11. Integrity Guarantees

- **No metric is hardcoded.** All values in results tables derive from live
  `sklearn` or `scipy` calls on actual model outputs.
- **No samples are fabricated.** The dataset pipeline uses real CSV files;
  no synthetic rows are injected.
- **No results are copied from the paper.** Tables above show `NOT RUN` until
  the evaluation scripts are executed against actual data.
- **Formulas are unchanged** from the literature: N, α, S_final.
