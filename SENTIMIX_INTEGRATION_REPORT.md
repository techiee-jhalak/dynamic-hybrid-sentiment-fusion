# SentiMix Hinglish Dataset Integration and 3-Class Adaptation Report

## Dataset Identity
- **Name**: SemEval-2020 Task 9 / SentiMix — Hindi-English (Hinglish)
- **Task**: Sentiment Analysis for Code-Mixed Social Media Text (Twitter)
- **Official Venue**: SemEval-2020 Task 9 (Coling 2020)
- **URL**: https://ritual.uh.edu/semeval-2020-task9/
- **Paper DOI**: 10.18653/v1/2020.semeval-1.10

---

## ⚠️ Explicit Warning: This is NOT SAIL 2017
> **CRITICAL RESEARCH INTEGRITY NOTICE**:
> This dataset is **SemEval-2020 Task 9 (SentiMix Hinglish)**. It is **NOT SAIL 2017** (NLP Tool Contest @ ICON-2017).
> - SentiMix features 3-class labels (`positive`, `negative`, `neutral`), 20,000 total sentences, and tweet-level token CoNLL annotations with word-level language tags (`Eng`, `Hin`, `O`).
> - SAIL 2017 has distinct provenance, distribution, and guidelines.
> - Under no circumstances should this dataset be referred to, renamed, or evaluated as SAIL 2017.

---

## File Inventory

| Filename | Source Path | Size (Bytes) | Role |
|----------|-------------|--------------|------|
| `Hinglish_train_14k_split_conll.txt` | `data/raw/sentimix/` | 3,437,706 | Training Split (14,000 sentences) |
| `Hinglish_dev_3k_split_conll.txt` | `data/raw/sentimix/` | 739,899 | Development Split (3,000 sentences) |
| `Hinglish_test_unlabelled_conll_updated.txt` | `data/raw/sentimix/` | 726,029 | Test Split Tokens (3,000 sentences) |
| `Hinglish_test_labels.txt` | `data/raw/sentimix/` | 43,222 | Test Split Labels (`Uid,Sentiment` CSV) |

---

## Actual Counts

| Split | Total Sentences | Unlabelled Sentences | Verified Labelled Sentences |
|-------|-----------------|----------------------|-----------------------------|
| **TRAIN** | 14,000 | 0 | 14,000 |
| **DEV** | 3,000 | 0 | 3,000 |
| **TEST** | 3,000 | 0 (mapped via label file) | 3,000 |
| **TOTAL** | **20,000** | **0** | **20,000** |

---

## Label Distribution

| Split | Positive | Negative | Neutral | Total |
|-------|----------|----------|---------|-------|
| **TRAIN** | 4,634 (33.10%) | 4,102 (29.30%) | 5,264 (37.60%) | 14,000 (100.0%) |
| **DEV** | 982 (32.73%) | 890 (29.67%) | 1,128 (37.60%) | 3,000 (100.0%) |
| **TEST** | 1,000 (33.33%) | 900 (30.00%) | 1,100 (36.67%) | 3,000 (100.0%) |
| **COMBINED** | **6,616 (33.08%)** | **5,892 (29.46%)** | **7,492 (37.46%)** | **20,000 (100.0%)** |

> **Key Takeaway**: Neutral is the plurality class across all splits (~37.5%). It cannot be dropped or silently merged into positive or negative classes without fundamentally corrupting the dataset semantics.

---

## Train/Dev/Test Structure
- The dataset provides official, canonical splits:
  - **Train**: 14,000 samples (70.0%)
  - **Dev**: 3,000 samples (15.0%)
  - **Test**: 3,000 samples (15.0%)
- **Policy**: These predefined partitions are preserved as-is in `SentimixDataLoader`. No artificial re-splitting or data shuffling across splits has been applied.

---

## Test Label Alignment
- **Status**: **EXACT MATCH (100% ALIGNED)**
- `Hinglish_test_unlabelled_conll_updated.txt` contains 3,000 unique sentence UIDs on `meta\t<uid>` lines.
- `Hinglish_test_labels.txt` contains 3,000 entries matching the exact UIDs with 0 missing UIDs and 0 unmatched records.
- Line-by-line UID matching is automated in `SentimixCoNLLParser._load_label_file()`.

---

## Duplicate/Leakage Analysis
Dataset integrity checks performed across sentence texts and token sequences:

| Check | Result |
|-------|--------|
| Exact duplicates within TRAIN | 0 |
| Exact duplicates within DEV | 0 |
| Exact duplicates within TEST | 0 |
| Overlap between TRAIN and DEV | 0 |
| Overlap between TRAIN and TEST | 0 |
| Overlap between DEV and TEST | 0 |

> **Conclusion**: Zero data leakage detected across the splits.

---

## Parser Status
- **Implemented Module**: `src/data/sentimix_loader.py`
  - `SentimixSample`: Typed dataclass preserving `uid`, space-joined raw `text`, per-token list `tokens`, per-token language tags `lang_tags`, integer `label`, and string `label_str`.
  - `SentimixCoNLLParser`: Streaming CoNLL block reader that retains emojis, character repetitions, punctuation, and language tags without lossy preprocessing.
  - `SentimixDataLoader`: High-level loader returning pandas DataFrames for train, dev, and test.
- **Verification**: 45 unit tests pass in `tests/test_sentimix_loader.py`.

---

## Three-Class Compatibility
- **Status**: Implemented with strict isolation from binary pipeline.
- SentiMix labels are indexed as:
  - `0`: Positive
  - `1`: Negative
  - `2`: Neutral
- The existing binary pipeline (`src/pipeline.py`, `src/models/dynamic_fusion.py`) remains unaltered.

---

## VADER Compatibility
- **Implemented Module**: `src/models/vader_3class.py` (`VADER3ClassAdapter`)
  - Maps native VADER lexical proportions (`pos`, `neg`, `neu`) to an L1-normalized 3-element score vector on the simplex $\Delta^2$ with ordering `[positive=0, negative=1, neutral=2]`.
  - Strictly deterministic, non-learned, and introduces zero fitted parameters or artificial thresholds.
  - Preserves isolation of the original binary model in `src/models/vader_model.py`.

---

## DistilBERT Compatibility
- **Implemented Module**: `src/models/distilbert_3class.py`
  - Defines `DistilBert3ClassModel` with `num_labels = 3`.
  - Structured output `DistilBert3ClassOutput` containing `positive_prob`, `negative_prob`, `neutral_prob`, `predicted_label`, and `logits`.
  - Default hyperparameters aligned with research spec: 3 epochs, batch size 16, learning rate 2e-5, max sequence length <= 128.
  - **No training executed** (adhering strictly to research protocol).

---

## Dynamic Fusion Compatibility
- **Implemented Module**: `src/models/fusion_3class.py` (`DynamicFusion3ClassFramework`, `fuse_3class_vectors`)
- **Formula**:
  $$\mathbf{P}_{final} = \alpha \cdot \mathbf{P}_{VADER} + (1 - \alpha) \cdot \mathbf{P}_{DistilBERT}$$
  $$\hat{y} = \arg\max(\mathbf{P}_{final})$$
- Preserves the exact mathematical router $\alpha \in [0.02, 0.25]$ and noise formulation $N = 0.25E + 0.25R + 0.30C + 0.20S$.
- Supports all ablation modes: `dynamic`, `static`, `distilbert_only`, and `vader_only`.

---

## Methodological Decisions Documented
All methodological questions have been formalized and resolved in `SENTIMIX_METHODOLOGY_DECISION.md`:
1. **Target Task Formulation**: Primary evaluation is 3-class Macro F1 (SemEval-2020 standard). Secondary Positive-vs-Negative subset analysis is provided strictly isolated.
2. **VADER 3-Class Representation**: Normalized lexical polarity proportions via deterministic `VADER3ClassAdapter`.
3. **3-Class Dynamic Fusion Formulation**: Convex combination on probability simplex $\Delta^2$ with identical noise-adaptive $\alpha$.

---

## Phase 6 Readiness

### **READY_FOR_PHASE_6**

**Readiness Checklist**:
- [x] SentiMix dataset available locally (14k train, 3k dev, 3k test)
- [x] CoNLL parser and dataset loader implemented and tested
- [x] 3-class label schema validated and aligned (`pos=0, neg=1, neu=2`)
- [x] VADER 3-class representation resolved (`VADER3ClassAdapter`)
- [x] Vector fusion implemented and tested (`DynamicFusion3ClassFramework`)
- [x] DistilBERT 3-class model architecture ready for training
- [x] 3-class evaluation metrics implemented (`src/evaluation/metrics_3class.py`)
- [x] Zero test regressions across entire repository (351 passed, 0 failed)
