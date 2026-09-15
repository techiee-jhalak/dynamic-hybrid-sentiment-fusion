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
- **Current State**: `src/models/vader_model.py` is configured for binary polarity scoring:
  - Outputs continuous compound score $c \in [-1, 1]$ mapped to scalar positive probability $S_{VADER} = (c + 1) / 2 \in [0, 1]$.
  - While VADER natively provides `{pos, neg, neu}` proportions, these raw proportions represent lexical percentages rather than a calibrated 3-class probability distribution.
- **Limitation**: Direct 3-class probabilistic inference with VADER without a formal calibration scheme is scientifically unsubstantiated.
- **Decision Needed**: Requires formal calibration or thresholding rule before SentiMix 3-class fusion.

---

## DistilBERT Compatibility
- **Implemented Module**: `src/models/distilbert_3class.py`
  - Defines `DistilBert3ClassModel` with `num_labels = 3`.
  - Structured output `DistilBert3ClassOutput` containing `positive_prob`, `negative_prob`, `neutral_prob`, `predicted_label`, and `logits`.
  - Default hyperparameters aligned with research spec: 3 epochs, batch size 16, learning rate 2e-5, max sequence length <= 128.
  - **No training executed** (adhering strictly to research protocol).

---

## Dynamic Fusion Compatibility
- **Core Formula**:
  $$S_{final} = \alpha \cdot S_{VADER} + (1 - \alpha) \cdot S_{DistilBERT}$$
- **Current Limitation**: The mathematical definition of $S_{final}$ operates on 1D scalar probabilities for binary classification with decision threshold 0.50.
- **3-Class Gap**:
  - A 3-class task requires either:
    1. **Vector Fusion**: $\mathbf{P}_{final} = \alpha \mathbf{P}_{VADER} + (1 - \alpha) \mathbf{P}_{DistilBERT}$ where $\mathbf{P} \in \Delta^2$.
    2. **Hierarchical Fusion**: Neutral vs Polarity detection followed by Positive vs Negative routing.
    3. **Continuous Polarity Fusion**: Fusion of continuous sentiment scores with 2 decision boundaries (e.g. $[-\tau, +\tau]$ for Neutral).
- Changing this formula without research authorization violates experimental integrity.

---

## Required Methodological Decisions
Before running experiments on SentiMix, the following methodological decisions must be finalized:
1. **Target Task Formulation**: Whether the paper will report 3-class macro-F1 (SemEval-2020 standard) or evaluate binary subsets.
2. **3-Class VADER Calibration**: Definition of VADER probability vector or score representation for Neutral instances.
3. **3-Class Fusion Formulation**: Explicit mathematical specification for fusing multi-class probability vectors or multi-threshold scores under the dynamic noise-adaptive weighting $\alpha$.

---

## Phase 6 Readiness

### **NOT_READY_FOR_PHASE_6**

**Rationale**:
1. The dataset integrated is **SemEval-2020 Task 9 (SentiMix Hinglish)**, which has 3 classes (`positive`, `negative`, `neutral`), whereas the active research specification and fusion mathematics are formulated for binary classification (`positive`, `negative`).
2. Training DistilBERT or running Phase 6 experiments prior to resolving the 3-class fusion formulation would invalidate the research claims and compromise experiment reproducibility.
3. The dataset loader, CoNLL parser, data structure, manifest, unit tests, and 3-class architecture scaffolding are fully completed and verified. Experiments must pause until methodology alignment is approved.
