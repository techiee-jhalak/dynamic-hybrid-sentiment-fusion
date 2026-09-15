# Dataset Manifest

## ⚠️ This dataset is NOT SAIL 2017.

---

## Dataset Identity

| Field | Value |
|-------|-------|
| **Dataset Name** | SemEval-2020 Task 9 / SentiMix — Hindi-English |
| **Task** | Sentiment Analysis for Code-Mixed Social Media Text |
| **Language Pair** | Hindi-English (Hinglish) |
| **Original Source** | SemEval-2020 Task 9 shared task |
| **Project Page** | https://ritual.uh.edu/semeval-2020-task9/ |
| **Paper DOI** | 10.18653/v1/2020.semeval-1.10 |

---

## ⚠️ Explicit Non-SAIL Declaration

> **This dataset is NOT SAIL 2017.**
> SAIL 2017 refers to the NLP Tool Contest @ ICON-2017 dataset from Jadavpur University.
> SentiMix is a separate dataset from SemEval-2020 Task 9, with different collection methods, annotation protocols, and license terms. Do NOT conflate the two.

---

## Original File Inventory

| Filename | Role | Sentences | Note |
|----------|------|-----------|------|
| `Hinglish_train_14k_split_conll.txt` | Training | 14,000 | Labelled |
| `Hinglish_dev_3k_split_conll.txt` | Development/Val | 3,000 | Labelled |
| `Hinglish_test_unlabelled_conll_updated.txt` | Test (tokens) | 3,000 | Unlabelled tokens |
| `Hinglish_test_labels.txt` | Test (labels) | 3,000 | `Uid,Sentiment` CSV with header |

---

## Actual Counts (Verified)

| Split | Sentences | Positive | Negative | Neutral |
|-------|-----------|----------|----------|---------|
| Train | 14,000 | 4,634 | 4,102 | 5,264 |
| Dev | 3,000 | 982 | 890 | 1,128 |
| Test | 3,000 | 1,000 | 900 | 1,100 |
| **Total** | **20,000** | **6,616** | **5,892** | **7,492** |

---

## Label Classes

| Integer | String | Meaning |
|---------|--------|---------|
| 0 | `positive` | Positive sentiment |
| 1 | `negative` | Negative sentiment |
| 2 | `neutral` | Neutral / no clear polarity |

> **IMPORTANT**: Neutral is a distinct third class. It must NOT be silently mapped to Positive or Negative.

---

## File Format (CoNLL)

```
meta  <uid>  <label>
<token>  <lang_tag>
<token>  <lang_tag>
...
<blank line>
```

Language tags: `Eng` (English), `Hin` (Hindi), `O` (symbols/other)

The test file omits the `<label>` field on the `meta` line. Labels are supplied via `Hinglish_test_labels.txt`.

---

## Test Label Alignment

- **Status: VERIFIED** — 3,000 sentence IDs in `Hinglish_test_unlabelled_conll_updated.txt` match exactly with 3,000 IDs in `Hinglish_test_labels.txt` (0 missing IDs in either direction).

---

## Duplicate / Leakage Analysis

| Check | Result |
|-------|--------|
| Duplicates within TRAIN | 0 |
| Duplicates within DEV | 0 |
| Duplicates within TEST | 0 |
| TRAIN ∩ DEV overlap | 0 |
| TRAIN ∩ TEST overlap | 0 |
| DEV ∩ TEST overlap | 0 |

> No leakage detected. Dataset is clean.

---

## Data Split Strategy

The SentiMix-provided TRAIN / DEV / TEST splits are used as-is. No re-splitting into 70:10:20 has been performed. If a different split scheme is required, it must be explicitly designed and documented before Phase 6.

---

## License / Source Information

The dataset was distributed as part of the SemEval-2020 Task 9 shared task. It was obtained from the following GitHub repository bundled in the ZIP file `Hinglish-Sentiment-Analysis-using-XLM-R-main.zip`:

- ZIP source: User-provided
- The original SentiMix dataset was released for academic research use by the shared task organizers.
- Redistribution terms: Follow the original SemEval-2020 Task 9 shared task data release terms. Do NOT redistribute without verifying the task's data use agreement.

---

## Loader

`src/data/sentimix_loader.py` — SentimixCoNLLParser + SentimixDataLoader

---

## Verification Status

| Item | Status |
|------|--------|
| Dataset identity confirmed | ✅ SentiMix, NOT SAIL 2017 |
| File counts verified | ✅ |
| Label alignment verified | ✅ |
| Duplicate/leakage check | ✅ Clean |
| Parser implemented | ✅ |
| Raw text preserved | ✅ |
