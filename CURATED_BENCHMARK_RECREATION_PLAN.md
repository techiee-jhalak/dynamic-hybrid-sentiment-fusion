# Curated Benchmark Recreation Plan

> **Status**: PLAN ONLY — samples must NOT be fabricated or auto-generated.
> **Activation**: This plan may only be executed after legitimate human data collection and annotation.

---

## Overview

The original 1,200-sample manually curated benchmark for this research project is unavailable and unrecoverable from any local or version-controlled source. This document defines a scientifically defensible procedure for creating a new, legitimate replacement benchmark from scratch.

**Samples must be genuinely collected and human-verified before being included in the research dataset.**

---

## 1. Target Size

- **1,200 samples** (minimum)
- Recommended: collect ~1,400–1,500 raw candidates to allow for rejection during quality control

---

## 2. Target Domain

- Hindi-English code-mixed (Hinglish) social media sentiment text
- Sources: Twitter/X, Facebook public posts, public Reddit (r/india, r/bollywood, r/cricket), public WhatsApp forwards (with consent)
- Content must be **publicly available** or collected with appropriate consent

---

## 3. Collection Sources (Legally Accessible Only)

| Source | Access Method | Notes |
|--------|--------------|-------|
| Twitter/X Academic API | Apply for academic research access | Keyword/hashtag-based; requires approval |
| Reddit public posts | PRAW library (public API) | Only public subreddits |
| Facebook public pages | CrowdTangle (where accessible) or manual collection | Public pages only |
| Previously published corpora | Citation + license check | E.g., LinCE, GLUECoS — verify license permits redistribution |

**Strictly excluded**:
- Private messages
- Gated or paywalled content
- Content from minors
- Content requiring special platform authorization not obtained

---

## 4. Inclusion Criteria

A sample is eligible only if it satisfies **all** of the following:

1. Contains genuine Hindi-English code-mixing (not purely Hindi or purely English)
2. Is sentiment-bearing (expresses an opinion, emotion, or attitude)
3. Has sufficient context to be interpretable in isolation (≥ 5 meaningful tokens)
4. Is not a duplicate of any other collected sample
5. Is not a near-duplicate (cosine similarity < 0.85 at character n-gram level)
6. Does not overlap with any known SAIL 2017 training/test samples (once obtained)
7. Is publicly available under a license or terms-of-service compatible with academic research use

---

## 5. Exclusion Criteria

A sample must be **rejected** if it meets any of the following:

- Purely English or purely Hindi (no code-mixing)
- Spam, bot-generated, or promotional content
- Only a URL, hashtag, or @mention with no semantic content
- Ambiguous or sarcastic beyond human interpretability without context
- Contains personally identifying information (PII)
- Sexually explicit, threatening, or otherwise harmful content
- Exact or near-duplicate of another sample in the collection pool
- Overlap with SAIL 2017 data (after SAIL data is obtained)

---

## 6. Annotation Protocol

### Label Set
- **Positive** (1): Text expresses positive sentiment, happiness, approval, or praise
- **Negative** (0): Text expresses negative sentiment, sadness, criticism, or disapproval
- *(Neutral is excluded per the binary research design — if this changes, update accordingly)*

### Annotation Unit
- Sentence-level polarity (not aspect-level, not token-level)

---

## 7. Human Annotation Process

> **This process must be completed before any sample is included.**

### Step 1 — Annotator Recruitment
- Minimum **3 independent annotators** per sample
- Annotators must be native/near-native Hindi speakers with English proficiency
- Annotators must NOT be the same person who designed the research system

### Step 2 — Annotator Training
- Provide written annotation guidelines with at least 30 worked examples
- Include edge cases: sarcasm, slang, abbreviations, emoji-heavy text
- Conduct a calibration round of 50 samples with group discussion before main annotation

### Step 3 — Independent Annotation
- Each annotator labels all samples independently without seeing others' labels
- No discussion until all labels are submitted for a batch

### Step 4 — Disagreement Resolution
- Samples with unanimous agreement (3/3): accepted directly
- Samples with 2/3 agreement: accepted with majority label; flagged for review
- Samples with 0/3 agreement (complete disagreement): **discarded**

### Step 5 — Inter-Annotator Agreement (IAA)
- Compute Cohen's Kappa (κ) for each annotator pair
- Minimum acceptable κ: **0.60**
- If κ < 0.60 for any pair, conduct re-calibration before proceeding

---

## 8. Quality Control

| Step | Action |
|------|--------|
| Exact duplicate detection | `df.drop_duplicates(subset=["text"])` |
| Near-duplicate detection | Character n-gram similarity (threshold: 0.85) |
| Minimum token count | ≥ 5 meaningful tokens after basic normalization |
| IAA check | κ ≥ 0.60 required |
| Final human review | Sample 100 random accepted items for spot-check |
| SAIL overlap check | Once SAIL data is obtained, run exact + near-duplicate check |

---

## 9. Data Leakage Prevention

- Maintain a separate hash list of all SAIL 2017 text entries (once obtained)
- Run exact and near-duplicate checks between the curated benchmark and SAIL before finalizing splits
- The curated benchmark test set must have **zero overlap** with the SAIL training set
- Record all deduplication operations and their outcomes

---

## 10. Dataset Documentation

Every released version of the benchmark must include a `BENCHMARK_DATASHEET.md` containing:

| Field | Value |
|-------|-------|
| Dataset name | Dynamic Fusion Hinglish Sentiment Benchmark v1.0 |
| Collection date | [TO BE FILLED AFTER COLLECTION] |
| Collection sources | [TO BE FILLED AFTER COLLECTION] |
| Platform(s) | [TO BE FILLED] |
| Annotation procedure | As described in this document |
| Number of annotators | [TO BE FILLED] |
| IAA (κ) | [TO BE FILLED AFTER ANNOTATION] |
| Final class distribution | [TO BE FILLED AFTER ANNOTATION] |
| License | [To be determined — CC BY 4.0 or equivalent academic research license] |
| Usage constraints | Academic research only unless license permits otherwise |
| PII handling | No PII retained; usernames anonymized |
| Intended use | Benchmarking sentiment analysis models on code-mixed Hindi-English text |

---

## 11. Reproducibility

- All collection scripts must be version-controlled and included in `data/collection_scripts/`
- All annotation logs must be archived (even if not committed publicly)
- Random seeds for any sampling must be documented
- The final dataset must include a SHA-256 checksum file

---

## ⚠️ Important Reminder

This document describes a **plan only**. No samples have been collected or annotated.

**Samples must be genuinely collected and human-verified before being included in the research dataset.**

Do not proceed to Phase 6 experiments using this benchmark until:
1. All ~1,200+ samples are collected from verified sources
2. All samples have been annotated by ≥ 3 human annotators
3. IAA (κ ≥ 0.60) has been confirmed
4. Disagreements have been resolved
5. Overlap with SAIL data has been checked
6. Final datasheet has been completed
