# Dataset Acquisition & Recovery Report

## 1. Current Dataset Status

**CRITICAL**: Both required research datasets are entirely absent from the local workspace and Git history.

- `data/` directory: **does not exist**
- No dataset files (`.csv`, `.tsv`, `.xlsx`, `.jsonl`, `.parquet`, `.zip`, `.rar`) were found anywhere outside the `.venv` library path
- Git object store confirms no dataset was ever committed to this repository

---

## 2. Local Dataset Search

**Search scope**: Entire workspace tree, excluding `.venv` library internals.

**Patterns searched**: SAIL, SAIL2017, Hinglish, Hindi-English, code-mixed, sentiment, curated, benchmark, 1200, 12568, 13768

**Extensions searched**: `.csv`, `.tsv`, `.txt`, `.json`, `.jsonl`, `.xlsx`, `.parquet`, `.zip`, `.rar`

**Locations checked**:
- `data/` — directory does not exist
- `datasets/` — directory does not exist
- `research_pipeline/` — directory does not exist
- `saved_models/checkpoints/` — empty
- Repository root — no data files

**Result**: **Zero candidate dataset files found.**

---

## 3. Git History Search

Commands executed:
```
git log --all --name-status
git rev-list --objects --all
git ls-tree -r --name-only HEAD (all 6 commits)
```

**Findings**:
- The repository has 6 commits total (initial + 5 phase commits).
- No `.csv`, `.tsv`, `.xlsx`, or any dataset file appears in any commit's object store.
- The `.gitignore` file explicitly excludes `data/raw/*` and `data/processed/*` — confirming the dataset was always intended to be kept off Git.
- **No dataset is recoverable from Git history.**

---

## 4. SAIL 2017

| Property | Value |
|----------|-------|
| **Provenance** | SAIL Code-Mixed @ ICON-2017, NLP Tool Contest, Jadavpur University |
| **Official source** | https://brajagopalcse.github.io/SAIL_CodeMixed-ICON-2017/ |
| **Official GitHub** | https://github.com/brajagopalcse/SAIL_CodeMixed-ICON-2017 |
| **Local availability** | **NOT FOUND** |
| **Access status** | OFFICIAL_ACCESS_REQUIRED |
| **Row count verified locally** | N/A |
| **Verification status** | NOT_FOUND |

**Critical finding**: The official SAIL GitHub repository (`brajagopalcse/SAIL_CodeMixed-ICON-2017`) contains **only evaluation scripts** (`evalSAIL.py`, `validateSAIL.py`, `randomBaseline.py`), competition results (`SAIL_CodeMixed_2017_results.xlsx`), and documentation. **The actual training/test dataset files are not publicly distributed** in the repository. The official page states the data was provided to registered participants only during the competition period.

**Classification**: `OFFICIAL_ACCESS_REQUIRED`

**What the researcher must do manually**:
1. Contact the SAIL 2017 organizers directly via the contact information on https://brajagopalcse.github.io/SAIL_CodeMixed-ICON-2017/
2. Email: The organizing committee (contact listed on the official page) to request post-contest dataset access for research purposes.
3. Cite the dataset properly per the organizers' instructions.
4. Alternatively, search academic repositories that may host derived versions (e.g., papers with code, Zenodo, Figshare) — but any such copy must be verified against the original source for authenticity.

---

## 5. Original 1,200 Curated Benchmark

| Property | Value |
|----------|-------|
| **Recovery status** | NOT FOUND |
| **Provenance** | Unknown — no documentation exists in the repo for how this was originally created |
| **Row count** | N/A |
| **Verification status** | NOT_FOUND |

**Classification**: `NOT_FOUND`

No file matching any expected benchmark naming convention was found locally or in Git history. There is no annotation log, creation script, or provenance record in the codebase to indicate how the original 1,200 samples were gathered or labeled.

---

## 6. Legitimate Acquisition Route

### For SAIL 2017:
- **Primary route**: Contact SAIL organizers at Jadavpur University directly.
- **Contact page**: https://brajagopalcse.github.io/SAIL_CodeMixed-ICON-2017/#contact
- **Secondary route**: Search Hugging Face Datasets Hub (`huggingface.co/datasets`) for any publicly licensed Hi-En code-mixed dataset derived from SAIL or compatible with its annotation scheme.
- **Do NOT**: Download any third-party file labelled "SAIL" without verifying authenticity with the original organizers.

### For the 1,200 Curated Benchmark:
- The original benchmark cannot be recovered.
- A **recreation plan** has been created (see `CURATED_BENCHMARK_RECREATION_PLAN.md`).
- The actual samples must be **genuinely collected and human-verified** before being used.

---

## 7. New Curated Benchmark

- **Original recovered**: NO
- **Recreation required**: YES
- **Recreation plan created**: YES — see `CURATED_BENCHMARK_RECREATION_PLAN.md`

---

## 8. Research Integrity

This report explicitly confirms:
- **No fabricated samples** were generated or placed in any file.
- **No fabricated labels** were created.
- **No fake metrics** were computed or stored.
- The `.gitignore` confirms datasets were deliberately excluded from version control — they were never committed and cannot be fabricated retroactively.

---

## 9. Phase 6 Readiness

**NOT_READY_FOR_PHASE_6**

Both required datasets are missing. No experiments can be executed with integrity.

---

## 10. Required Next Action

**Contact the SAIL 2017 organizers** at Jadavpur University to request legitimate access to the Hindi-English code-mixed sentiment dataset for research reproduction purposes.

Until the actual dataset files are physically present and verified, Phase 6 cannot and must not proceed.
