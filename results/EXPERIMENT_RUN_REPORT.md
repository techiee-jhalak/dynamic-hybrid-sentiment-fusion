# EXPERIMENT RUN REPORT — Phase 6 Execution

> **Execution Status**: HALTED / NOT RUN (Missing Required Assets)

## 1. Dataset Used
**NOT RUN**: The requested SAIL 2017 Hinglish and Curated benchmark datasets do not exist within the workspace (`data/` is entirely missing/empty). No data files were found to process.

## 2. Dataset Size
**NOT RUN**: 0 rows processed.

## 3. Class Distribution
**NOT RUN**: N/A.

## 4. Train/validation/test split
**NOT RUN**: N/A.

## 5. Model Configurations
**NOT RUN**: Baseline, VADER, DistilBERT, Static Fusion, and Dynamic Fusion models were configured correctly in code, but execution was halted because training and testing loops cannot execute without real data samples.

## 6. Main Results
**NOT RUN**: No predictions were made. No metrics (Accuracy, Precision, Recall, F1, ROC-AUC) could be calculated without data.

## 7. Ablation Results
**NOT RUN**: Ablation configs A, B, C, D were not run.

## 8. Noise Sensitivity
**NOT RUN**: No texts were evaluated, therefore no partitioning by noise group (LOW, MODERATE, HIGH, EXTREME) occurred.

## 9. McNemar Statistical Tests
**NOT RUN**: No paired predictions available to perform significance testing.

## 10. Error Analysis
**NOT RUN**: No incorrect predictions to analyze.

## 11. Latency
**NOT RUN**: Inference latency cannot be accurately tested over an empty test set.

## 12. Model Size
**NOT RUN**: No fine-tuned DistilBERT checkpoint exists in `saved_models/checkpoints/` to measure.

## 13. Reproducibility Metadata
- **Timestamp**: 2026-09-15T23:45:00+05:30
- **Git Commit**: `326feb1`
- **Execution Status**: HALTED DUE TO MISSING DATA.

## 14. Any Experiments Not Run
**ALL EXPERIMENTS NOT RUN**.

## 15. Limitations
The entire experimental phase is blocked because the repository does not contain the underlying dataset files (e.g., `SAIL 2017 Hinglish`) or pre-trained model weights. 

## 16. Suitability for Inclusion in Research Paper
**UNSUITABLE**. There are zero experimentally derived metrics available in this repository instance. Any metrics reported in the paper must be clearly identified as prior reference values, because this reproduction pipeline could not generate verifiable results.
