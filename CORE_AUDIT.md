# CORE CORRECTNESS AUDIT

## 1. System Components Audit

### DATASET: PASS
- **Stratified Splitting:** Enforces exact 70:10:20 train/val/test stratified partitioning via `sklearn.model_selection.train_test_split`.
- **Label Normalization:** Strict mapping of various representations to integer labels `0` (Negative) and `1` (Positive) with rejection of ambiguous/NaN values.
- **Leakage & Duplicate Prevention:** Deduplication on `(text, label)` pairs, null/empty-string filtering, and strict partition separation with zero data leakage across splits.
- **Reproducibility:** Seeded deterministic random splitting (`random_seed=42`).
- **Integrity:** Zero fabricated samples.

### PREPROCESSING: PASS
- **Morphological & Noise Preservation:** Conservative TweetTokenizer preserves emojis, exclamation/question punctuation clusters, character repetitions (`reduce_len=False`), and Hinglish lexical elements.
- **Unicode & Cleaning:** Standardizes Unicode to NFC; handles URLs and handles cleanly without stripping symbols needed by downstream feature extraction.
- **Fidelity:** No aggressive lowercasing or stemmer/lemmatizer normalization that would discard morphological cues required for noise quantification.

### NOISE QUANTIFICATION: PASS
*(Fixed during audit)*
- **Audit Findings:** Previously, character repetition ($R$) and symbol density ($S$) normalized over character length instead of token count $N_{\text{tokens}}$, and code-mixing intensity ($C$) used a naive proportion instead of the literature C-Index formula.
- **Correction Applied:** Replaced formulas with exact paper formulations:
  - $E = \frac{N_{\text{emoji}}}{N_{\text{tokens}}}$
  - $R = \frac{N_{\text{repeat}}}{N_{\text{tokens}}}$, where $N_{\text{repeat}}$ is the number of characters in runs repeated consecutively more than twice ($\ge 3$).
  - $C = \frac{\min(\text{Tokens}_{\text{eng}}, \text{Tokens}_{\text{hin}})}{\max(\text{Tokens}_{\text{eng}}, \text{Tokens}_{\text{hin}}) + 1}$
  - $S = \frac{N_{\text{symbol}}}{N_{\text{tokens}}}$, where $N_{\text{symbol}}$ is the number of non-alphanumeric tokens excluding emojis.
  - $N = 0.25E + 0.25R + 0.30C + 0.20S$
- **Bounds:** All intermediate noise features and composite score $N$ are strictly clamped to $[0.0, 1.0]$.

### VADER: PASS
- **Lexicon Integration:** Uses standard NLTK `SentimentIntensityAnalyzer` singleton initialized once per instance.
- **Calibration & Mapping:** Standard linear normalization from compound score $[-1.0, 1.0]$ to continuous probability $S_{\text{vader}} \in [0.0, 1.0]$ via $S_{\text{vader}} = \frac{\text{compound} + 1.0}{2.0}$.
- **Formula Integrity:** No unsupported heuristic or invented calibration formulas.

### DISTILBERT: PASS
- **Architecture:** Hugging Face Transformers `AutoModelForSequenceClassification` with PyTorch backend.
- **Inference Efficiency:** Persistent tokenizer and model loaded once; inference runs strictly within `torch.no_grad()` and model in `eval()` mode.
- **Configurability:** Device selection (CUDA/CPU) is fully configurable.
- **Sequence Constraints:** Input sequence length bounded by `max_length <= 128`.

### ADAPTIVE ROUTER: PASS
- **Reference Parameters:** $L_0 = 20$, noise threshold $= 0.20$, $w_1 = 0.05$, $w_2 = 12.0$.
- **Routing Logic:**
  - If $N \le 0.20 \implies \alpha = 0.02$.
  - If $N > 0.20 \implies z = 0.05 \cdot (20 - L) + 12.0 \cdot N$, $\alpha_{\text{raw}} = \sigma(z) = \frac{1}{1 + e^{-z}}$, $\alpha = \text{clamp}(\alpha_{\text{raw}}, 0.02, 0.25)$.
- **Stability:** Numerically stable sigmoid function; hard clamping within $[0.02, 0.25]$.

### DYNAMIC FUSION: PASS
- **Interpolation Formula:** $S_{\text{final}} = \alpha \cdot S_{\text{vader}} + (1 - \alpha) \cdot S_{\text{distilbert}}$.
- **Binary Decision:** $\hat{y} = 1$ (Positive) if $S_{\text{final}} \ge 0.50$, else $0$ (Negative).
- **Output Container:** Implements full structured traceability (`FusionResult`) reporting component probabilities, dynamic alpha, fused score, binary decision, and confidence.

### TEST SUITE: PASS
- **Status:** 172 passed, 0 failed, 0 errors across 14 test modules.
- **Updates:**
  - `tests/test_noise_quantifier.py`: Updated `test_code_mixing_ratio_calculation` to validate exact paper formula $C = \frac{\min(T_{\text{eng}}, T_{\text{hin}})}{\max(T_{\text{eng}}, T_{\text{hin}}) + 1}$.
  - `tests/test_models.py`: Populated previously empty test suite with rigorous `BaseSentimentModel` interface compliance tests, probability bounds verification, and batch prediction validation.

---

## 2. Dependencies

- **pyproject.toml**: Added `"emoji>=2.8.0"` to declared core dependencies to ensure clean environment provisioning and eliminate manual installation requirements.

---

## 3. Readiness Conclusion

- **Ready for Evaluation Stage:** **YES**
- All mathematical formulations, architecture contracts, and baselines are validated and covered by automated tests.
