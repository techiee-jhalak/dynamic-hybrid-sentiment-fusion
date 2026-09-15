# PROJECT SPECIFICATION

## Project Title
**Dynamic Noise-Aware Lexicon–Transformer Fusion Framework for Sentiment Analysis of Code-Mixed Social Media Text**

---

## 1. Task Definition
- **Task:** Binary Sentiment Classification
- **Classes / Labels:**
  - `0`: Negative
  - `1`: Positive

---

## 2. Core Architectural Components
1. **Preprocessing Pipeline:** Text normalization while preserving noise cues (emojis, punctuation, repetition, Hinglish tokens).
2. **Multi-Dimensional Noise Quantification Module:** Extracts and normalizes structural, lexical, and orthographic noise indicators.
3. **VADER Sentiment Lexicon Model:** Rule-based lexicon model providing continuous sentiment probability/score $S_{\text{vader}} \in [0, 1]$.
4. **DistilBERT Sentiment Transformer Model:** Fine-tuned Transformer providing contextual sentiment probability $S_{\text{distilbert}} \in [0, 1]$.
5. **Noise-Aware Adaptive Routing Mechanism:** Computes dynamic weighting coefficient $\alpha \in [0.02, 0.25]$.
6. **Dynamic Fusion Layer:** Interpolates lexicon and Transformer predictions based on input noise and sequence length.
7. **Final Classification Decision:** Thresholds fused score for final binary prediction.

---

## 3. Mathematical Formulations

### 3.1 Multi-Dimensional Noise Features
All individual noise features and the composite score $N$ are normalized to $[0, 1]$:
- $E$: Emoji Density
- $R$: Character Repetition Ratio
- $C$: Code-Mixing Intensity
- $S$: Symbol Density

### 3.2 Composite Noise Score ($N$)
$$N = 0.25E + 0.25R + 0.30C + 0.20S$$
$$\text{where } N \in [0, 1]$$

### 3.3 Adaptive Routing & Dynamic Weighting ($\alpha$)
- **Reference Token Length ($L_0$):** $20$
- **Noise Threshold:** $0.20$
- **Routing Constants:**
  - $w_1 = 0.05$
  - $w_2 = 12.0$

**Routing Rule:**
- If $N \le 0.20$:
  $$\alpha = 0.02$$
- Otherwise ($N > 0.20$):
  $$z = w_1 \cdot (L_0 - L) + w_2 \cdot N$$
  $$\alpha_{\text{raw}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  $$\alpha = \text{clamp}(\alpha_{\text{raw}}, 0.02, 0.25)$$

### 3.4 Dynamic Fusion Score ($S_{\text{final}}$)
$$S_{\text{final}} = \alpha \cdot S_{\text{vader}} + (1 - \alpha) \cdot S_{\text{distilbert}}$$

### 3.5 Final Decision Rule
$$\hat{y} = \begin{cases} 1 \ (\text{Positive}), & \text{if } S_{\text{final}} \ge 0.50 \\ 0 \ (\text{Negative}), & \text{if } S_{\text{final}} < 0.50 \end{cases}$$

---

## 4. Dataset Specification
- **SAIL 2017 Hinglish Dataset:** 12,568 samples
- **Curated Benchmark Dataset:** 1,200 samples
- **Total Combined Benchmark:** 13,768 samples
- **Data Splitting Strategy:**
  - Train: 70%
  - Validation: 10%
  - Test: 20%
  - Split Method: Stratified random split

---

## 5. Model Training Configuration (DistilBERT)
- **Epochs:** 3
- **Batch Size:** 16
- **Learning Rate:** $2 \times 10^{-5}$ (`2e-5`)
- **Max Sequence Length:** $\le 128$

---

## 6. Required Baselines & Comparative Models
1. **Logistic Regression** (Classical TF-IDF / N-gram baseline)
2. **VADER** (Lexicon standalone baseline)
3. **DistilBERT** (Transformer standalone baseline)
4. **Static Fusion** (Fixed weight combination baseline)
5. **Dynamic Fusion** (Proposed noise-aware adaptive framework)
6. **BERTweet** (Optional social media domain baseline)

---

## 7. Research & Engineering Guidelines
- **No Hardcoded Results:** Reported literature metrics and paper results are reference points only and must never be hardcoded into evaluation outputs.
- **Zero Data Leakage:** Strict separation between training, validation, and test splits across all preprocessing, feature extraction, and model selection steps.
- **Formula Integrity:** Mathematical equations, constants ($L_0=20, w_1=0.05, w_2=12.0$, threshold $=0.20$, clamp bounds $[0.02, 0.25]$), and weights must not be modified without explicit instruction.
- **Feature Preservation:** Preprocessing must preserve emojis, character repetitions, punctuation, and code-mixed tokens necessary for noise quantification.
- **Modularity & Testability:** All ML, feature engineering, and inference components must be strictly modular, unit-testable, and deterministic/reproducible.
