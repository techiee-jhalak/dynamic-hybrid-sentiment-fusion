# Dynamic Noise-Aware Lexicon–Transformer Fusion Framework

Research-grade sentiment analysis framework for code-mixed social media text (Hinglish), combining rule-based lexicon signals (VADER) and contextual representations (DistilBERT) through a dynamic, noise-aware adaptive routing mechanism.

---

## 1. Mathematical Formulation

### 1.1 Multi-Dimensional Noise Quantification
- **Noise Features (normalized to $[0, 1]$):**
  - $E$: Emoji Density
  - $R$: Character Repetition Ratio
  - $C$: Code-Mixing Intensity
  - $S$: Symbol Density
- **Composite Noise Score ($N$):**
  $$N = 0.25E + 0.25R + 0.30C + 0.20S \quad (N \in [0, 1])$$

### 1.2 Noise-Aware Adaptive Routing
- **Constants:** $L_0 = 20, \quad w_1 = 0.05, \quad w_2 = 12.0$
- **Routing Rule:**
  $$\alpha = \begin{cases} 0.02, & \text{if } N \le 0.20 \\ \text{clamp}(\sigma(w_1(L_0 - L) + w_2 N), 0.02, 0.25), & \text{if } N > 0.20 \end{cases}$$

### 1.3 Dynamic Fusion & Classification
- **Fusion Score:**
  $$S_{\text{final}} = \alpha \cdot S_{\text{vader}} + (1 - \alpha) \cdot S_{\text{distilbert}}$$
- **Decision:**
  $$\hat{y} = \begin{cases} 1 \ (\text{Positive}), & \text{if } S_{\text{final}} \ge 0.50 \\ 0 \ (\text{Negative}), & \text{if } S_{\text{final}} < 0.50 \end{cases}$$

---

## 2. Project Structure

```
dynamic-hybrid-sentiment-fusion/
├── configs/
│   ├── config.py             # Central dataclass configuration
│   └── config.yaml           # YAML configuration
├── src/
│   ├── data/                 # Ingestion, preprocessing, and stratified splits
│   ├── features/             # Noise quantification (E, R, C, S -> N)
│   ├── models/               # VADER, DistilBERT, AdaptiveRouter, DynamicFusion
│   ├── evaluation/           # Metric computation (Accuracy, F1, Confusion Matrix)
│   └── api/                  # FastAPI backend service
├── tests/                    # Unit and integration test suite
├── pyproject.toml            # Project dependencies and packaging
├── verify_env.py             # Environment verification script
└── PROJECT_SPEC.md           # Research and system specification
```

---

## 3. Environment Setup

### 3.1 Requirements
- Python 3.11+
- Core ML Stack:
  - `numpy`, `pandas`, `scikit-learn`, `nltk`
  - `torch`, `transformers`, `datasets`, `accelerate`

### 3.2 Verification
Run the environment verification script:
```bash
python verify_env.py
```
