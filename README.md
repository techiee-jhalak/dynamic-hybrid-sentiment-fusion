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
├── configs/                  # Global system configuration
├── src/
│   ├── data/                 # Ingestion and preprocessing
│   ├── features/             # Noise quantification pipeline
│   ├── models/               # Architecture (VADER, DistilBERT, Routing)
│   ├── evaluation/           # Evaluation and testing suite
│   └── api/                  # FastAPI backend service
├── frontend/                 # UI dashboard for model explainability
├── tests/                    # Unit and integration test suite
├── pyproject.toml            # Project dependencies and packaging
├── PROJECT_SPEC.md           # Research and system specification
└── FINAL_AUDIT.md            # Final implementation verification status
```

---

## 3. Environment Setup & Testing

### 3.1 Requirements
- Python 3.11+
- Install backend and development dependencies via `pyproject.toml`.

```bash
pip install -e ".[dev]"
```

### 3.2 Verification & Testing
Run the complete unit and integration test suite (ensures research integrity without running full model training):
```bash
python -m pytest -q
```

---

## 4. API & Frontend

### FastAPI Backend
The project exposes a highly structured API for predictions and explainability.

Start the backend:
```bash
python -m uvicorn src.api.app:app --reload
```
- `GET /api/health`: Service readiness
- `POST /api/predict`: Returns minimal prediction parameters.
- `POST /api/analyze`: Returns the full component decomposition (E, R, C, S, N, router trace).

### Explainability UI
A professional, minimal HTML5/JS frontend is provided to visualize the adaptive fusion mechanics.
- Open `frontend/index.html` in any browser to query the running API.

---

## 5. Experiment Status

The underlying architecture for extracting statistical evaluations (Accuracy, Precision, Recall, McNemar's Test, Error Analysis) is fully implemented in `src/evaluation`. The experimental suite is designed to parse real model weights, and does not invent mock metric strings when the dataset/checkpoint is unavailable. 

> *Note: Model training and evaluation over the raw sentiment dataset are configured but marked as `NOT RUN` pending actual dataset population and DistilBERT fine-tuning execution.*
