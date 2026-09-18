# Dynamic Noise-Aware Lexicon–Transformer Fusion Framework

Research-grade sentiment analysis framework for code-mixed social media text (Hinglish), combining rule-based lexicon priors (VADER) and contextual transformer representations (DistilBERT) through a dynamic, noise-aware adaptive routing mechanism.

Evaluated on the official **SemEval-2020 Task 9 / SentiMix Hindi-English** benchmark (14,000 train, 3,000 dev, 3,000 test; 3 classes: Positive=0, Negative=1, Neutral=2).

---

## 1. Mathematical Formulation & Architecture

### 1.1 Multi-Dimensional Noise Quantification Module (NQM)
- **Four-dimensional noise features (normalized to $[0, 1]$):**
  - $E$: Emoji Density
  - $R$: Character Repetition Ratio
  - $C$: Code-Mixing Intensity (Hinglish lexical overlap)
  - $S$: Special Symbol Density
- **Composite Noise Score ($N$):**
  $$N = 0.25E + 0.25R + 0.30C + 0.20S \quad (N \in [0, 1])$$

### 1.2 Noise-Aware Adaptive Router
- **Constants:** $L_0 = 20, \quad w_1 = 0.05, \quad w_2 = 12.0$
- **Routing Rule:**
  $$\alpha = \begin{cases} 0.02, & \text{if } N \le 0.20 \\ \text{clamp}(\sigma(w_1(L_0 - L) + w_2 N), 0.02, 0.25), & \text{if } N > 0.20 \end{cases}$$
- $\alpha$ dynamically balances VADER ($P_{\text{VADER}}$) and DistilBERT ($P_{\text{DistilBERT}}$) based on sentence length $L$ and noise intensity $N$. $\alpha$ is not learned via gradient descent.

### 1.3 3-Class Simplex Fusion ($\Delta^2$)
- **Fused Probability Vector:**
  $$\mathbf{P}_{\text{final}} = \alpha \cdot \mathbf{P}_{\text{VADER}} + (1 - \alpha) \cdot \mathbf{P}_{\text{DistilBERT}}$$
- **Predicted Class:**
  $$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} \mathbf{P}_{\text{final}}[c]$$
  where $0 = \text{positive}, \ 1 = \text{negative}, \ 2 = \text{neutral}$.

---

## 2. Benchmark Results on Untouched SentiMix Test Set (3,000 Samples)

All figures below are genuine empirical measurements from Phase 6 evaluation:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Pos F1 | Neg F1 | Neu F1 |
|---|---|---|---|---|---|---|---|
| **VADER (3-Class Adapter)** | 0.3753 | 0.5768 | 0.3423 | 0.2009 | 0.0598 | 0.0044 | 0.5384 |
| **DistilBERT (Fine-Tuned)** | 0.6863 | 0.6980 | 0.6906 | 0.6917 | 0.7505 | 0.7144 | 0.6102 |
| **Static Fusion ($\alpha=0.02$)** | 0.6860 | 0.6997 | 0.6894 | 0.6918 | 0.7492 | 0.7125 | 0.6137 |
| **Dynamic Hybrid Fusion (Proposed)** | **0.6883** | **0.7124** | 0.6879 | **0.6946** | 0.7404 | 0.7059 | **0.6374** |

> *Note: Dynamic Fusion achieves the highest Macro F1 and Accuracy on the benchmark, with notable gains in classifying code-mixed Neutral utterances. These results reflect empirical research performance on social media code-mixing; the model is intended for research and analytical demonstration, not safety-critical moderation.*

---

## 3. Project Structure

```
dynamic-hybrid-sentiment-fusion/
├── configs/                  # Hyperparameters and path configurations
├── src/
│   ├── api/                  # FastAPI backend service (endpoints, schemas, lifecycle)
│   ├── data/                 # SentiMix CoNLL loader and preprocessing
│   ├── features/             # Noise quantification pipeline (E, R, C, S, N)
│   ├── models/               # VADER 3-class adapter, DistilBERT 3-class, AdaptiveRouter
│   ├── evaluation/           # Phase 6 experiment runner and metrics
│   ├── pipeline.py           # Unified pipeline exports (binary + 3-class)
│   └── pipeline_3class.py    # Production SentiMix 3-class inference pipeline
├── saved_models/
│   └── sentimix_distilbert_best/  # Fine-tuned checkpoint (safetensors, config, tokenizer)
├── results/                  # Empirical evaluation CSVs, JSONs, and Markdown reports
├── frontend/                 # Explainability web application (HTML5, Vanilla CSS, JS)
├── tests/                    # 366 unit, regression, and integration tests
├── .env.example              # Example environment configuration
├── pyproject.toml            # Dependencies and packaging
└── WEBAPP_FINAL_AUDIT.md     # Production deployment and verification audit
```

---

## 4. Local Setup & Execution

### 4.1 Prerequisites & Installation
- Python 3.11 or 3.12
- PyTorch 2.0+

```bash
# Clone the repository
git clone https://github.com/techiee-jhalak/dynamic-hybrid-sentiment-fusion.git
cd dynamic-hybrid-sentiment-fusion

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 4.2 Running the Test Suite
```bash
python -m pytest -q
```

---

## 5. Starting Backend & Web Application

### 5.1 Start Backend (FastAPI + Uvicorn)
```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Upon startup, the server initializes the SentiMix 3-class DistilBERT model once in memory and mounts both API endpoints and the frontend interface.

### 5.2 Accessing the Frontend
- **Direct browser access:** Open `http://127.0.0.1:8000/` directly in any web browser.
- **Separate dev server (optional):** You can also serve `frontend/` using any static HTTP server:
  ```bash
  cd frontend
  python -m http.server 3000
  ```
  Then navigate to `http://localhost:3000/`.

---

## 6. API Endpoints & Usage

### 6.1 Health Check (`GET /health` or `GET /api/health`)
Returns service liveness, compute device, and verified checkpoint status:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "device": "cpu",
  "model_type": "SentiMix 3-Class Dynamic Hybrid Fusion",
  "checkpoint_loaded": true,
  "num_classes": 3,
  "classes": ["positive", "negative", "neutral"]
}
```

### 6.2 3-Class Sentiment Analysis (`POST /analyze` or `POST /api/analyze`)

#### Example Request:
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Yeh movie bohot achhi hai yaar! 😊🔥"}'
```

#### Example Response:
```json
{
  "text": "Yeh movie bohot achhi hai yaar! 😊🔥",
  "predicted_label": "neutral",
  "predicted_class_index": 2,
  "confidence": 0.544,
  "vader_vector": {
    "positive": 0.0,
    "negative": 0.0,
    "neutral": 1.0
  },
  "distilbert_vector": {
    "positive": 0.4214,
    "negative": 0.0438,
    "neutral": 0.5347
  },
  "fused_vector": {
    "positive": 0.413,
    "negative": 0.043,
    "neutral": 0.544
  },
  "alpha": 0.02,
  "noise": {
    "emoji_density": 0.2222,
    "repetition_score": 0.0,
    "code_mixing_ratio": 0.1667,
    "symbol_density": 0.1111,
    "composite_noise": 0.1278,
    "band": "LOW"
  },
  "explanation": "Noise score: 0.13 (LOW). The dynamic router assigned α = 0.02 to the VADER component and 1−α = 0.98 to DistilBERT."
}
```

### 6.3 Legacy Binary Endpoint (`POST /predict` or `POST /api/predict`)
Preserved for backward compatibility. Returns `sentiment`, `final_score`, `alpha`, and `noise_score`.

---

## 7. Deployment Configuration

The application is containerization-ready and supports common container/PaaS platforms (e.g., Docker, Render, Railway, Hugging Face Spaces):

1. **Environment Variables** (see `.env.example`):
   - `SENTIMIX_MODEL_PATH`: Relative or absolute path to model checkpoint (default: `saved_models/sentimix_distilbert_best`).
   - `INFERENCE_DEVICE`: `cpu` or `cuda`.
   - `CORS_ORIGINS`: Comma-separated allowed frontend origins.
   - `PORT`: Binding port (default: 8000).
2. **Production Startup Command:**
   ```bash
   gunicorn src.api.app:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
   ```
   *(Note: A single worker `-w 1` is recommended on memory-constrained servers so that transformer weights are loaded once in RAM).*
