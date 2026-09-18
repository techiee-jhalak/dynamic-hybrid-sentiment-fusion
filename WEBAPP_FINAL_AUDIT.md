# Final Web Application Audit Report — Dynamic Hybrid Sentiment Fusion

**Audit Date**: September 18, 2026  
**Repository**: `techiee-jhalak/dynamic-hybrid-sentiment-fusion`  
**Target Architecture**: SentiMix 3-Class Dynamic Noise-Aware Lexicon–Transformer Hybrid Sentiment Fusion  
**Production Status**: **READY**

---

## 1. Model & Checkpoint Verification

- **Checkpoint Path**: `saved_models/sentimix_distilbert_best/`
  - Secondary backup checkpoint: `saved_models/checkpoints/sentimix_distilbert_3class/`
  - Model weights file: `model.safetensors` (267.8 MB, verified intact)
  - Config & Tokenizer: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `training_metadata.json`
- **Model Architecture**: `DistilBertForSequenceClassification` (`distilbert-base-uncased`, `num_labels=3`)
- **Target Label Mapping**:
  - `0`: `positive`
  - `1`: `negative`
  - `2`: `neutral`
- **Model Loading Status**: **LOADED & VERIFIED**
  - Model weights and tokenizer are loaded **once** at application startup during the FastAPI `lifespan` context.
  - No per-request re-initialization occurs.
  - Supports automatic compute device resolution (`cpu` or `cuda`).
  - Tested on CPU and verified operating in evaluation mode (`torch.no_grad()`).

---

## 2. Backend & API Service

- **Framework**: FastAPI with Uvicorn / Gunicorn ASGI server
- **Endpoints Available**:
  - `GET /health` and `GET /api/health`: Service readiness, device, model architecture, target classes, and verified checkpoint status.
  - `POST /analyze` and `POST /api/analyze`: Production 3-class inference pipeline returning full vector explainability (VADER vector, DistilBERT vector, Dynamic Fused vector, noise metrics, routing weight $\alpha$, and human-readable explanation).
  - `POST /predict` and `POST /api/predict`: Legacy binary inference pathway preserved with full backward compatibility.
  - `GET /`: Direct serving of frontend `index.html`.
  - `GET /styles.css` & `GET /app.js`: Static frontend asset distribution.
- **Validation & Error Handling**:
  - Reject empty input string (`422 Unprocessable Entity`)
  - Reject whitespace-only input (`422 Unprocessable Entity`)
  - Reject text exceeding character budget of 2,048 chars (`422 Unprocessable Entity`)
  - Reject malformed or missing JSON payloads (`422 Unprocessable Entity`)
  - Catch unhandled internal exceptions globally without leaking Python stack traces to clients.
- **CORS Configuration**:
  - Read from `CORS_ORIGINS` environment variable (defaults to `http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000`).
  - Safe for local cross-origin development and production domain restrictions.

---

## 3. Frontend & Explainability Dashboard

- **Technology**: Semantic HTML5, Vanilla CSS, Vanilla JavaScript (zero external UI dependencies or heavy bundle requirements).
- **Design System**: Professional pastel visual language, clean card typography, CSS grid layouts, responsive on mobile and desktop.
- **Main User Flow**:
  - Input field with Hinglish placeholder (`Yeh movie bohot achhi hai yaar! 😊🔥`) and character counter.
  - Prominent `ANALYZE SENTIMENT` action button.
  - Loading spinner with visual feedback during inference; duplicate submissions disabled during execution.
  - Error messages cleanly rendered inline without page crashes.
- **Explainability Signals Exposed**:
  1. **Sentiment Result**: Badge prominently displays `POSITIVE`, `NEGATIVE`, or `NEUTRAL` alongside confidence percentage `XX.X%`.
  2. **Model Contributions & Vector Distributions**:
     - Final Fused Prediction ($P_{\text{final}}$): Positive, Negative, Neutral probabilities with color-coded horizontal bars.
     - DistilBERT Contextual Model ($P_{\text{DistilBERT}}$) with weight $1 - \alpha$.
     - VADER Lexicon Model ($P_{\text{VADER}}$) with weight $\alpha$.
  3. **Dynamic Routing Card**:
     - Displays exact $\alpha \in [0.02, 0.25]$ and relative percentage contributions.
     - Clarifies that $\alpha$ balances VADER and DistilBERT adaptively based on noise and sequence length (not learned via backpropagation).
  4. **Noise Analysis Card**:
     - Qualitative noise band (`LOW`, `MODERATE`, `HIGH`, `EXTREME`).
     - Composite noise score ($N$) with dynamic progress bar.
     - Four-dimensional components: Emoji Density ($E$), Repetition Score ($R$), Code-Mixing Ratio ($C$), and Symbol Density ($S$).
  5. **Router Explanation Card**:
     - Concise human-readable explanation derived directly from computed values:  
       *"Noise score: 0.13 (LOW). The dynamic router assigned α = 0.02 to the VADER component and 1−α = 0.98 to DistilBERT."*
  6. **Technical Details Accordion**:
     - Collapsible section documenting the exact mathematical equations and token count.

---

## 4. Research Integrity Audit

- **Noise Quantification Module (NQM)**: **100% UNCHANGED**
  $$E = \frac{N_{\text{emoji}}}{N_{\text{tokens}}}, \quad R = \frac{N_{\text{repeat}}}{N_{\text{tokens}}}, \quad C = \frac{\min(\text{Eng}, \text{Hin})}{\max(\text{Eng}, \text{Hin}) + 1}, \quad S = \frac{N_{\text{symbol}}}{N_{\text{tokens}}}$$
  $$N = 0.25E + 0.25R + 0.30C + 0.20S$$
- **AdaptiveRouter Constants & Formulas**: **100% UNCHANGED**
  $$L_0 = 20, \quad w_1 = 0.05, \quad w_2 = 12.0$$
  $$\alpha = 0.02 \quad \text{if } N \le 0.20, \quad \text{else } \alpha = \text{clamp}(\sigma(0.05(20 - L) + 12.0N), 0.02, 0.25)$$
- **3-Class Simplex Fusion**: **100% PRESERVED**
  $$\mathbf{P}_{\text{final}} = \alpha \mathbf{P}_{\text{VADER}} + (1 - \alpha) \mathbf{P}_{\text{DistilBERT}}, \quad \hat{y} = \arg\max \mathbf{P}_{\text{final}}$$
- **Metric Provenance**:
  - No synthetic predictions or fabricated evaluation metrics.
  - All test results originate from real inference over the 3,000-sample test partition of SemEval-2020 Task 9 (SentiMix Hinglish).
  - Accuracy: `68.83%` | Macro F1: `0.6946` | Neutral F1: `0.6374`.

---

## 5. Automated Testing & End-to-End Verification

- **Full pytest Suite**:
  - Total Tests: **366**
  - Passed: **366**
  - Failed: **0**
  - Regressions: **0**
- **Test Modules**:
  - `tests/test_api_3class.py`: 12 integration and mathematical invariant tests.
  - `tests/test_api.py`: 11 route and error handling tests.
  - `tests/test_pipeline_3class.py`: 3 end-to-end 3-class pipeline tests.
  - `tests/test_models.py`, `tests/test_features.py`, `tests/test_data.py`: Unit tests across all core modules.
- **End-to-End Browser Verification**:
  - Real FastAPI server launched on `http://127.0.0.1:8000/`.
  - Browser agent navigated to the web app, entered `"Yeh movie bohot achhi hai yaar! 😊🔥"`, triggered `ANALYZE SENTIMENT`.
  - Verified full UI rendering:
    - Sentiment Badge: `NEUTRAL`
    - Confidence: `52.6%`
    - Model Contributions: Fused (`41.6%` Pos, `5.8%` Neg, `52.6%` Neu), DistilBERT (`42.4%` Pos, `6.0%` Neg, `51.6%` Neu), VADER (`0.0%` Pos, `0.0%` Neg, `100.0%` Neu)
    - Routing $\alpha$: `0.020` (DistilBERT weight: `0.980`)
    - Composite Noise $N$: `0.079` (`LOW`)
    - Router explanation and technical details populated correctly.
  - Captured verification recording and screenshot.

---

## 6. Deployment Readiness & Environment Specification

- **Environment File**: `.env.example` created with all configurable parameters:
  - `SENTIMIX_MODEL_PATH`: Location of the trained model directory.
  - `INFERENCE_DEVICE`: `cpu` or `cuda`.
  - `CORS_ORIGINS`: Comma-separated allowed web domains.
  - `API_MAX_TEXT_CHARS`: Maximum input length (default: 2048).
- **Backend Startup Command**:
  ```bash
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
  ```
- **Production ASGI Command**:
  ```bash
  gunicorn src.api.app:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
  ```
- **Frontend Access**: Available directly at `http://localhost:8000/` or via any static file host.
