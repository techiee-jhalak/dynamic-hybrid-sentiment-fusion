# FRONTEND AUDIT — Dynamic Hybrid Sentiment Fusion

> **Status:** Professional UI integrated with existing FastAPI backend endpoints successfully.

## 1. UI Sections Implemented

- **Header**: Minimal project branding with subtitle establishing the research context (Code-Mixed Social Media Text).
- **Text Input**: Responsive standard `textarea` restricted to `2048` characters (consistent with backend Pydantic validation limit) with real-time char counting.
- **Main Result (Core Result)**: Large badge and numerical display showing the fused sentiment classification (Positive / Negative) and Final Fusion Score.
- **Dynamic Fusion Explanation**: Extracted component probabilities (`VADER Score`, `DistilBERT Score`, and the dynamic weighting `α`) are mapped cleanly from the backend JSON response to display the interpolation variables natively.
- **Noise Analysis**:
  - `E`, `R`, `C`, `S` displayed in a compact format.
  - Overall `Composite Noise (N)` is shown alongside a visual progress bar tracking magnitude.
  - Noise Band (LOW / MODERATE / HIGH / EXTREME) colored tags dynamically styled on classification.
- **Router Explanation**: Shows the deterministic backend-generated rule/decision string explaining *why* α was chosen, completely sourced from `/analyze`.
- **Technical Details**: An expandable `<details>` accordion block that displays the underlying formulas and rules without cluttering the initial load.

## 2. API Integration

- Integration is built cleanly inside `frontend/app.js` using standard `fetch`.
- Endpoint used: `POST /api/analyze`.
- Extensibility: Environment override uses a simple short-circuit: `window.APP_CONFIG?.API_BASE_URL || 'http://localhost:8000/api'`.
- Logic Seam: No math or ML logic is recalculated natively in the frontend. All values, badges, and texts (even router explanation texts) are parsed directly from the `/analyze` structured schema response to guarantee formula adherence.

## 3. Responsiveness & UI/UX Design

- **Grid Architecture**: `explainability-grid` is split seamlessly from `2` columns to `1` on `max-width: 768px`.
- **Visuals**: Hand-picked professional pastel colors (indigo, slate, emerald, amber) utilizing CSS custom properties for uniform `border-radius`, `box-shadow`, and `padding`.
- **Animations**: Limited to state changes (buttons, borders, and a minimal CSS `spin` on the loading indicator) for performance and professionalism.
- **Error/Loading States**:
  - Button disabling on `analyze` prevents duplicate requests.
  - Spinner injected while awaiting Fetch.
  - Safe extraction of `data.detail` intercepts structural FastAPI `HTTP 422` validation rejections directly to UI.
  - Network disconnection `Failed to fetch` safely abstracted inside `try/catch`.

## 4. Accessibility

- Implemented semantic markup (`<main>`, `<header>`, `<section>`, `<details>`).
- Enforced `aria-label` / `aria-live` flags on changing states (character count, text area, error states).
- Colors adhere to standard contrast ratio checks natively via the slate-surface dark-text contrasts.

## 5. Research Integrity

- **Zero Fabrication**: All results shown are directly populated from API outputs.
- Metrics are cleanly surfaced using `formatNumber(x, 3)` to cap decimal inflation, preserving authenticity without obfuscating precision.

## 6. Validation / Test Results

- All `JavaScript` functionality is strictly `vanilla` without errors or missing polyfills.
- The Python integration suite `python -m pytest -q` was run on the backend to guarantee ML integrations were not mutated by the frontend construction logic.
- Total Backend Result: **267 passed, 1 skipped** (`test_api.py` import hook), 0 failures.

## 7. Remaining Limitations

- **Model Execution Cost**: The frontend can still suffer UI latency if the backend hasn't warmed up the `DistilBERT` transformer instance.
- **Batch Processing**: The current UI does not allow bulk file upload / batch inference processing, as it is scoped to a single `textarea` input demonstration.
