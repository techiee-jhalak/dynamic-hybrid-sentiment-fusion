# Dynamic Hybrid Sentiment Fusion — Railway Deployment
# Python 3.12 slim; CPU-only PyTorch; trained checkpoint baked into image.
#
# Build context must include:
#   requirements.txt
#   src/
#   configs/
#   frontend/
#   saved_models/sentimix_distilbert_best/   ← baked in; NOT in .dockerignore
#
# Railway injects $PORT at runtime (default 8000 if unset).

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1  : required by PyTorch CPU kernels (OpenMP runtime)
# curl      : optional healthcheck helper; negligible image cost
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
 && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Step 1: upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Step 2: CPU-only PyTorch FIRST via the dedicated whl index.
# Installing torch separately before -r requirements.txt prevents pip from
# resolving a CUDA wheel when it sees torch==2.14.0 without the +cpu suffix
# (requirements.txt already pins +cpu, but an explicit pre-install is safer).
RUN pip install --no-cache-dir \
        torch==2.14.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Step 3: remaining dependencies (requirements.txt also has --extra-index-url
# pointing at the CPU whl index, so torch is never upgraded to a CUDA build).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY src/       src/
COPY configs/   configs/
COPY frontend/  frontend/

# ── Trained model checkpoint (baked into image — no runtime Hub download) ────
# ModelManager._checkpoint_is_complete() looks for:
#   config.json, tokenizer_config.json, tokenizer.json, model.safetensors
# All four files are present in saved_models/sentimix_distilbert_best/.
COPY saved_models/ saved_models/

# ── Runtime ───────────────────────────────────────────────────────────────────
# Railway sets $PORT; fall back to 8000 for local `docker run` without -e PORT.
# One worker only (memory constraint on free tier).
ENV PORT=8000
EXPOSE ${PORT}

CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
