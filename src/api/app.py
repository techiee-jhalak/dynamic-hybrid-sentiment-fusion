"""FastAPI application entry point for the Dynamic Hybrid Sentiment Fusion API.

Model lifecycle
---------------
All ML models (VADER, DistilBERT, router, preprocessor) are loaded once
during the ``lifespan`` startup phase via ``ModelManager.initialize()``.
They are never reloaded per-request.

CORS
----
Allowed origins are read from the ``CORS_ORIGINS`` environment variable
(comma-separated list). When the variable is unset the server defaults to
``http://localhost:3000`` (typical React/Vite dev server) so that the
existing frontend works out of the box without hardcoding broad access.

Set ``CORS_ORIGINS=*`` only for local development; in production supply
the exact frontend domain.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import ModelManager
from src.api.routes import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CORS configuration from environment
# ---------------------------------------------------------------------------

def _get_cors_origins() -> List[str]:
    """Read allowed origins from ``CORS_ORIGINS`` environment variable.

    Returns a list of origin strings. Falls back to localhost:3000 when
    the variable is not set so the dev frontend works without configuration.
    """
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins if origins else ["http://localhost:3000"]


# ---------------------------------------------------------------------------
# Application lifespan — load models once on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all ML models once at startup; release on shutdown."""
    import sys, time as _time

    _t0 = _time.perf_counter()
    logger.info("[STARTUP] Creating application — Python %s", sys.version.split()[0])

    try:
        import torch as _torch
        logger.info("[STARTUP] torch %s loaded (CUDA available: %s)",
                    _torch.__version__, _torch.cuda.is_available())
    except Exception:
        logger.info("[STARTUP] torch import deferred")

    logger.info("[STARTUP] Initializing model manager")
    _t1 = _time.perf_counter()

    try:
        ModelManager.initialize()
    except RuntimeError as exc:
        logger.critical("[STARTUP] FAILED after %.1fs — %s", _time.perf_counter() - _t0, exc)
        raise

    _t2 = _time.perf_counter()
    logger.info("[STARTUP] Model loaded in %.1fs (checkpoint: %s)",
                _t2 - _t1, ModelManager.get_checkpoint_path() or "none")
    logger.info("[STARTUP] Application ready — total startup %.1fs", _t2 - _t0)

    yield  # application runs

    logger.info("Application shutdown: releasing model references.")
    ModelManager.shutdown()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dynamic Noise-Aware Lexicon-Transformer Sentiment Fusion API",
    description=(
        "Research-grade binary sentiment analysis for code-mixed social media text. "
        "Implements the Dynamic Hybrid Fusion framework: "
        "VADER + DistilBERT with noise-aware adaptive routing (α ∈ [0.02, 0.25]).\n\n"
        "**Endpoints**\n"
        "- `GET /api/health` — liveness / readiness check\n"
        "- `POST /api/predict` — minimal prediction (sentiment, final_score, α, N)\n"
        "- `POST /api/analyze` — full explainability breakdown\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_cors_origins = _get_cors_origins()
logger.info("CORS allowed origins: %s", _cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)


# ---------------------------------------------------------------------------
# Global exception handler — prevent stack traces leaking to clients
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that returns a structured JSON error without exposing internals."""
    logger.error(
        "Unhandled exception on %s %s [%s]: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
        str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "code": "INTERNAL_ERROR"},
    )


from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# Router registration (both with /api prefix and root for full flexibility)
# ---------------------------------------------------------------------------

app.include_router(router, prefix="/api")
app.include_router(router)

# ---------------------------------------------------------------------------
# Frontend static asset serving
# ---------------------------------------------------------------------------

frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(frontend_dir / "index.html")

    @app.get("/styles.css", include_in_schema=False)
    async def serve_css():
        return FileResponse(frontend_dir / "styles.css")

    @app.get("/app.js", include_in_schema=False)
    async def serve_js():
        return FileResponse(frontend_dir / "app.js")
