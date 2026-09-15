"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load models once on startup."""
    # Initialization placeholder
    yield
    # Cleanup placeholder


app = FastAPI(
    title="Dynamic Noise-Aware Lexicon-Transformer Sentiment Fusion API",
    description="Research-grade sentiment analysis system for code-mixed social media text.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for decoupled frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
