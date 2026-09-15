"""FastAPI dependency injection and singleton model lifecycle management.

Models are loaded exactly once during application lifespan startup and
reused across all requests. No model is instantiated per-request.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.models.dynamic_fusion import DynamicFusionFramework
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel
from src.models.adaptive_router import AdaptiveRouter
from src.features.noise_quantifier import NoiseQuantifier
from src.data.preprocessor import TextPreprocessor
from src.pipeline import SentimentInferencePipeline
from configs.config import config

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton container managing the loaded inference pipeline.

    Lifecycle:
    - ``initialize()`` is called once at application startup (in lifespan).
    - ``get_pipeline()`` is used by FastAPI dependencies on every request.
    - ``shutdown()`` releases resources at application shutdown.
    """

    _pipeline: Optional[SentimentInferencePipeline] = None
    _device: str = "cpu"

    @classmethod
    def initialize(cls, device: Optional[str] = None) -> None:
        """Instantiate and load all models once.

        Args:
            device: Compute device string (``"cpu"`` or ``"cuda"``).
                    Falls back to ``INFERENCE_DEVICE`` env var, then
                    ``config.device``, then ``"cpu"``.
        """
        if cls._pipeline is not None:
            logger.debug("ModelManager.initialize called but models already loaded; skipping.")
            return

        resolved_device = (
            device
            or os.environ.get("INFERENCE_DEVICE")
            or config.device
            or "cpu"
        )
        cls._device = resolved_device
        logger.info("Initializing models on device: %s", resolved_device)

        try:
            preprocessor = TextPreprocessor()
            noise_quantifier = NoiseQuantifier()
            vader_model = VaderSentimentModel()
            distilbert_model = DistilBertSentimentModel(device=resolved_device)
            router = AdaptiveRouter()

            cls._pipeline = SentimentInferencePipeline(
                preprocessor=preprocessor,
                noise_quantifier=noise_quantifier,
                vader_model=vader_model,
                distilbert_model=distilbert_model,
                router=router,
            )
            logger.info("All models loaded successfully.")
        except Exception as exc:
            # Log the error type but not the full traceback to avoid leaking paths
            logger.error(
                "Model initialization failed [%s]: %s",
                type(exc).__name__,
                str(exc),
            )
            raise RuntimeError(
                "Model initialization failed; the service cannot start."
            ) from exc

    @classmethod
    def get_pipeline(cls) -> SentimentInferencePipeline:
        """Return the initialized pipeline singleton.

        Raises:
            RuntimeError: if ``initialize()`` was not called first.
        """
        if cls._pipeline is None:
            raise RuntimeError(
                "Inference pipeline has not been initialized. "
                "Call ModelManager.initialize() during application startup."
            )
        return cls._pipeline

    @classmethod
    def is_ready(cls) -> bool:
        """Return True if the pipeline is loaded and ready."""
        return cls._pipeline is not None

    @classmethod
    def get_device(cls) -> str:
        """Return the resolved compute device string."""
        return cls._device

    @classmethod
    def shutdown(cls) -> None:
        """Release model references (called on application shutdown)."""
        cls._pipeline = None
        logger.info("ModelManager shut down; model references released.")


# ---------------------------------------------------------------------------
# FastAPI dependency functions
# ---------------------------------------------------------------------------


def get_pipeline() -> SentimentInferencePipeline:
    """FastAPI dependency: returns the shared inference pipeline singleton."""
    return ModelManager.get_pipeline()


# Keep backward-compatible alias used by existing route skeletons
def get_fusion_model() -> DynamicFusionFramework:
    """Backward-compatible dependency returning the fusion framework.

    Note: the primary dependency is ``get_pipeline()``. This alias delegates
    to the pipeline's internal framework for callers that require a
    ``DynamicFusionFramework`` instance directly.
    """
    pipeline = ModelManager.get_pipeline()
    # The pipeline owns the distilbert model; we can surface the fusion
    # framework by wrapping the pipeline's components — but to avoid
    # duplicate object construction we build a thin wrapper on demand.
    # For new code, prefer get_pipeline().
    return DynamicFusionFramework(
        vader_model=pipeline.vader_model,
        distilbert_model=pipeline.distilbert_model,
        noise_quantifier=pipeline.noise_quantifier,
        router=pipeline.router,
        preprocessor=pipeline.preprocessor,
    )
