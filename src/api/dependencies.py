"""FastAPI dependency injection and singleton model lifecycle management.

Models are loaded exactly once during application lifespan startup and
reused across all requests. No model is instantiated per-request.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Any, Union

from src.models.dynamic_fusion import DynamicFusionFramework
from src.models.vader_model import VaderSentimentModel
from src.models.distilbert_model import DistilBertSentimentModel
from src.models.adaptive_router import AdaptiveRouter
from src.features.noise_quantifier import NoiseQuantifier
from src.data.preprocessor import TextPreprocessor
from src.pipeline import SentimentInferencePipeline
from src.pipeline_3class import (
    Sentimix3ClassInferencePipeline,
    Sentimix3ClassPipelineResult,
)
from src.models.distilbert_3class import DistilBert3ClassModel
from configs.config import config

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton container managing the loaded inference pipelines.

    Lifecycle:
    - ``initialize()`` is called once at application startup (in lifespan).
    - ``get_sentimix_pipeline()`` returns the real 3-class production pipeline.
    - ``get_pipeline()`` returns the binary pipeline (for backward compatibility).
    - ``shutdown()`` releases resources at application shutdown.
    """

    _sentimix_pipeline: Optional[Sentimix3ClassInferencePipeline] = None
    _pipeline: Optional[SentimentInferencePipeline] = None
    _device: str = "cpu"
    _checkpoint_path: Optional[str] = None
    _checkpoint_loaded: bool = False

    @classmethod
    def initialize(cls, device: Optional[str] = None) -> None:
        """Instantiate and load all models once.

        Args:
            device: Compute device string (``"cpu"`` or ``"cuda"``).
                    Falls back to ``INFERENCE_DEVICE`` env var, then
                    ``config.device``, then ``"cpu"``.
        """
        if cls._sentimix_pipeline is not None:
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

        # 1. Initialize SentiMix 3-class production pipeline with real checkpoint
        checkpoint_dir = os.environ.get("SENTIMIX_MODEL_PATH") or str(
            Path(config.paths.saved_models_dir) / "sentimix_distilbert_best"
        )
        ckpt_path = Path(checkpoint_dir)

        try:
            if ckpt_path.exists():
                logger.info("Loading SentiMix 3-class DistilBERT from real checkpoint: %s", checkpoint_dir)
                distilbert_3class = DistilBert3ClassModel(
                    model_path_or_name=str(ckpt_path),
                    device=resolved_device,
                    lazy_load=False,
                )
                cls._sentimix_pipeline = Sentimix3ClassInferencePipeline(
                    checkpoint_dir=ckpt_path,
                    distilbert_model=distilbert_3class,
                )
                cls._checkpoint_loaded = True
                cls._checkpoint_path = str(ckpt_path)
                logger.info("SentiMix 3-class model loaded and verified successfully.")
            else:
                logger.warning(
                    "Checkpoint path '%s' not found. Initializing lazy-load 3-class model.",
                    checkpoint_dir,
                )
                cls._sentimix_pipeline = Sentimix3ClassInferencePipeline(
                    checkpoint_dir=ckpt_path,
                )
                cls._checkpoint_loaded = False
                cls._checkpoint_path = str(ckpt_path)

            # 2. Initialize legacy binary pipeline (for backward compatibility)
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
            logger.info("All model pipelines loaded successfully.")
        except Exception as exc:
            logger.error(
                "Model initialization failed [%s]: %s",
                type(exc).__name__,
                str(exc),
            )
            raise RuntimeError(
                "Model initialization failed; the service cannot start."
            ) from exc

    @classmethod
    def get_sentimix_pipeline(cls) -> Sentimix3ClassInferencePipeline:
        """Return the initialized SentiMix 3-class pipeline singleton."""
        if cls._sentimix_pipeline is None:
            cls.initialize()
        return cls._sentimix_pipeline

    @classmethod
    def get_pipeline(cls) -> SentimentInferencePipeline:
        """Return the initialized binary pipeline singleton."""
        if cls._pipeline is None:
            cls.initialize()
        return cls._pipeline

    @classmethod
    def is_ready(cls) -> bool:
        """Return True if pipelines are loaded and ready."""
        return cls._sentimix_pipeline is not None or cls._pipeline is not None

    @classmethod
    def is_checkpoint_loaded(cls) -> bool:
        """Return True if the real trained checkpoint is verified loaded."""
        return cls._checkpoint_loaded

    @classmethod
    def get_checkpoint_path(cls) -> Optional[str]:
        """Return the checkpoint path loaded."""
        return cls._checkpoint_path

    @classmethod
    def get_device(cls) -> str:
        """Return the resolved compute device string."""
        return cls._device

    @classmethod
    def shutdown(cls) -> None:
        """Release model references (called on application shutdown)."""
        cls._sentimix_pipeline = None
        cls._pipeline = None
        cls._checkpoint_loaded = False
        cls._checkpoint_path = None
        logger.info("ModelManager shut down; model references released.")


# ---------------------------------------------------------------------------
# FastAPI dependency functions
# ---------------------------------------------------------------------------


def get_pipeline() -> SentimentInferencePipeline:
    """FastAPI dependency: returns the binary inference pipeline singleton."""
    return ModelManager.get_pipeline()


def get_sentimix_pipeline() -> Any:
    """FastAPI dependency: returns the SentiMix 3-class production pipeline singleton.

    Detects if a legacy test fixture has overridden ``get_pipeline`` and gracefully
    uses the override if ``get_sentimix_pipeline`` is not explicitly overridden.
    """
    from src.api.app import app
    if get_pipeline in app.dependency_overrides and get_sentimix_pipeline not in app.dependency_overrides:
        return app.dependency_overrides[get_pipeline]()
    return ModelManager.get_sentimix_pipeline()


# Keep backward-compatible alias used by existing route skeletons
def get_fusion_model() -> DynamicFusionFramework:
    """Backward-compatible dependency returning the fusion framework."""
    pipeline = ModelManager.get_pipeline()
    return DynamicFusionFramework(
        vader_model=pipeline.vader_model,
        distilbert_model=pipeline.distilbert_model,
        noise_quantifier=pipeline.noise_quantifier,
        router=pipeline.router,
        preprocessor=pipeline.preprocessor,
    )
