"""FastAPI dependency injection and singleton model management.

Ensures models are loaded once during application startup and reused across requests.
"""

from typing import Optional
from src.models.dynamic_fusion import DynamicFusionFramework


class ModelManager:
    """Singleton container managing loaded model instances."""

    _fusion_engine: Optional[DynamicFusionFramework] = None

    @classmethod
    def initialize(cls) -> None:
        """Instantiate and load all models once."""
        raise NotImplementedError

    @classmethod
    def get_fusion_engine(cls) -> DynamicFusionFramework:
        """Retrieve singleton dynamic fusion engine."""
        if cls._fusion_engine is None:
            raise RuntimeError("Models have not been initialized.")
        return cls._fusion_engine


def get_fusion_model() -> DynamicFusionFramework:
    """FastAPI dependency for accessing the initialized fusion framework."""
    return ModelManager.get_fusion_engine()
