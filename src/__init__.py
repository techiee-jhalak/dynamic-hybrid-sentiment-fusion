"""Source package root.

Lazy exports: SentimentInferencePipeline is NOT imported eagerly here
to avoid triggering torch/transformers loading at package import time.
Import directly when needed:
    from src.pipeline import SentimentInferencePipeline, PipelineResult
"""

__all__ = ["SentimentInferencePipeline", "PipelineResult"]


def __getattr__(name: str):
    if name in ("SentimentInferencePipeline", "PipelineResult"):
        from src.pipeline import SentimentInferencePipeline, PipelineResult
        globals()["SentimentInferencePipeline"] = SentimentInferencePipeline
        globals()["PipelineResult"] = PipelineResult
        return globals()[name]
    raise AttributeError(f"module 'src' has no attribute {name!r}")
