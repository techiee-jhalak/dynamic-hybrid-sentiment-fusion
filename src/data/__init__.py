"""Data handling package.

Lazy exports: heavy modules (DatasetPipeline, pandas, sklearn) are NOT imported
at package level to avoid adding 7+ seconds to application startup.
Import them directly from their submodules when needed:
    from src.data.dataset import DatasetPipeline, LABEL_MAPPING
    from src.data.preprocessor import TextPreprocessor, PreprocessingResult
"""

__all__ = ["DatasetPipeline", "LABEL_MAPPING", "TextPreprocessor", "PreprocessingResult"]


def __getattr__(name: str):
    """Lazy attribute loader — only imports when the symbol is actually accessed."""
    if name in ("DatasetPipeline", "LABEL_MAPPING"):
        from src.data.dataset import DatasetPipeline, LABEL_MAPPING
        globals()["DatasetPipeline"] = DatasetPipeline
        globals()["LABEL_MAPPING"] = LABEL_MAPPING
        return globals()[name]
    if name in ("TextPreprocessor", "PreprocessingResult"):
        from src.data.preprocessor import TextPreprocessor, PreprocessingResult
        globals()["TextPreprocessor"] = TextPreprocessor
        globals()["PreprocessingResult"] = PreprocessingResult
        return globals()[name]
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
