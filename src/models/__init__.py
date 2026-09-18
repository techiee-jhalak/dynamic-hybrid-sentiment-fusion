"""Models package.

Lazy exports: heavy ML models (DistilBERT, baselines) are NOT imported at
package level to avoid torch/transformers loading during application startup.
Import directly from submodules when needed:
    from src.models.distilbert_model import DistilBertSentimentModel
    from src.models.baselines import LogisticRegressionBaseline
"""

__all__ = [
    "BaseSentimentModel",
    "VaderSentimentModel",
    "VaderSentimentOutput",
    "DistilBertSentimentModel",
    "DistilBertSentimentOutput",
    "ModelLoader",
    "AdaptiveRouter",
    "RoutingDecision",
    "stable_sigmoid",
    "DynamicFusionFramework",
    "FusionResult",
    "fuse_scores",
    "LogisticRegressionBaseline",
    "StaticFusionBaseline",
    "BERTweetBaseline",
]

_LIGHTWEIGHT = (
    "BaseSentimentModel",
    "VaderSentimentModel", "VaderSentimentOutput",
    "AdaptiveRouter", "RoutingDecision", "stable_sigmoid",
    "DynamicFusionFramework", "FusionResult", "fuse_scores",
)

_HEAVY = (
    "DistilBertSentimentModel", "DistilBertSentimentOutput", "ModelLoader",
    "LogisticRegressionBaseline", "StaticFusionBaseline", "BERTweetBaseline",
)


def __getattr__(name: str):
    if name == "BaseSentimentModel":
        from src.models.base import BaseSentimentModel
        globals()[name] = BaseSentimentModel
        return BaseSentimentModel
    if name in ("VaderSentimentModel", "VaderSentimentOutput"):
        from src.models.vader_model import VaderSentimentModel, VaderSentimentOutput
        globals()["VaderSentimentModel"] = VaderSentimentModel
        globals()["VaderSentimentOutput"] = VaderSentimentOutput
        return globals()[name]
    if name in ("AdaptiveRouter", "RoutingDecision", "stable_sigmoid"):
        from src.models.adaptive_router import AdaptiveRouter, RoutingDecision, stable_sigmoid
        globals()["AdaptiveRouter"] = AdaptiveRouter
        globals()["RoutingDecision"] = RoutingDecision
        globals()["stable_sigmoid"] = stable_sigmoid
        return globals()[name]
    if name in ("DynamicFusionFramework", "FusionResult", "fuse_scores"):
        from src.models.dynamic_fusion import DynamicFusionFramework, FusionResult, fuse_scores
        globals()["DynamicFusionFramework"] = DynamicFusionFramework
        globals()["FusionResult"] = FusionResult
        globals()["fuse_scores"] = fuse_scores
        return globals()[name]
    if name in ("DistilBertSentimentModel", "DistilBertSentimentOutput", "ModelLoader"):
        from src.models.distilbert_model import DistilBertSentimentModel, DistilBertSentimentOutput, ModelLoader
        globals()["DistilBertSentimentModel"] = DistilBertSentimentModel
        globals()["DistilBertSentimentOutput"] = DistilBertSentimentOutput
        globals()["ModelLoader"] = ModelLoader
        return globals()[name]
    if name in ("LogisticRegressionBaseline", "StaticFusionBaseline", "BERTweetBaseline"):
        from src.models.baselines import LogisticRegressionBaseline, StaticFusionBaseline, BERTweetBaseline
        globals()["LogisticRegressionBaseline"] = LogisticRegressionBaseline
        globals()["StaticFusionBaseline"] = StaticFusionBaseline
        globals()["BERTweetBaseline"] = BERTweetBaseline
        return globals()[name]
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
