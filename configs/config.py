"""Centralized Configuration Module.

Stores all system hyperparameters, model settings, noise feature weights,
routing constants, dataset paths, and device selection.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class NoiseWeights:
    w_emoji: float = 0.25      # E
    w_repetition: float = 0.25 # R
    w_code_mix: float = 0.30   # C
    w_symbol: float = 0.20     # S


@dataclass(frozen=True)
class RoutingConfig:
    reference_length: int = 20     # L0
    noise_threshold: float = 0.20  # Noise threshold
    w1: float = 0.05               # Length sensitivity weight
    w2: float = 12.0               # Noise sensitivity weight
    alpha_min: float = 0.02        # Lower clamp bound
    alpha_max: float = 0.25        # Upper clamp bound
    decision_threshold: float = 0.50


@dataclass(frozen=True)
class ModelTrainingConfig:
    model_name: str = "distilbert-base-uncased"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 128
    train_split: float = 0.70
    val_split: float = 0.10
    test_split: float = 0.20
    random_seed: int = 42


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    saved_models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "saved_models")


@dataclass
class AppConfig:
    noise: NoiseWeights = field(default_factory=NoiseWeights)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    training: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


config = AppConfig()
