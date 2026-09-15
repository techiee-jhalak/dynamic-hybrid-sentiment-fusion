"""Noise-Aware Adaptive Routing Module.

Computes the dynamic fusion weighting coefficient alpha in [0.02, 0.25] based on
the input composite noise score N in [0, 1] and token length L.

Exact Mathematical Specification:
- Reference Token Length: L0 = 20
- Noise Threshold: 0.20
- Constants: w1 = 0.05, w2 = 12.0
- Clamping Bounds: alpha_min = 0.02, alpha_max = 0.25

Routing Logic:
If N <= 0.20:
    alpha = 0.02
Otherwise (N > 0.20):
    z = w1 * (L0 - L) + w2 * N
    alpha_raw = sigmoid(z)
    alpha = clamp(alpha_raw, 0.02, 0.25)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from configs.config import RoutingConfig, config


def stable_sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid function: sigma(z) = 1 / (1 + exp(-z))."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


@dataclass(frozen=True)
class RoutingDecision:
    """Structured container for adaptive routing output."""
    alpha: float              # Final clamped alpha in [0.02, 0.25]
    alpha_raw: float          # Raw sigmoid(z) value in (0, 1)
    z_score: float            # Intermediate linear combination z
    noise_score: float        # Input composite noise N in [0, 1]
    token_length: int         # Input token count L
    routing_state: str        # State: 'low_noise_default', 'adaptive_active', 'clamped_max', 'clamped_min'

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "alpha": self.alpha,
            "alpha_raw": self.alpha_raw,
            "z_score": self.z_score,
            "noise_score": self.noise_score,
            "token_length": self.token_length,
            "routing_state": self.routing_state,
        }


class AdaptiveRouter:
    """Calculates adaptive lexicon weighting coefficient alpha based on noise and token length."""

    def __init__(self, routing_config: RoutingConfig = config.routing) -> None:
        self.cfg = routing_config

    def route(self, noise_score: float, token_length: int) -> RoutingDecision:
        """Perform deterministic adaptive routing returning structured decision and alpha."""
        n = float(noise_score)
        l = int(max(0, token_length))

        # Linear combination z = w1 * (L0 - L) + w2 * N
        z = self.cfg.w1 * (self.cfg.reference_length - l) + self.cfg.w2 * n
        alpha_raw = stable_sigmoid(z)

        # Exact conditional branching
        if n <= self.cfg.noise_threshold:
            # Low noise regime (N <= 0.20): alpha is fixed to minimal weight 0.02
            alpha = self.cfg.alpha_min
            state = "low_noise_default"
        else:
            # High / Moderate noise regime (N > 0.20): alpha is clamped within [0.02, 0.25]
            if alpha_raw <= self.cfg.alpha_min:
                alpha = self.cfg.alpha_min
                state = "clamped_min"
            elif alpha_raw >= self.cfg.alpha_max:
                alpha = self.cfg.alpha_max
                state = "clamped_max"
            else:
                alpha = float(alpha_raw)
                state = "adaptive_active"

        # Hard invariant enforcement: alpha can NEVER be below alpha_min or above alpha_max
        final_alpha = min(self.cfg.alpha_max, max(self.cfg.alpha_min, float(alpha)))

        return RoutingDecision(
            alpha=final_alpha,
            alpha_raw=float(alpha_raw),
            z_score=float(z),
            noise_score=n,
            token_length=l,
            routing_state=state,
        )

    def compute_alpha(self, noise_score: float, token_length: int) -> float:
        """Convenience method returning strictly clamped alpha in [0.02, 0.25]."""
        return self.route(noise_score, token_length).alpha
