"""Interactive machine learning for the digital twin.

The twin is a collaborator, not a black box. The woman (and optionally a
clinician) can correct estimates; the learner records corrections, computes an
empirical calibration and conformal-style intervals, and adjusts a transparent
scale/bias — so the model improves *with* her, on-device, without shipping raw
data anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Correction:
    """A single in-the-loop correction of one estimated variable."""

    t: float
    variable: str
    predicted: float
    observed: float

    @property
    def residual(self) -> float:
        return self.observed - self.predicted


class InteractiveLearner:
    """Records corrections and produces calibrated intervals and adjustments."""

    def __init__(self) -> None:
        self.corrections: List[Correction] = []
        # per-variable multiplicative scale and additive bias corrections
        self._scale: Dict[str, float] = {}
        self._bias: Dict[str, float] = {}

    def submit_correction(self, corr: Correction) -> None:
        self.corrections.append(corr)
        self._recompute(corr.variable)

    def _recompute(self, variable: str) -> None:
        preds = [c.predicted for c in self.corrections if c.variable == variable]
        obs = [c.observed for c in self.corrections if c.variable == variable]
        if len(preds) < 2:
            return
        preds_arr = np.asarray(preds, dtype=float)
        obs_arr = np.asarray(obs, dtype=float)
        # least-squares scale (through origin correction) + residual mean bias
        denom = float(np.dot(preds_arr, preds_arr)) or 1.0
        scale = float(np.dot(preds_arr, obs_arr) / denom)
        bias = float(np.mean(obs_arr - scale * preds_arr))
        self._scale[variable] = scale
        self._bias[variable] = bias

    def adjust(self, variable: str, value: float) -> float:
        """Apply learned scale/bias to a raw estimate."""

        return self._scale.get(variable, 1.0) * value + self._bias.get(variable, 0.0)

    def conformal_interval(self, variable: str, value: float, alpha: float = 0.10) -> Dict[str, float]:
        """Split-conformal-style interval from past absolute residuals.

        Returns a two-sided interval covering ~1-alpha of past errors. With few
        corrections the interval is wide — by design, uncertainty is never hidden.
        """

        resid = np.abs([c.residual for c in self.corrections if c.variable == variable])
        if resid.size == 0:
            half = max(0.2, abs(value) * 0.5)
        else:
            level = min(1.0, 1.0 - alpha + 1.0 / (resid.size + 1))
            half = float(np.quantile(resid, level))
        return {"low": value - half, "high": value + half, "half_width": half, "alpha": alpha}

    def state(self) -> Dict[str, object]:
        return {
            "n_corrections": len(self.corrections),
            "scale": {k: round(v, 3) for k, v in self._scale.items()},
            "bias": {k: round(v, 3) for k, v in self._bias.items()},
        }
