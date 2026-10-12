"""Signal conditioning, windowing, and multi-modal sensor fusion.

The fusion step converts the per-window physiological features and the
psychological proxies into a single, normalized, explicable latent observation
that the digital twin consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ..sensors.physiological import PhysiologicalFeatures
from ..sensors.psychological import PsychologicalState


def rolling_baseline(series: Sequence[Tuple[float, float]], window_s: float) -> List[Tuple[float, float]]:
    """Subtract a trailing rolling-mean baseline (drift removal)."""

    out: List[Tuple[float, float]] = []
    q: List[float] = []
    tq: List[float] = []
    for t, v in series:
        q.append(v)
        tq.append(t)
        while tq and t - tq[0] > window_s:
            tq.pop(0)
            q.pop(0)
        base = sum(q) / len(q)
        out.append((t, v - base))
    return out


@dataclass
class FusedObservation:
    """Normalized, explicable latent observation for the digital twin."""

    t: float
    vector: List[float]
    named: Dict[str, float]
    contributions: Dict[str, float] = field(default_factory=dict)


# Canonical order of the fused feature vector.
FEATURE_NAMES: List[str] = [
    "vasomotor_index",
    "temp_gradient",
    "eda_mean",
    "eda_peaks",
    "hr_mean",
    "hrv_low",
    "humidity",
    "stress",
    "fatigue",
    "fog",
    "sleep_frag",
]


class SensorFusion:
    """Fuses physiological + psychological evidence into a latent vector.

    A simple, auditable standardization is used (fixed research means/scales)
    rather than an opaque learned encoder, so every contribution is explicable
    to the woman and her clinician.
    """

    def __init__(self) -> None:
        self.scale = {
            "vasomotor_index": 1.0,
            "temp_gradient": 0.6,
            "eda_mean": 4.0,
            "eda_peaks": 8.0,
            "hr_mean": 80.0,
            "hrv_low": 45.0,
            "humidity": 70.0,
            "stress": 1.0,
            "fatigue": 1.0,
            "fog": 1.0,
            "sleep_frag": 1.0,
        }

    def fuse(
        self,
        phys: PhysiologicalFeatures,
        psych: PsychologicalState,
        *,
        t: float,
    ) -> FusedObservation:
        hrv_low = max(0.0, (45.0 - phys.hrv_rmssd_ms) / 45.0)
        raw = {
            "vasomotor_index": phys.vasomotor_index,
            "temp_gradient": phys.temp_gradient_c,
            "eda_mean": phys.eda_mean_us,
            "eda_peaks": float(phys.eda_phasic_peaks),
            "hr_mean": phys.hr_mean_bpm,
            "hrv_low": hrv_low,
            "humidity": phys.humidity_mean_pct,
            "stress": psych.stress_mean,
            "fatigue": psych.fatigue_mean,
            "fog": psych.cognitive_fog_mean,
            "sleep_frag": psych.sleep_fragmentation,
        }
        named = {k: round(v / self.scale[k], 4) for k, v in raw.items()}
        vector = [named[k] for k in FEATURE_NAMES]
        contributions = {
            "physiological": round(
                sum(named[k] for k in FEATURE_NAMES if k not in {"stress", "fatigue", "fog", "sleep_frag"}), 3
            ),
            "psychological": round(named["stress"] + named["fatigue"] + named["fog"] + named["sleep_frag"], 3),
        }
        return FusedObservation(t=t, vector=vector, named=named, contributions=contributions)
