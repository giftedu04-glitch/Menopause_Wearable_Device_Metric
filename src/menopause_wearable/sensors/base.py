"""Sensor primitives shared by all modalities.

The wearable is modeled as a set of nodes at distinct body sites. Each node
emits timestamped :class:`Reading` objects. A :class:`SyntheticBodyStream`
produces a realistic multi-site stream (with injected vasomotor events) so the
whole pipeline can run without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional
import math
import random


@dataclass(frozen=True)
class Reading:
    """A single timestamped sensor sample from one body site."""

    t: float            # seconds since stream start
    node_id: str        # e.g. "sternum"
    site: str           # body site label, e.g. "sternum"
    sensor: str         # modality, e.g. "temperature"
    value: float


@dataclass
class BodySiteState:
    """Latent physical state of a body site used by the synthetic generator."""

    site: str
    baseline_temp: float
    thermal_mass: float = 1.0        # delay constant for vasomotor propagation
    baseline_eda: float = 2.0        # microsiemens


@dataclass
class SyntheticConfig:
    """Parameters controlling the synthetic perimenopausal individual."""

    seed: int = 7
    duration_s: int = 600
    flush_rate_per_hour: float = 4.0     # vasomotor events per hour
    flush_intensity: float = 1.0         # 0..2
    hrv_baseline_ms: float = 45.0
    nocturnal: bool = False
    sites: List[BodySiteState] = field(default_factory=list)


# Default body sites match config/sensors.yaml (spatial gradient ordering).
DEFAULT_SITES: List[BodySiteState] = [
    BodySiteState("sternum", baseline_temp=36.60, thermal_mass=0.6),
    BodySiteState("neck", baseline_temp=36.70, thermal_mass=0.9),
    BodySiteState("left_axilla", baseline_temp=36.80, thermal_mass=1.1),
    BodySiteState("forehead", baseline_temp=36.40, thermal_mass=1.3),
    BodySiteState("right_wrist", baseline_temp=35.60, thermal_mass=1.6),
    BodySiteState("left_wrist", baseline_temp=35.50, thermal_mass=1.8),
]


class SyntheticBodyStream:
    """Generates a multi-site, multi-modal sensor stream.

    The generator injects *spatially propagating* vasomotor events: a hot flush
    begins at the sternum and propagates outward with site-specific delays, so
    the discriminative signal lives in the spatiotemporal structure.
    """

    def __init__(self, config: Optional[SyntheticConfig] = None) -> None:
        self.config = config or SyntheticConfig()
        self._rng = random.Random(self.config.seed)
        if not self.config.sites:
            self.config.sites = list(DEFAULT_SITES)
        self._flush_events = self._schedule_flushes()

    def _schedule_flushes(self) -> List[float]:
        expected = self.config.flush_rate_per_hour * (self.config.duration_s / 3600.0)
        count = self._rng.poisson(expected) if hasattr(self._rng, "poisson") else int(expected)
        count = max(0, int(count))
        return sorted(self._rng.uniform(0, self.config.duration_s) for _ in range(count))

    def _vasomotor_amplitude(self, t: float, site: BodySiteState) -> float:
        amp = 0.0
        for onset in self._flush_events:
            delay = site.thermal_mass * 4.0  # seconds of propagation delay
            dt = t - (onset + delay)
            if dt < 0:
                continue
            # rapid rise then exponential decay
            rise = math.exp(-((dt - 20.0) ** 2) / (2 * 30.0 ** 2))
            decay = math.exp(-dt / 120.0) if dt > 20 else 1.0
            amp += self.config.flush_intensity * rise * decay
        return amp

    def stream(self) -> Iterator[Reading]:
        cfg = self.config
        dt = 0.25  # 4 Hz base sampling
        n = int(cfg.duration_s / dt)
        for i in range(n):
            t = i * dt
            for site in cfg.sites:
                vaso = self._vasomotor_amplitude(t, site)
                noise = self._rng.gauss(0, 0.01)
                temp = site.baseline_temp + 0.35 * vaso + noise
                yield Reading(t, site.site, site.site, "temperature", temp)

                # EDA surges with the flush (sympathetic activation)
                eda = site.baseline_eda + 1.8 * vaso + self._rng.gauss(0, 0.05)
                yield Reading(t, site.site, site.site, "eda", max(0.1, eda))

                # PPG-derived heart rate & HRV (peripheral sites only here)
                if site.site in {"left_wrist", "sternum", "forehead"}:
                    hr = 62.0 + 6.0 * vaso + self._rng.gauss(0, 1.5)
                    hrv = max(5.0, cfg.hrv_baseline_ms - 8.0 * vaso + self._rng.gauss(0, 2.0))
                    yield Reading(t, site.site, site.site, "hr", hr)
                    yield Reading(t, site.site, site.site, "hrv", hrv)

                # Humidity tracks sweating
                humidity = 40.0 + 12.0 * vaso + self._rng.gauss(0, 0.5)
                yield Reading(t, site.site, site.site, "humidity", humidity)


def group_by_sensor(readings: List[Reading]) -> Dict[str, List[Reading]]:
    """Group readings by sensor modality."""

    out: Dict[str, List[Reading]] = {}
    for r in readings:
        out.setdefault(r.sensor, []).append(r)
    return out
