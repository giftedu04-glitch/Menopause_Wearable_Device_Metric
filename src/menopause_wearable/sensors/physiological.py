"""Physiological sensing: multi-site temperature, EDA, PPG, humidity.

This module turns the raw multi-site stream into interpretable physiological
features, concentrating on the *spatiotemporal* vasomotor signature that single
point sensors miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .base import Reading, SyntheticBodyStream

# Marker used across the pipeline to denote a detected vasomotor event.
HOT_FLUSH_MARKER = "vasomotor_event"


@dataclass
class VasomotorEvent:
    """A detected hot-flush-like vasomotor event."""

    onset_s: float
    end_s: float
    peak_delta_c: float
    propagation_sites: List[str]
    spatial_delay_s: float


@dataclass
class PhysiologicalFeatures:
    """Windowed physiological feature vector."""

    window_start_s: float
    window_end_s: float
    mean_skin_temp_c: Dict[str, float]
    temp_gradient_c: float          # max inter-site difference
    eda_mean_us: float
    eda_phasic_peaks: int
    hr_mean_bpm: float
    hrv_rmssd_ms: float
    humidity_mean_pct: float
    vasomotor_index: float          # 0..1 intensity of flushing in window


class PhysiologicalSensorArray:
    """Extracts physiological features and vasomotor events from a stream."""

    def __init__(self, flush_delta_threshold_c: float = 0.25) -> None:
        self.flush_delta_threshold_c = flush_delta_threshold_c

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _series(readings: Sequence[Reading], sensor: str) -> List[Tuple[float, float]]:
        return [(r.t, r.value) for r in readings if r.sensor == sensor]

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _sites(self, readings: Sequence[Reading], sensor: str) -> Dict[str, List[Tuple[float, float]]]:
        out: Dict[str, List[Tuple[float, float]]] = {}
        for r in readings:
            if r.sensor == sensor:
                out.setdefault(r.site, []).append((r.t, r.value))
        return out

    # ---- event detection -------------------------------------------------
    def detect_vasomotor_events(
        self, readings: Sequence[Reading], propagation_path: Sequence[str]
    ) -> List[VasomotorEvent]:
        """Detect flush events by earliest per-site temperature threshold crossing.

        A small per-site delta above the site baseline marks a site as "firing".
        The event onset is the first site to fire; the spatial delay is the spread
        of firing times across the ordered propagation path.
        """
        site_series = self._sites(readings, "temperature")
        if not site_series:
            return []

        # Baseline = first 10% of samples per site.
        baselines: Dict[str, float] = {}
        fire_times: Dict[str, List[float]] = {}
        for site, series in site_series.items():
            vals = [v for _, v in series]
            k = max(1, len(vals) // 10)
            base = self._mean(vals[:k])
            baselines[site] = base
            fires = [t for t, v in series if v - base >= self.flush_delta_threshold_c]
            fire_times[site] = fires

        all_fires = sorted(t for fires in fire_times.values() for t in fires)
        if not all_fires:
            return []

        # Cluster firing times into events with a 60 s gap criterion.
        events: List[VasomotorEvent] = []
        cluster = [all_fires[0]]
        for t in all_fires[1:]:
            if t - cluster[-1] <= 60.0:
                cluster.append(t)
            else:
                events.append(cluster)
                cluster = [t]
        events.append(cluster)

        results: List[VasomotorEvent] = []
        for cl in events:
            onset, end = cl[0], cl[-1]
            participating = [
                s for s in propagation_path
                if any(onset - 30 <= ft <= end + 30 for ft in fire_times.get(s, []))
            ]
            peak = 0.0
            for site in participating:
                for t, v in site_series.get(site, []):
                    if onset <= t <= end:
                        peak = max(peak, v - baselines[site])
            delay = (end - onset) if len(participating) > 1 else 0.0
            results.append(
                VasomotorEvent(
                    onset_s=onset,
                    end_s=end,
                    peak_delta_c=round(peak, 3),
                    propagation_sites=participating,
                    spatial_delay_s=round(delay, 1),
                )
            )
        return results

    # ---- windowed features ----------------------------------------------
    def features(
        self,
        readings: Sequence[Reading],
        window: Tuple[float, float],
        propagation_path: Sequence[str],
    ) -> PhysiologicalFeatures:
        start, end = window
        win = [r for r in readings if start <= r.t < end]

        temp_sites = self._sites(win, "temperature")
        site_means = {s: self._mean([v for _, v in ser]) for s, ser in temp_sites.items()}
        gradient = (max(site_means.values()) - min(site_means.values())) if site_means else 0.0

        eda_vals = [v for _, v in self._series(win, "eda")]
        # phasic peaks: simple local-maxima count above mean + 1 sd
        eda_peaks = 0
        if len(eda_vals) >= 3:
            m = self._mean(eda_vals)
            sd = (sum((x - m) ** 2 for x in eda_vals) / len(eda_vals)) ** 0.5
            for i in range(1, len(eda_vals) - 1):
                if eda_vals[i] > eda_vals[i - 1] and eda_vals[i] >= eda_vals[i + 1] and eda_vals[i] > m + sd:
                    eda_peaks += 1

        hr_vals = [v for _, v in self._series(win, "hr")]
        hrv_vals = [v for _, v in self._series(win, "hrv")]
        hum_vals = [v for _, v in self._series(win, "humidity")]

        events = self.detect_vasomotor_events(win, propagation_path)
        vasomotor_index = min(1.0, sum(e.peak_delta_c for e in events) / 1.0) if events else 0.0

        return PhysiologicalFeatures(
            window_start_s=start,
            window_end_s=end,
            mean_skin_temp_c={s: round(v, 3) for s, v in site_means.items()},
            temp_gradient_c=round(gradient, 3),
            eda_mean_us=round(self._mean(eda_vals), 3),
            eda_phasic_peaks=eda_peaks,
            hr_mean_bpm=round(self._mean(hr_vals), 2),
            hrv_rmssd_ms=round(self._mean(hrv_vals), 2),
            humidity_mean_pct=round(self._mean(hum_vals), 2),
            vasomotor_index=round(vasomotor_index, 3),
        )


def collect(readings: Sequence[Reading]) -> PhysiologicalSensorArray:
    """Convenience accessor mirroring `sensors.collect`."""

    return PhysiologicalSensorArray()
