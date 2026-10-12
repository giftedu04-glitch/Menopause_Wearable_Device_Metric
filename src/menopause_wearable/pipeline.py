"""End-to-end pipeline: synthetic wearable -> features -> twin -> snapshot.

This is the glue used by the webapp and the tests. It runs the full assembly
without any hardware.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .sensors.base import SyntheticBodyStream, SyntheticConfig, DEFAULT_SITES
from .sensors.physiological import PhysiologicalSensorArray
from .sensors.psychological import PsychologicalEstimator, BehavioralTelemetry
from .signal.processing import SensorFusion
from .digital_twin.twin import DigitalTwin

# Spatial propagation order for vasomotor waves.
PROPAGATION_PATH: Sequence[str] = [s.site for s in DEFAULT_SITES]


def run_demo(
    person_id: str = "demo-person-001",
    age: float = 49.0,
    duration_s: int = 600,
    window_s: int = 60,
    stride_s: int = 10,
    reference_cohort: Optional[Dict[str, List[float]]] = None,
    flush_rate_per_hour: float = 8.0,
) -> Dict[str, object]:
    """Run the full pipeline and return the twin's trajectory.

    Returns a dict with `snapshots` (list of twin snapshots) and `events`
    (detected vasomotor events) for the webapp to render.
    """

    cfg = SyntheticConfig(duration_s=duration_s, flush_rate_per_hour=flush_rate_per_hour)
    stream = SyntheticBodyStream(cfg)
    readings = list(stream.stream())

    phys = PhysiologicalSensorArray()
    psych = PsychologicalEstimator()
    fusion = SensorFusion()
    twin = DigitalTwin(person_id=person_id, age=age)

    # Seed the aging model: if none supplied, build a synthetic age-matched
    # asymptomatic reference so disentangling is demonstrated end-to-end.
    if reference_cohort is None:
        reference_cohort = _synthetic_reference()
    twin.seed_reference_aging(reference_cohort["ages"], reference_cohort["features"])
    for name, observed in _person_baselines().items():
        twin.disentangler.calibrate_person(name, observed, age)

    events = phys.detect_vasomotor_events(readings, PROPAGATION_PATH)

    snapshots = []
    start = 0.0
    while start + window_s <= duration_s:
        window = (start, start + window_s)
        feats = phys.features(readings, window, PROPAGATION_PATH)
        t = start + window_s
        pstate = psych.estimate(
            t,
            hr_mean_bpm=feats.hr_mean_bpm,
            hrv_rmssd_ms=feats.hrv_rmssd_ms,
            eda_phasic_peaks=feats.eda_phasic_peaks,
            vasomotor_index=feats.vasomotor_index,
            telemetry=BehavioralTelemetry(
                youtube_focus_min=12.0,
                social_min=8.0,
                app_switches=14,
                screen_on_min=25.0,
            ),
        )
        obs = fusion.fuse(feats, pstate, t=t)
        twin.observe(obs)
        snap = twin.snapshot().as_dict()
        snap["vasomotor_index"] = feats.vasomotor_index
        snap["temp_gradient"] = feats.temp_gradient_c
        snap["stress"] = round(pstate.stress_mean, 3)
        snapshots.append(snap)
        start += stride_s

    return {
        "person_id": person_id,
        "age": age,
        "snapshots": snapshots,
        "events": [
            {
                "onset_s": e.onset_s,
                "end_s": e.end_s,
                "peak_delta_c": e.peak_delta_c,
                "propagation_sites": e.propagation_sites,
                "spatial_delay_s": e.spatial_delay_s,
            }
            for e in events
        ],
        "disclaimer": (
            "Probabilistic research output. NOT a diagnosis and NOT a certainty."
        ),
    }


def _synthetic_reference() -> Dict[str, List[float]]:
    """Age-matched asymptomatic reference cohort (illustrative)."""

    ages = [40, 43, 46, 49, 52, 55, 58, 61]
    # temperature gradient and HRV drift slowly with age in asymptomatic women
    return {
        "ages": ages,
        "features": {
            "temp_gradient": [0.15, 0.16, 0.17, 0.18, 0.20, 0.21, 0.22, 0.23],
            "hrv_low": [0.05, 0.07, 0.09, 0.11, 0.14, 0.16, 0.18, 0.20],
            "vasomotor_index": [0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05],
        },
    }


def _person_baselines() -> Dict[str, float]:
    """This individual's baseline values (for person-offset calibration)."""

    return {"temp_gradient": 0.18, "hrv_low": 0.11, "vasomotor_index": 0.03}
