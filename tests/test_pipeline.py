import math

import pytest

from menopause_wearable.sensors.base import SyntheticBodyStream, SyntheticConfig
from menopause_wearable.sensors.physiological import PhysiologicalSensorArray
from menopause_wearable.pipeline import run_demo, PROPAGATION_PATH
from menopause_wearable.models.fda_knowledge import FDAPrior, load_fda_reference
from menopause_wearable.models.aging_menopause import AgingMenopauseDisentangler
from menopause_wearable.digital_twin.interactive_ml import InteractiveLearner, Correction


def test_synthetic_stream_emits_readings():
    readings = list(SyntheticBodyStream(SyntheticConfig(duration_s=10)).stream())
    assert len(readings) > 0
    sensors = {r.sensor for r in readings}
    assert {"temperature", "eda", "hr", "hrv", "humidity"} <= sensors


def test_vasomotor_events_detected():
    readings = list(
        SyntheticBodyStream(
            SyntheticConfig(duration_s=600, flush_rate_per_hour=12.0, flush_intensity=1.5)
        ).stream()
    )
    events = PhysiologicalSensorArray().detect_vasomotor_events(readings, PROPAGATION_PATH)
    assert len(events) >= 1
    assert events[0].peak_delta_c > 0


def test_pipeline_produces_bounded_posterior():
    result = run_demo(duration_s=300)
    snaps = result["snapshots"]
    assert len(snaps) > 0
    for s in snaps:
        total = sum(s["stage_posterior"].values())
        # values are rounded for display; allow rounding tolerance
        assert math.isclose(total, 1.0, abs_tol=5e-3)
        # certainty is never reached
        assert s["confidence"] < 1.0
        assert s["certain"] is False


def test_fda_prior_normalized():
    prior_model = FDAPrior(load_fda_reference())
    prior = prior_model.prior(49.0)
    assert math.isclose(sum(prior.values()), 1.0, rel_tol=1e-9)
    assert set(prior) == set(prior_model.stage_ids)


def test_aging_menopause_residual():
    dd = AgingMenopauseDisentangler()
    ages = [40, 45, 50, 55, 60]
    dd.fit_reference(ages, {"vasomotor_index": [0.02, 0.03, 0.04, 0.05, 0.06]})
    eff = dd.disentangle("vasomotor_index", age=50, observed=0.30)
    # observed well above the aging trajectory -> positive menopause residual
    assert eff.menopause_component > 0


def test_interactive_learner_interval():
    learner = InteractiveLearner()
    for i in range(6):
        learner.submit_correction(Correction(t=i, variable="stress", predicted=0.5, observed=0.3))
    interval = learner.conformal_interval("stress", 0.5)
    assert interval["low"] < 0.5 < interval["high"]
    assert interval["half_width"] > 0
