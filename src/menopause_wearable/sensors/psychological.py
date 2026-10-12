"""Real-time psychological variable estimation (affect / stress proxies).

These are *proxies*, derived from physiology and on-phone behavioral telemetry.
They are exposed as probability distributions, never as clinical psychiatric
diagnoses or certainties. All computation is intended to run on-device/on-phone
to preserve privacy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class BehavioralTelemetry:
    """On-phone behavioral signals (the 'compiled recursion' of daily life).

    Derived locally from app/passive signals; no raw content leaves the device.
    """

    youtube_focus_min: float = 0.0      # engaged viewing minutes in window
    social_min: float = 0.0             # social app minutes in window
    app_switches: int = 0               # context-switching count (fragmentation)
    screen_on_min: float = 0.0
    typing_latency_ms: float = 220.0    # psychomotor speed proxy
    self_report_stress: Optional[float] = None   # 0..10, optional woman-reported
    self_report_mood: Optional[float] = None     # 0..10, optional woman-reported


@dataclass
class PsychologicalState:
    """Probabilistic psychological state (mean, sd) — never a certainty."""

    t: float
    stress_mean: float
    stress_sd: float
    mood_mean: float                    # higher = better valence
    mood_sd: float
    arousal_mean: float
    fatigue_mean: float
    cognitive_fog_mean: float
    sleep_fragmentation: float
    distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "stress": {"mean": round(self.stress_mean, 3), "sd": round(self.stress_sd, 3)},
            "mood": {"mean": round(self.mood_mean, 3), "sd": round(self.mood_sd, 3)},
            "arousal": {"mean": round(self.arousal_mean, 3)},
            "fatigue": {"mean": round(self.fatigue_mean, 3)},
            "cognitive_fog": {"mean": round(self.cognitive_fog_mean, 3)},
            "sleep_fragmentation": round(self.sleep_fragmentation, 3),
            "note": "probabilistic proxy; not a clinical diagnosis",
        }


class PsychologicalEstimator:
    """Maps physiological features + telemetry to psychological proxies.

    Uses a transparent weighted-evidence model so the twin can explain *why*.
    """

    def __init__(self) -> None:
        # weights are research defaults, tunable via interactive ML
        self.w_hrv_low = 0.35       # low HRV -> stress
        self.w_eda = 0.25           # phasic EDA -> stress/arousal
        self.w_vasomotor = 0.20     # flushing -> discomfort/fatigue
        self.w_switches = 0.10      # fragmentation -> fog
        self.w_report = 0.60        # self-report dominates when present

    def estimate(
        self,
        t: float,
        *,
        hr_mean_bpm: float,
        hrv_rmssd_ms: float,
        eda_phasic_peaks: int,
        vasomotor_index: float,
        telemetry: Optional[BehavioralTelemetry] = None,
    ) -> PsychologicalState:
        tel = telemetry or BehavioralTelemetry()

        hrv_term = max(0.0, (45.0 - hrv_rmssd_ms) / 45.0)   # 0..1 stress contribution
        eda_term = min(1.0, eda_phasic_peaks / 8.0)
        switch_term = min(1.0, tel.app_switches / 40.0)

        stress = self.w_hrv_low * hrv_term + self.w_eda * eda_term + 0.15 * vasomotor_index
        if tel.self_report_stress is not None:
            stress = 0.4 * stress + self.w_report * (tel.self_report_stress / 10.0)

        arousal = 0.5 * eda_term + 0.3 * hrv_term + 0.2 * (hr_mean_bpm - 60.0) / 20.0

        fatigue = 0.4 * vasomotor_index + 0.3 * switch_term + 0.3 * hrv_term
        fog = 0.5 * switch_term + 0.3 * fatigue + 0.2 * hrv_term

        mood = 1.0 - 0.6 * stress - 0.3 * fatigue + 0.1 * (tel.youtube_focus_min / 30.0)
        if tel.self_report_mood is not None:
            mood = 0.4 * mood + self.w_report * (tel.self_report_mood / 10.0)

        # uncertainty shrinks when self-report is available
        base_sd = 0.18
        sd = base_sd * (0.7 if tel.self_report_stress is not None else 1.0)

        return PsychologicalState(
            t=t,
            stress_mean=stress,
            stress_sd=sd,
            mood_mean=mood,
            mood_sd=sd,
            arousal_mean=arousal,
            fatigue_mean=fatigue,
            cognitive_fog_mean=fog,
            sleep_fragmentation=min(1.0, 0.5 * switch_term + 0.5 * vasomotor_index),
            distributions={
                "stress": {"mean": stress, "sd": sd},
                "mood": {"mean": mood, "sd": sd},
            },
        )
