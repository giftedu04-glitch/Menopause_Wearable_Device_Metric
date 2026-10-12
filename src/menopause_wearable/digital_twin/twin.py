"""The per-person digital twin.

A :class:`DigitalTwin` is a continuously updated computational model of ONE
woman. It ingests fused sensor observations, maintains a posterior over
perimenopausal stages (seeded from the ingested FDA reference prior), and
exposes a snapshot for the webapp. It never returns a certainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models.fda_knowledge import FDAPrior, load_fda_reference
from ..models.aging_menopause import AgingMenopauseDisentangler
from ..signal.processing import FusedObservation
from .interactive_ml import InteractiveLearner


@dataclass
class TwinSnapshot:
    """Serializable state of the twin at a moment in time."""

    person_id: str
    age: float
    t: float
    latest_named: Dict[str, float]
    stage_posterior: Dict[str, float]
    most_likely_stage: str
    confidence: float
    menopause_index_z: float
    history_len: int
    learner_state: Dict[str, object]
    certain: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "person_id": self.person_id,
            "age": self.age,
            "t": self.t,
            "latest_named": self.latest_named,
            "stage_posterior": self.stage_posterior,
            "most_likely_stage": self.most_likely_stage,
            "confidence": self.confidence,
            "menopause_index_z": self.menopause_index_z,
            "history_len": self.history_len,
            "learner": self.learner_state,
            "certain": self.certain,
            "disclaimer": (
                "Probabilistic research estimate. NOT a diagnosis and NOT a certainty. "
                "Consult a qualified clinician."
            ),
        }


class DigitalTwin:
    """A unique individual's continuously updated model."""

    def __init__(self, person_id: str, age: float, reference: Optional[Dict] = None) -> None:
        self.person_id = person_id
        self.age = float(age)
        self.prior_model = FDAPrior(reference or load_fda_reference())
        self.posterior: Dict[str, float] = self.prior_model.prior(self.age)
        self.disentangler = AgingMenopauseDisentangler()
        self.learner = InteractiveLearner()
        self.history: List[FusedObservation] = []
        self.latest_named: Dict[str, float] = {}
        self.menopause_index_z: float = 0.0
        self._seeded = False

    # ---- setup -----------------------------------------------------------
    def seed_reference_aging(
        self,
        ages: List[float],
        feature_values: Dict[str, List[float]],
    ) -> None:
        """Provide an age-matched asymptomatic reference cohort."""

        self.disentangler.fit_reference(ages, feature_values)
        self._seeded = True

    # ---- observation -----------------------------------------------------
    def observe(self, obs: FusedObservation, evidence_strength: float = 0.5) -> None:
        """Ingest a fused observation and update the twin's state.

        Evidence strength is capped so no single observation can push the
        posterior to certainty.
        """

        self.history.append(obs)
        self.latest_named = dict(obs.named)

        # Build a sensor-evidence likelihood over stages from vasomotor + psych
        # load. Higher symptom load shifts mass toward peri/post stages.
        load = min(1.0, obs.named.get("vasomotor_index", 0.0)
                   + 0.5 * obs.named.get("stress", 0.0)
                   + 0.5 * obs.named.get("fatigue", 0.0))
        likelihood: Dict[str, float] = {}
        for sid in self.prior_model.stage_ids:
            if "early_peri" in sid:
                lk = 0.5 + load
            elif "late_peri" in sid:
                lk = 0.4 + 1.2 * load
            elif "early_post" in sid:
                lk = 0.4 + 1.4 * load
            elif "late_reproductive" in sid:
                lk = 0.7 + 0.5 * (1.0 - load)
            else:  # late_post
                lk = 0.5 + 0.3 * load
            likelihood[sid] = lk

        strength = min(0.8, max(0.05, evidence_strength))
        self.posterior = self.prior_model.update(self.posterior, likelihood, strength=strength)

        # Disentangling update (falls back to identity if no reference)
        if self._seeded:
            result = self.disentangler.menopause_index(named_features=self.latest_named, age=self.age)
            self.menopause_index_z = float(result["menopause_index_z"])

    # ---- output ----------------------------------------------------------
    def snapshot(self) -> TwinSnapshot:
        summary = self.prior_model.summarize(self.posterior)
        return TwinSnapshot(
            person_id=self.person_id,
            age=self.age,
            t=self.history[-1].t if self.history else 0.0,
            latest_named={k: round(v, 3) for k, v in self.latest_named.items()},
            stage_posterior=summary["distribution"],  # type: ignore[arg-type]
            most_likely_stage=str(summary["most_likely_stage"]),
            confidence=float(summary["probability"]),
            menopause_index_z=self.menopause_index_z,
            history_len=len(self.history),
            learner_state=self.learner.state(),
        )
