"""Disentangle the effects of AGING from the effects of MENOPAUSE.

Perimenopausal symptoms overlap heavily with normal aging, so an observed
signal is confounded. We model:

    observed(t) = aging_component(age) + menopause_component(stage) + person + noise

The aging component is learned from an age-matched reference trajectory (or
supplied as a slope/intercept). The menopause component is the residual after
removing aging — a per-woman "menopause-attributable" signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DisentangledEffect:
    """Result of disentangling one feature for one observation."""

    feature: str
    age: float
    observed: float
    aging_component: float
    menopause_component: float
    person_offset: float
    stage_prior_used: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "age": round(self.age, 1),
            "observed": round(self.observed, 4),
            "aging_component": round(self.aging_component, 4),
            "menopause_component": round(self.menopause_component, 4),
            "person_offset": round(self.person_offset, 4),
            "stage_prior_used": self.stage_prior_used,
            "note": "menopause component is a residual estimate, not a diagnosis",
        }


class AgingMenopauseDisentangler:
    """Learns an aging trajectory and residualizes menopause effects.

    Fit on a reference cohort of age -> feature values (women NOT reporting
    vasomotor symptoms), then call :meth:`disentangle` for an individual.
    """

    def __init__(self) -> None:
        # feature -> (slope_per_year, intercept, residual_sd)
        self.aging_models: Dict[str, Tuple[float, float, float]] = {}
        self.person_offsets: Dict[str, float] = {}

    def fit_reference(
        self,
        ages: Sequence[float],
        feature_values: Dict[str, Sequence[float]],
    ) -> "AgingMenopauseDisentangler":
        """Fit a linear aging trajectory per feature on a reference cohort."""

        ages_arr = np.asarray(ages, dtype=float)
        for name, vals in feature_values.items():
            y = np.asarray(vals, dtype=float)
            if ages_arr.size < 2:
                slope, intercept = 0.0, float(np.mean(y)) if y.size else 0.0
            else:
                slope, intercept = np.polyfit(ages_arr, y, 1)
            pred = slope * ages_arr + intercept
            resid_sd = float(np.std(y - pred)) or 1.0
            self.aging_models[name] = (float(slope), float(intercept), resid_sd)
        return self

    def _person_offset(self, name: str) -> float:
        return self.person_offsets.get(name, 0.0)

    def calibrate_person(self, name: str, observed_baseline: float, age: float) -> None:
        """Set a person offset so the aging model matches her baseline."""

        if name not in self.aging_models:
            return
        slope, intercept, _ = self.aging_models[name]
        self.person_offsets[name] = observed_baseline - (slope * age + intercept)

    def disentangle(
        self,
        name: str,
        age: float,
        observed: float,
        stage_prior: Optional[str] = None,
    ) -> DisentangledEffect:
        if name not in self.aging_models:
            slope, intercept, _ = 0.0, observed, 1.0
        else:
            slope, intercept, _ = self.aging_models[name]
        aging = slope * age + intercept + self._person_offset(name)
        menopause = observed - aging
        return DisentangledEffect(
            feature=name,
            age=age,
            observed=observed,
            aging_component=aging,
            menopause_component=menopause,
            person_offset=self._person_offset(name),
            stage_prior_used=stage_prior,
        )

    def menopause_index(
        self,
        named_features: Dict[str, float],
        age: float,
        stage_prior: Optional[str] = None,
    ) -> Dict[str, object]:
        """Aggregate menopause-attributable residuals into a single index.

        The index is normalized by reference residual sd so that a value of +1
        means "one aging-adjusted sd beyond the asymptomatic trajectory".
        """

        zs: List[float] = []
        details: List[Dict[str, object]] = []
        for name, observed in named_features.items():
            eff = self.disentangle(name, age, observed, stage_prior)
            _, _, sd = self.aging_models.get(name, (0.0, 0.0, 1.0))
            z = eff.menopause_component / (sd or 1.0)
            zs.append(z)
            details.append(eff.as_dict())
        index = float(np.mean(zs)) if zs else 0.0
        return {
            "menopause_index_z": round(index, 3),
            "interpretation": (
                "positive => signal beyond age-matched asymptomatic trajectory"
            ),
            "n_features": len(zs),
            "details": details,
            "certainty": False,
        }
