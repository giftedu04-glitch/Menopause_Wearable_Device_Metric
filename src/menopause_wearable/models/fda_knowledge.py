"""Ingested FDA menopausal diagnostic reference, used strictly as a PRIOR.

The reference is *not* a verdict. Sensor evidence updates the prior into a
posterior distribution over perimenopausal stages. Outputs are predictive and
uncertainty-quantified — never a certainty result.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "fda_menopause_reference.json"


def load_fda_reference(path: Optional[Path | str] = None) -> Dict:
    """Load the FDA menopause reference JSON."""

    p = Path(path) if path else DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class FDAPrior:
    """A prior distribution over STRAW+10 stages built from the reference.

    The prior depends only on age (and optionally hormone context). It is the
    *starting point* the sensors will update; on its own it is intentionally
    weak and wide.
    """

    reference: Dict

    def __post_init__(self) -> None:
        self.stages: List[Dict] = self.reference["stages"]
        self.stage_ids: List[str] = [s["id"] for s in self.stages]

    def _age_weight(self, stage: Dict, age: float) -> float:
        peak = float(stage["prior_weight_by_age_peak"])
        lo, hi = (float(x) for x in stage["typical_age_range"])
        width = max(1.0, (hi - lo))
        return math.exp(-((age - peak) ** 2) / (2 * width ** 2))

    def prior(self, age: float) -> Dict[str, float]:
        """Normalized prior over stages for a given age."""

        weights = {s["id"]: self._age_weight(s, age) for s in self.stages}
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    def hormone_likelihood(
        self, fsh: Optional[float] = None, estradiol: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        """Optional Gaussian likelihood from hormone surrogates (if available).

        Maps FSH/estradiol ranges to a per-stage likelihood using a broad
        Gaussian centered on the range midpoint. Returns None if no hormones.
        """

        if fsh is None and estradiol is None:
            return None
        ref = self.reference["hormone_reference"]
        likelihood: Dict[str, float] = {}
        for sid in self.stage_ids:
            lk = 1.0
            if fsh is not None:
                group = "postmenopause" if "post" in sid else ("perimenopause" if "peri" in sid else "reproductive")
                lo, hi = ref["FSH_IU_L"][group]
                lk *= self._gauss_out_of_range(fsh, lo, hi)
            if estradiol is not None:
                group = "postmenopause" if "post" in sid else ("perimenopause" if "peri" in sid else "reproductive")
                lo, hi = ref["estradiol_pg_mL"][group]
                lk *= self._gauss_out_of_range(estradiol, lo, hi)
            likelihood[sid] = lk
        total = sum(likelihood.values()) or 1.0
        return {k: v / total for k, v in likelihood.items()}

    @staticmethod
    def _gauss_out_of_range(x: float, lo: float, hi: float) -> float:
        mid = (lo + hi) / 2.0
        sigma = max(1.0, (hi - lo))
        return math.exp(-((x - mid) ** 2) / (2 * sigma ** 2))

    def update(
        self,
        prior: Dict[str, float],
        likelihood: Dict[str, float],
        strength: float = 1.0,
    ) -> Dict[str, float]:
        """Bayesian update of the prior by a likelihood, with a strength knob.

        `strength` in [0, 1] tempers how much sensor evidence can move the
        prior. Full certainty is prevented by construction: no single update can
        collapse the posterior onto a point.
        """

        tempered = {k: v ** max(0.05, strength) for k, v in likelihood.items()}
        post = {k: prior[k] * tempered.get(k, 1.0) for k in prior}
        total = sum(post.values()) or 1.0
        return {k: v / total for k, v in post.items()}

    @staticmethod
    def summarize(posterior: Dict[str, float], max_confidence: float = 0.95) -> Dict[str, object]:
        """Summarize a posterior, capping displayed confidence below certainty."""

        items = sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_p = items[0]
        capped = min(top_p, max_confidence)
        return {
            "most_likely_stage": top_id,
            "probability": round(capped, 3),
            "distribution": {k: round(v, 3) for k, v in posterior.items()},
            "certainty": False,
        }
