"""Digital-twin webapp backend (FastAPI).

Serves the digital-twin UI and exposes the twin API. Provides a hardware-free
demo endpoint that runs the full pipeline and returns a twin trajectory.

Reminder: every response carries the "not a diagnosis / not a certainty" note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..pipeline import run_demo
from ..digital_twin.twin import DigitalTwin
from ..sensors.psychological import BehavioralTelemetry
from ..signal.processing import FusedObservation

WEBAPP_DIR = Path(__file__).resolve().parents[3] / "webapp"

app = FastAPI(
    title="Menopause Digital Twin",
    version="0.1.0",
    description=(
        "Probabilistic digital-twin estimate. NOT a diagnosis. NOT a certainty. "
        "Research use only."
    ),
)

# In-memory twin registry (replace with persistent store in production).
_TWINS: Dict[str, DigitalTwin] = {}


class CorrectionIn(BaseModel):
    variable: str = Field(..., examples=["stress"])
    predicted: float
    observed: float
    t: float = 0.0


class ObservationIn(BaseModel):
    """A pre-computed fused observation pushed into a twin."""

    t: float
    named: Dict[str, float]


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "disclaimer": "not a medical device; estimates only"}


@app.get("/api/twin/demo")
def twin_demo(
    person_id: str = "demo-person-001",
    age: float = 49.0,
    duration_s: int = 600,
) -> JSONResponse:
    """Run the hardware-free end-to-end demo and return a twin trajectory.

    Also registers a twin under `person_id` so the interactive-ML correction
    endpoint works for the demo person.
    """

    result = run_demo(person_id=person_id, age=age, duration_s=duration_s)
    _TWINS.setdefault(person_id, DigitalTwin(person_id=person_id, age=age))
    return JSONResponse(result)


@app.get("/api/twin/{person_id}")
def get_twin(person_id: str) -> JSONResponse:
    twin = _TWINS.get(person_id)
    if twin is None:
        raise HTTPException(status_code=404, detail="twin not found")
    return JSONResponse(twin.snapshot().as_dict())


@app.post("/api/twin/{person_id}/create")
def create_twin(person_id: str, age: float = 49.0) -> JSONResponse:
    _TWINS[person_id] = DigitalTwin(person_id=person_id, age=age)
    return JSONResponse({"created": person_id, "age": age})


@app.post("/api/twin/{person_id}/observe")
def observe(person_id: str, obs: ObservationIn) -> JSONResponse:
    twin = _TWINS.get(person_id)
    if twin is None:
        raise HTTPException(status_code=404, detail="twin not found")
    twin.observe(FusedObservation(t=obs.t, vector=[], named=obs.named))
    return JSONResponse(twin.snapshot().as_dict())


@app.post("/api/twin/{person_id}/correct")
def correct(person_id: str, corr: CorrectionIn) -> JSONResponse:
    """Interactive ML: submit an in-the-loop correction."""

    twin = _TWINS.get(person_id)
    if twin is None:
        raise HTTPException(status_code=404, detail="twin not found")
    from ..digital_twin.interactive_ml import Correction

    twin.learner.submit_correction(
        Correction(t=corr.t, variable=corr.variable, predicted=corr.predicted, observed=corr.observed)
    )
    interval = twin.learner.conformal_interval(corr.variable, corr.predicted)
    return JSONResponse(
        {
            "adjusted": twin.learner.adjust(corr.variable, corr.predicted),
            "interval": interval,
            "learner": twin.learner.state(),
            "note": "interval reflects calibration; uncertainty is never hidden",
        }
    )


# ---- static webapp ------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEBAPP_DIR / "index.html")


if WEBAPP_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR)), name="static")


def main() -> None:
    import uvicorn

    print("Digital twin webapp: http://127.0.0.1:8000")
    print("NOT a medical device. Estimates only, never a certainty.")
    uvicorn.run("menopause_wearable.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
