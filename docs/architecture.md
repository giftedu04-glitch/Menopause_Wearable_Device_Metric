# Architecture

The platform is a layered pipeline. Each layer has a single responsibility and
is individually testable without hardware.

```
 raw stream        fusion features        per-person model        UI
────────────►  sensors/  ──►  signal/  ──►  digital_twin/  ──►  webapp/
  (or synthetic)   physiological   processing     twin.py          index.html
                   psychological   SensorFusion   interactive_ml   app.js
                                                        ▲
                                        models/         │ prior
                                        ├─ fda_knowledge.py     (Bayesian prior)
                                        └─ aging_menopause.py   (disentangle)
```

## Layers

1. **`sensors/`** — acquire and describe the signal.
   - `base.py` — `Reading`, `SyntheticBodyStream` (hardware-free stream with
     spatially propagating vasomotor events).
   - `physiological.py` — windowed features + vasomotor event detection from
     multi-site temperature/EDA/PPG/humidity.
   - `psychological.py` — affect/stress/fatigue/fog **proxies** as distributions.

2. **`signal/`** — condition and fuse.
   - `processing.py` — drift removal (`rolling_baseline`) and `SensorFusion`,
     which produces one explicable `FusedObservation` per window.

3. **`digital_twin/`** — the per-person model.
   - `twin.py` — `DigitalTwin`, seeded by the FDA prior, updated by sensor
     evidence with a **capped** evidence strength (prevents certainty).
   - `interactive_ml.py` — in-the-loop corrections, calibration, conformal
     intervals.

4. **`models/`** — knowledge and identification.
   - `fda_knowledge.py` — ingests `data/fda_menopause_reference.json` and turns
     it into a **prior** + optional hormone likelihood + tempered Bayesian
     update.
   - `aging_menopause.py` — learns an age-matched aging trajectory and returns
     the **menopause-attributable residual**.

5. **`api/`** — FastAPI backend serving the twin and the static webapp.

6. **`webapp/`** — the digital-twin UI (vanilla HTML/CSS/JS, no build step).

## Why this decomposition

- **No certainty by construction.** Confidence is capped (`max_confidence`),
  evidence strength is capped, and every consumer receives a *distribution*.
- **Explicability.** Fusion uses fixed, auditable scaling and reports per-domain
  contributions, so the twin can explain *why*.
- **Privacy.** Psychological estimation is designed for on-device execution; the
  backend never needs raw behavioral content.
- **Hardware-free development.** `SyntheticBodyStream` lets the full stack run
  in CI and on a laptop.
