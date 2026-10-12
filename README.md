# Menopause Wearable Device Metric

**An at-home wearable sensing + digital-twin platform for modeling the perimenopausal
transition in an individual woman.**

> ⚠️ **Not a medical device. Not for diagnosis, treatment, or cure of any disease.**
> This repository is a **research and engineering reference**. It produces **probabilistic,
> explainable signals — never certainty**. See [Disclaimer](#-disclaimer).

---

## Table of contents

- [Motivation](#-motivation)
- [What this repository is](#-what-this-repository-is)
- [The wearable](#-the-wearable)
  - [Sensing the subtle: sensor suite](#sensing-the-subtle-sensor-suite)
  - [Spatially distributed body sensing](#spatially-distributed-body-sensing)
  - [Real-time psychological variables](#real-time-psychological-variables)
- [The digital twin assembly](#-the-digital-twin-assembly)
- [The webapp (digital twin of an individual)](#-the-webapp-digital-twin-of-an-individual)
- [FDA menopausal diagnosis as ingested prior — predictive, not certain](#-fda-menopausal-diagnosis-as-ingested-prior--predictive-not-certain)
- [Novel biophysics & data-driven analysis](#-novel-biophysics--data-driven-analysis)
  - [Interactive machine learning](#-interactive-machine-learning)
  - [Disentangling aging from menopause](#-disentangling-aging-from-menopause)
- [Architecture](#-architecture)
- [Quickstart](#-quickstart)
- [Repository layout](#-repository-layout)
- [The device is one node in a larger whole](#-the-device-is-one-node-in-a-larger-whole)
- [Roadmap](#-roadmap)
- [Disclaimer](#-disclaimer)
- [References](#-references)
- [License](#-license)

---

## Motivation

Women experience a range of health issues across the lifespan that affect how they live
their daily lives and impact the health and resilience of society as a whole. Diseases and
conditions that exclusively impact women represent a significant burden; **hysterectomy**
and **breast cancer** are all too common, and the effects of **menopause** impose a costly
health, financial, and career burden across families, workplaces, and communities. Other
health challenges simply affect women differently or disproportionately, adding to the
societal cost.

Many of these conditions are challenging to study using traditional research techniques
because of a complex mix of factors:

- **Timescales** associated with chronic diseases can be far longer than those achievable
  in laboratory experiments.
- **Mimicking complex human hormonal cycling** and the sexual development time course is
  challenging in cell- and animal-based experiments.
- **Human reproductive tissues** are often unavailable for experimentation due to ethical
  concerns.

Wearables change the experimental unit from "a cohort in a lab for a week" to **a single
woman, in her own life, across years**. This repository builds the measurement and modeling
substrate for that shift.

## What this repository is

1. A **reference firmware + driver layer** for a spatially distributed, multi-modal
   wearable sensor network.
2. A **signal-processing and fusion pipeline** that converts raw micro-signals into
   interpretable physiological and psychological state estimates.
3. A **digital twin** — a per-person, continuously updated computational model of that
   woman's physiology and behavior.
4. A **webapp** that visualizes the digital twin and returns probabilistic, explainable
   states.
5. **Models** for interactive machine learning and for disentangling **aging** from
   **menopause**.

## The wearable

The wearable is designed for **at-home, continuous, unobtrusive monitoring** of the
parameters that shift during the perimenopausal transition. It captures **subtle
physiological changes and signals** that are **spatially distributed within the body** —
hormonal axis surrogates, thermoregulatory instability (hot flushes / night sweats),
autonomic shifts, sleep architecture, and mood/affect proxies — and fuses them into a
coherent, person-specific trajectory.

### Sensing the subtle: sensor suite

| Domain | Modality | What it captures | Candidate sensitive technology |
| --- | --- | --- | --- |
| Thermoregulation | Skin temp. gradient (multiple sites) | Hot flush onset, nocturnal sweating | High-resolution micro-thermistor array (±0.01 °C) |
| Thermoregulation | Galvanic skin response (EDA) | Sudden sympathetic surges | Dry-electrode EDA / skin conductance |
| Cardiovascular | PPG (optical) | HR, HRV, pulse-wave morphology | Multi-wavelength PPG (green/IR/red) |
| Cardiovascular | Continuous BP trend | Vascular tone shift over time | PTT-based cuffless BP proxy |
| Electrodermal | Core/surface humidity | Sweat onset & rate | Capacitive humidity film |
| Neuro-autonomic | EEG-lite / EOG | Sleep staging, micro-arousals | Dry-EEG headband / EOG patch |
| Movement | 9-axis IMU | Sleep position, restlessness, tremor | IMU + accelerometer |
| Hormonal surrogate | Circadian skin-temp rhythm | Shift in thermoregulatory set-point | Derived from temp + actigraphy |
| Vascular | Local blood perfusion | Vasomotor tone | Multi-site PPG amplitude |
| Psychological | Voice prosody | Stress, fatigue, mood | On-device audio feature extraction |
| Psychological | Behavioral telemetry | Affect, engagement, routine | Phone/app usage + self-report |
| Biochemical (emerging) | Sweat/interstitial analytes | Cortisol/LH/FSH trend proxies | Sweat patch / microneedle (research-stage) |

> Sensor choices are **research-stage candidates**, not certified clinical instrumentation.
> See [`docs/sensors.md`](docs/sensors.md) for placement, sampling rates, and drift handling.

### Spatially distributed body sensing

The system treats the body as a **sensor field**, not a single point device. Nodes are
placed at **wrist, sternum, axilla, torso, and neck** so that *spatial gradients* (not just
magnitudes) become features:

- A **hot flush** is a wave of vasodilation and sweating whose **spatial propagation**
  (sternum → neck → face) is more informative than any single reading.
- **Autonomic surges** show up as time-locked EDA + HRV + peripheral PPG amplitude changes
  that only cohere when viewed across **multiple sites**.

This is the "new biophysics" angle: the discriminative signal lives in the **spatiotemporal
structure** of the body's response, not in one channel.

### Real-time psychological variables

"Psychological variable" here means **affective and cognitive state proxies** derived from
behavioral and physiological streams in real time — never a clinical psychiatric diagnosis.
Examples the pipeline estimates:

- perceived stress index,
- mood valence / arousal proxy,
- fatigue and cognitive-fog proxy,
- sleep continuity / fragmentation,
- symptom interference with daily life.

These are computed **on-device / on-phone** to preserve privacy, and exposed to the twin as
**probabilistic distributions** (see `src/menopause_wearable/sensors/psychological.py`).

## The digital twin assembly

The device is **part of a whole digital-twin assembly** — it is *not* a standalone gadget.
The digital twin is a continuously updated computational model of **one unique person**:

```
            ┌─────────────────────────────────────────────────────────┐
            │                    DIGITAL TWIN                          │
            │                                                          │
  Wearable ─┤  physiological state  ──┐                              │
  (this repo)│                        ├──► fused latent state ──►    │
            │  psychological state  ──┘        │                     │
            │                                  ▼                     │
  Phone app ─┤  behavioral telemetry ─► interactive ML ─► twin update │
  (this repo)│                                  │                     │
            │                                  ▼                     │
  FDA prior ─┤  ingested diagnostic     ─► probabilistic, explainable │
            │  reference (knowledge)        outputs (NOT certainty)   │
            └─────────────────────────────────────────────────────────┘
```

**The phone carries the twin.** In the same way a person moves through the world with a
phone whose apps hold a *compiled recursion* of what that person loves to watch on YouTube
or Facebook — the twin holds a *compiled recursion* of that person's physiology and mood,
updated every time a sensor sample lands. The twin travels with her; the wearable is its
highest-fidelity sense organ.

## The webapp (digital twin of an individual)

A **webapp that represents the digital twin of an individual has been developed** and ships
in this repository (`webapp/` + `src/menopause_wearable/api/app.py`). It:

- visualizes the twin's **live sensor field** (per-site temperature, EDA, HRV, etc.),
- shows the **fused latent state** over time (nightly/weekly),
- renders **probabilistic outputs with explanations** ("why this estimate"),
- lets the user **correct the model in the loop** (interactive ML),
- ingests **FDA menopausal diagnostic reference knowledge** as a prior — critically, it is
  **used to provide a *predictable* (forecast-like, uncertainty-quantified) diagnosis, NOT
  to derive a certainty.**

The app **must never present a certain result.** Every screen exposes uncertainty
(intervals, likelihoods, confidence bands) and a persistent disclaimer.

## FDA menopausal diagnosis as ingested prior — predictive, not certain

FDA-recognized menopausal diagnostic references (e.g. reproductive hormone panel
interpretation guidance, STRAW+10 staging concepts) are **ingested into the model as
structured prior knowledge** (`data/fda_menopause_reference.json`,
`src/menopause_wearable/models/fda_knowledge.py`).

The ingestion is deliberately constrained:

- It is a **prior**, not a verdict. Sensor evidence **updates** the prior into a
  posterior **distribution over perimenopausal stages**.
- Outputs are **predictive and uncertainty-quantified** — e.g. *"posterior P(late
  perimenopause) = 0.61, 80% interval [0.44, 0.77]"* — never **"you are in menopause."**
- The design goal is a **predictable diagnosis** (a forecast a clinician can act on and a
  woman can plan around), explicitly **not a certainty result**.

## Novel biophysics & data-driven analysis

Models tailored to a woman's complex physiology, covering:

- **Novel biophysics** — thermoregulatory set-point drift, vasomotor wave propagation,
  autonomic balance, and hormone-axis surrogates modeled as coupled dynamical systems.
- **Data-driven analysis** — learned representations over multi-site, multi-modal streams.

### Interactive machine learning

New **digital-twin computational models based on interactive machine learning**: the twin
learns from the woman's own labels, corrections, and feedback (self-report of flushes,
sleep, mood), and from clinician-in-the-loop review. The model is a **collaborator**, not a
black box — see `src/menopause_wearable/digital_twin/interactive_ml.py`.

### Disentangling aging from menopause

A core scientific problem: perimenopausal symptoms overlap with **normal aging**. This repo
implements **advanced modeling to disentangle the effects of aging from the effects of
menopause** using an **age-matched counterfactual / residualization** approach
(`src/menopause_wearable/models/aging_menopause.py`):

```
observed(t) = aging_component(age) + menopause_component(stage_prior) + person_random_effect + noise
```

The menopause component is estimated as the **residual** after removing the aging
component learned from age-matched reference trajectories — yielding a per-woman
"menopause-attributable" signal instead of a confounded one.

## Architecture

```
 sensors/ ──► signal/ ──► digital_twin/ ──► api/ ──► webapp/
   raw        fusion       per-person        REST     UI
 stream      features      state model       + WS
                                   ▲
                  models/          │
                  ├─ fda_knowledge │ (prior)
                  └─ aging_menopause (disentangle)
```

Details: [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```powershell
# 1. install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. run the digital-twin webapp
python -m menopause_wearable.api.app
# open http://127.0.0.1:8000

# 3. run the tests
pytest -q
```

No hardware required — a **synthetic sensor stream** simulates the wearable so the twin and
webapp run end-to-end out of the box.

## Repository layout

```
Menopause_Wearable_Device_Metric/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── config/sensors.yaml          # node placement, sampling, drift
├── firmware/esp32_sensor_node/  # reference embedded firmware
├── src/menopause_wearable/
│   ├── sensors/                 # physiological + psychological sensors
│   ├── signal/                  # filtering, fusion, feature extraction
│   ├── digital_twin/            # twin state + interactive ML
│   ├── models/                  # FDA prior, aging/menopause disentangling
│   └── api/                     # webapp backend (FastAPI)
├── webapp/                      # digital-twin UI (HTML/CSS/JS)
├── data/fda_menopause_reference.json
├── docs/                        # architecture, sensors, clinical disclaimer
└── tests/
```

## The device is one node in a larger whole

This device is **a part of the whole digital-twin assembly**. The app is already developed
to be used **with this device** to create a **virtual twin of a unique person** on the
platform. Think of it as someone moving around with their phone: the apps on it already
hold a *compiled recursion* of what that person loves watching on YouTube or Facebook. The
twin is the physiological/psychological analogue of that recursion — portable, personal,
continuously compiled.

## Roadmap

- [ ] Real sweat/interstitial hormone-surrogate patch (research collaboration).
- [ ] Federated learning so twins improve without centralizing sensitive data.
- [ ] Clinician review console for interactive ML labels.
- [ ] Validation protocol aligned with STRAW+10 staging endpoints.
- [ ] Wearable-grade power/thermal envelope and FCC/CE pathway (non-diagnostic class).

## Disclaimer

This software and the described device are **research tools**. They are **not** intended to
diagnose, treat, cure, or prevent any disease and have **not** been cleared or approved by
the FDA or any regulator as a medical device. All outputs are **probabilistic estimates**
and **must not be interpreted as a certainty or a clinical diagnosis.** Always consult a
qualified healthcare professional. See [`docs/clinical_disclaimer.md`](docs/clinical_disclaimer.md).

## References

- STRAW+10 staging of reproductive aging (Harlow et al.).
- FDA guidance on reproductive hormone diagnostics (as ingested prior reference).
- Literature on vasomotor symptom physiology and wearable thermography.
- Literature on digital twins in health and interactive machine learning.

## License

MIT — see [LICENSE](LICENSE).
