# Sensors

Sensor choices are **research-stage candidates**, not certified clinical
instrumentation. The design principle: capture **subtle** changes via
**spatially distributed** nodes so the discriminative information is in the
**spatiotemporal structure** of the body's response.

## Placement

| Node id | Body site | Sensors |
| --- | --- | --- |
| `sternum` | sternum | temperature, PPG, humidity |
| `neck` | neck | temperature |
| `axilla` | left axilla | temperature, humidity |
| `headband` | forehead | dry-EEG, EOG, temperature |
| `right_wrist` | right wrist | temperature, IMU |
| `left_wrist` | left wrist | PPG, EDA, IMU, temperature |

Propagation path for vasomotor waves: `sternum → neck → axilla → forehead →
right_wrist → left_wrist`.

## Modalities and why

- **Skin temperature array (±0.01 °C).** Hot flushes are a wave of peripheral
  vasodilation; the *gradient across sites* and its *propagation delay* matter
  more than any single value.
- **Electrodermal activity (EDA).** Fast sympathetic marker; phasic peaks track
  autonomic surges that accompany flushes and stress.
- **PPG (multi-wavelength).** HR, HRV, and pulse-wave morphology; HRV is a
  robust stress/autonomic proxy.
- **Humidity film.** Sweat onset and rate; corroborates thermal events.
- **Dry-EEG / EOG.** Sleep staging and micro-arousals — night sweats fragment
  sleep.
- **IMU.** Sleep position, restlessness, tremor.
- **Behavioral telemetry (phone).** The "compiled recursion" of daily life used
  for psychological proxies — computed locally.

## Sampling (defaults)

See `config/sensors.yaml`: temperature 4 Hz, EDA 8 Hz, PPG 64 Hz, IMU 32 Hz,
EEG 128 Hz. Analysis windows are 60 s with a 10 s stride.

## Drift and artifacts

- Slow baseline drift is removed with rolling baselines (temp 30 min, EDA 10
  min, PPG 5 min).
- Motion artifact on PPG is flagged via IMU co-movement.
- Node placement quality is logged; gradients require ≥ 3 co-located sites.

## Open engineering questions

- Power/thermal envelope for all-day wear.
- Skin-contact stability of dry electrodes over days.
- Cross-device clock alignment for propagation-delay estimation (target < 100 ms).
- Regulatory pathway for a non-diagnostic wellness device (FCC/CE; not FDA-cleared).
