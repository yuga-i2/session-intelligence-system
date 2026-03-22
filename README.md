# Arvyax Session Intelligence System
### ML Internship Assignment — RevoltronX / Team ArvyaX

---

## Quick Start

```bash
pip install -r requirements.txt
python arvyax_system.py
```

Outputs `predictions.csv` with all 120 test predictions. No internet required. Runs fully locally.

---

## What This Builds

Not a classifier. A reasoning system:

```
messy journal text + context signals
        ↓
  interpretation  (what state is the person in?)
        ↓
    judgment      (how confident are we? any contradictions?)
        ↓
    decision      (what should they do, and when?)
```

---

## Files

| File | Purpose |
|---|---|
| `arvyax_system.py` | Full pipeline — run this |
| `predictions.csv` | Output: 120 test predictions |
| `README.md` | This file |
| `ERROR_ANALYSIS.md` | 10 failure cases with honest analysis |
| `EDGE_PLAN.md` | Mobile / on-device deployment strategy |
| `arvyax_system_design.pdf` | Full system design document |

---

## Approach

### Part 1 — Emotional State Prediction
- **Model:** Random Forest (200 trees, depth 8)
- **Why RF over LR:** Non-linear interactions matter — `high energy + ambivalent text` behaves differently from `low energy + ambivalent text`. Logistic regression misses that.
- **CV F1:** ~0.49 (5-fold, weighted) on 120 training records
- **Features:** text signals + structured metadata + categoricals

### Part 2 — Intensity Prediction
- **Treated as regression**, not classification
- **Why:** Intensity is ordinal (1–5). Regression respects the ordering; classification treats 1 and 5 as equally distant from 3.
- **Model:** Random Forest Regressor
- **CV MAE:** ~0.91 (±0.15)

### Part 3 — Decision Engine
Behavioral patterns, not if-else rules:

| State + Context | What to Do | When |
|---|---|---|
| overwhelmed + low energy | rest | later_today / tonight |
| overwhelmed + high stress | box_breathing | now |
| restless + high energy + action urge | deep_work | now / within_15_min |
| restless + high stress | box_breathing | within_15_min |
| focused + adequate energy | deep_work | now |
| mixed + contradiction | grounding | within_15_min |
| mixed + rest urge | rest | later_today |
| neutral + low energy | movement | now / within_15_min |
| short entry + low confidence | journaling | within_15_min |
| stress ≥ 5 (any state) | box_breathing | now |

Time-of-day adjusts the **when**: morning sessions get `now` or `within_15_min`; evening sessions shift rest to `tonight`; low-intensity entries in late night shift to `tomorrow_morning`.

### Part 4 — Uncertainty Modeling
Confidence is not just model probability. It's a blend of:
- Model's predicted class probability
- Uncertainty flags: very short entry, multiple hedges, contradictory signals, missing data, no face signal

```
final_confidence = model_confidence × (0.5 + 0.5 × uncertainty_modifier)
```

`uncertain_flag = 1` when any of these fires. The system says "I'm not sure" rather than pretending otherwise.

### Part 5 — Feature Understanding

**Most important features (by RF importance):**
1. `energy_stress_gap` — direction of capacity vs load
2. `sleep_stress_ratio` — sleep quality adjusted for stress
3. `stress_level` — strongest single structured predictor
4. `sleep_hours`, `energy_level`, `duration_min`
5. `tension_resolution_ratio` — did the session land?
6. `has_action_urge`, `has_resolved` — text signals

**Text vs metadata:** Metadata adds ~+0.026 F1 over text-only (see ablation). Small but consistent. The face_emotion_hint and reflection_quality are the strongest non-text signals — when present, they often override what the text says.

### Part 6 — Ablation Study
```
Text-only model  — CV F1: 0.467
Text + metadata  — CV F1: 0.493
Gain:            + 0.026
```
Metadata helps, especially for short/vague entries where text gives almost nothing. For long expressive entries, text dominates.

### Part 7 — Error Analysis
See `ERROR_ANALYSIS.md` for 10 detailed failure cases.

Main failure patterns:
- Model can't read narrative arc ("started scattered **but** ended clear" → should be mixed, not focused)
- Natural language mixed-state phrases not in signal vocabulary
- Short entries over-rely on metadata, which can mislead
- ~1–2 noisy labels from timing drift (person rated mood before writing)

### Part 8 — Edge / Offline
See `EDGE_PLAN.md`.

Summary: compress RF to 50 trees + ONNX int8 → ~2–3 MB, <150ms on mid-range Android. Decision engine runs fully offline as rules. Hybrid approach recommended.

### Part 9 — Robustness

| Input Type | Handling |
|---|---|
| Very short ("ok", "fine") | `text_is_short` flag fires → routes to `journaling` + `uncertain_flag=1` |
| Missing values | Median imputation for numeric; "unknown" category for categoricals |
| Contradictory inputs | `contradiction_score` feature fires → often routes to `grounding` |
| Missing face signal | `no_face_signal` uncertainty reason → confidence penalty |
| Conflicting text + metadata | Confidence drops; system reports uncertainty rather than forcing a prediction |

---

## Setup

```bash
pip install -r requirements.txt
python arvyax_system.py
```

**Dependencies:** pandas, scikit-learn, scipy, numpy — all standard, all local.

---

## Output Format

`predictions.csv`:
```
id, predicted_state, predicted_intensity, confidence, uncertain_flag, what_to_do, when_to_do
10001, calm, 3, 0.559, 0, light_planning, later_today
10002, calm, 3, 0.275, 0, light_planning, later_today
...
```

---

*Submitted for RevoltronX / ArvyaX ML Internship — March 2026*
