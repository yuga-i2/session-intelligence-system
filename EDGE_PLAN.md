# EDGE PLAN
## Deployment on Mobile / On-Device

---

## The Constraint

A wellness app that takes 3+ seconds to respond after a session has a product problem.
Target: inference in **< 300ms** on a mid-range Android phone.

---

## Current Model Size (Development)

| Component | Size | Notes |
|---|---|---|
| RandomForest (200 trees, depth 8) | ~8–15 MB serialized | Too large for mobile |
| TF-IDF vocabulary (300 features) | ~50 KB | Acceptable |
| Feature engineering code | Negligible | Pure Python logic |
| Full pipeline (pickled) | ~12–18 MB | Needs compression |

---

## Deployment Strategy

### Option A — Compressed On-Device (Recommended)

Reduce the model to something that actually fits and runs fast on a phone.

**Steps:**

1. **Shrink the Random Forest**
   - Drop from 200 trees → 50 trees
   - Reduce max_depth from 8 → 5
   - This cuts model size by ~75% with roughly 5–8% F1 loss — acceptable tradeoff
   - Estimated size after: ~2–3 MB

2. **Quantize**
   - Export to ONNX format, apply int8 quantization
   - Cuts another 50% off size
   - ONNX Runtime Mobile handles inference on Android/iOS natively

3. **Limit TF-IDF vocabulary**
   - Current: 300 features
   - Mobile: trim to top 100 most informative terms
   - Run mutual information selection against emotional_state labels to pick the 100 that matter

4. **Decision engine runs fully on-device**
   - The decision logic is pure Python/JS — no model needed
   - Can be ported to Kotlin/Swift directly
   - Zero latency, zero network dependency

**Expected mobile performance:**
- Model size: ~2–3 MB
- Inference latency: 50–150ms on mid-range Android
- F1 drop vs full model: ~5–8%

---

### Option B — Lightweight Server + Cache (Fallback)

If on-device inference isn't feasible for a v1 launch:

- Run inference on a small FastAPI server (can run on a $5 VPS)
- Cache predictions for repeated/similar inputs
- Acceptable latency: < 500ms with good connection
- Fallback: if no network, use rule-based decision engine only (no ML state prediction)

---

## What Runs Fully Offline

Even without the ML model, the system can still function:

| Component | Offline? | Notes |
|---|---|---|
| Text signal extraction | ✅ Yes | Pure keyword matching |
| Uncertainty detection | ✅ Yes | Rule-based |
| Decision engine (what + when) | ✅ Yes | Behavioral rules only |
| Emotional state prediction | ❌ Needs model | Degrade to rule-based fallback |
| Intensity prediction | ❌ Needs model | Default to stress_level proxy |

**Offline fallback logic:**
```
if no model available:
    use stress_level + energy_level + text signals → rough state estimate
    run decision engine as normal
    flag confidence as 'low' automatically
```

---

## Tradeoffs

| Approach | Pros | Cons |
|---|---|---|
| Full on-device (compressed RF) | Fast, private, offline | Slightly lower accuracy, harder to update |
| Lightweight server | Easy to update model | Network dependency, latency |
| Rule-based only offline | Zero dependencies | Misses nuanced patterns |
| Hybrid (model on-device + rules fallback) | Best of both | More complex to maintain |

**Recommendation:** Hybrid. Ship compressed model on-device for the core prediction. Rules-based fallback when model unavailable. Server sync for model updates weekly.

---

## Privacy Note

Journal text is sensitive health-adjacent data. On-device inference means:
- Text never leaves the phone
- No server logs of session content
- GDPR/HIPAA friendlier

This is a product advantage, not just a technical one.

---

## Update Strategy

- Model retrained server-side as new data accumulates
- Distributed as a small binary update (< 5 MB delta)
- Users get improved predictions without app store update
- Old model stays active until new one verified

---

## What This Probably Won't Do Well

- Very old / low-end phones (< 2GB RAM): the ONNX model might still be too heavy. Would need to fall back to rules-only mode.
- First-time users with no history: without previous_day_mood and session history, predictions are weaker. The uncertainty flag handles this gracefully.
- Languages other than English: the TF-IDF and signal vocabulary is English-only. Full re-training needed for other languages.
