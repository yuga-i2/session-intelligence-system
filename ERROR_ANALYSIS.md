# ERROR ANALYSIS
## Arvyax Session Intelligence System

> "A model that knows where it fails is more useful than one that doesn't."

---

## Overview

10 failure cases analysed from the training set using leave-one-out inspection.
Each case shows: what went wrong, why the model failed, and what would fix it.

---

## Case 1 — ID 24 | True: `mixed` → Predicted: `neutral`

**Text:** *"I liked the ocean session but my mood is still split between calm and tension. idk."*

**What went wrong:** The word "split" and the phrase "calm and tension" together are the clearest possible indicators of a mixed state. The model predicted neutral.

**Why:** "calm and tension" don't individually fire the tension or ambivalent signal markers strongly enough. "idk" at the end should raise uncertainty but instead suppresses the overall signal, nudging toward neutral.

**Fix:** Add "split between" and "two moods" as explicit ambivalent markers. Weight `idk` as an uncertainty flag rather than a negative sentiment anchor.

---

## Case 2 — ID 83 | True: `mixed` → Predicted: `focused`

**Text:** *"I started scattered but the ocean session helped me lock in on what matters. My to-do list feels less chaotic."*

**What went wrong:** The text ends on a positive, resolved note — "lock in", "less chaotic". The model correctly sees resolution but misses that the entry starts with "scattered", meaning the person arrived in a mixed state and partially resolved it. True label is `mixed`.

**Why:** Bag-of-words can't read narrative arc. "started scattered... but helped me lock in" is a before/after story. The model only sees tokens, not the trajectory.

**Fix:** Sentence-position weighting — first-sentence signals and last-sentence signals should carry different weight. The opening describes the entry state; the ending describes the outcome.

---

## Case 3 — ID 95 | True: `restless` → Predicted: `neutral`

**Text:** *"I noticed the rain sounds but emotionally I still feel mostly the same. idk."*

**What went wrong:** "mostly the same" and "idk" pulled the prediction toward neutral. True label is restless — the person was restless going in and the session didn't shift anything.

**Why:** The model doesn't have access to the *starting* emotional state. "Mostly the same" means restless → restless, but the model reads it as flat → neutral. Without previous session context, this is essentially unresolvable from text alone.

**Fix:** Include previous_day_mood more aggressively in the feature space. A `previous_day_mood = restless` + "mostly the same" text should predict restless, not neutral.

---

## Case 4 — ID 7 | True: `neutral` → Predicted: `restless`

**Text:** *"Nothing strong came up during the rain session; I feel fairly normal. At least I paused for a moment."*

**What went wrong:** "Nothing strong" and "fairly normal" are disengaged signals. Model predicted restless, probably because the reflection_quality was `conflicted`.

**Why:** `conflicted` reflection quality and `calm` previous day mood created a pull toward restless. The text itself is clearly neutral, but the metadata contradicted it.

**Fix:** When text signals are strong and unambiguous (disengaged markers firing clearly), text should outweigh conflicting metadata. The current weighting is roughly equal.

---

## Case 5 — ID 37 | True: `mixed` → Predicted: `neutral`

**Text:** *"I feel both comforted and distracted after the cafe ambience. I can't tell if I need rest or momentum."*

**What went wrong:** "comforted and distracted" and "can't tell if I need rest or momentum" are textbook mixed-state language. Predicted neutral.

**Why:** "comforted" isn't in the resolved markers list. "Distracted" isn't in the tension list. The explicit contradiction is there in natural language but not captured by the current signal vocabulary.

**Fix:** Add "comforted and distracted", "rest or momentum", "can't tell what I need" as explicit mixed-state phrases.

---

## Case 6 — ID 65 | True: `neutral` → Predicted: `mixed`

**Text:** *"after the mountain sounds i feel better than before but not completely okay. it's like two moods are sitting together."*

**What went wrong:** "two moods sitting together" is ambivalent language — should be mixed. But the label is neutral. This is a noisy label case.

**Why:** Likely label timing drift — the person may have rated their state before writing, when they felt more settled. The writing itself reveals residual tension. The model is arguably *more correct* than the label here.

**Fix:** This isn't really a model fix — it's a data quality issue. Better labeling protocol: always label after writing the reflection, not before.

---

## Case 7 — Short Entry — ID 91 | True: `neutral` → Predicted: `restless`

**Text:** *"The cafe track was fine. I feel steady not especially better or worse."*

**What went wrong:** "cafe" ambience + `overwhelmed` previous day mood + low duration (20min) pushed prediction toward restless. Text says clearly neutral/flat.

**Why:** With short, flat text, the model relies more heavily on metadata. Previous overwhelmed mood + cafe ambience misled it.

**Fix:** For short, disengaged entries, down-weight previous_day_mood and ambience features. The text is the primary signal here even if it's brief.

---

## Case 8 — Contradiction Entry — ID 46 | True: `mixed` → Predicted: `neutral`

**Text:** *"The mountain track helped a little though something still feels off underneath. There is relief but also some lingering pressure."*

**What went wrong:** "relief but also some lingering pressure" — this is the clearest possible mixed-state description. Predicted neutral.

**Why:** "Relief" isn't in the resolved markers. "Lingering pressure" contains "pressure" (tension marker) but the full phrase is softer than the individual word suggests. The model sees low signal strength and defaults to neutral.

**Fix:** Add phrase-level matching for "relief but" and "feels off underneath" as mixed-state indicators.

---

## Case 9 — High Stress Mislabeled — ID 116 | True: `neutral` → Predicted: `restless`

**Text:** *"Nothing strong came up during the rain session; I feel fairly normal. I can continue the day as usual."*

**What went wrong:** face_emotion_hint = `tense_face` despite the text being completely flat and neutral. Model trusted the tense face signal too much.

**Why:** The tense_face feature is strong in the model. When it fires alongside moderate stress, it pulls toward restless regardless of what the text says.

**Fix:** When text signals are explicitly disengaged (multiple disengaged markers), the face signal should be treated as background context, not a primary predictor.

---

## Case 10 — Missing Data — ID 64 | True: `focused` → Predicted: `neutral`

**Text:** *"The ocean background made it easier to organize my thoughts and work plan. I can see my priorities more clearly."*

**What went wrong:** This is clear focused-state language — "organize thoughts", "work plan", "priorities". Predicted neutral.

**Why:** `sleep_hours` is missing for this record. The `sleep_stress_ratio` feature gets imputed to median, which in this dataset pulls the feature toward a neutral zone. Combined with low intensity label (2), the model underweights the strong text signals.

**Fix:** When key text signals are strong and unambiguous (3+ resolved/action markers), intensity and sleep shouldn't override them. Add a text-confidence gate: if signal_strength > 0.5, prioritize text-derived features.

---

## Summary of Failure Patterns

| Pattern | Count | Fix Direction |
|---|---|---|
| Text says one thing, metadata says another | 4 | Context-aware feature weighting |
| Natural language not in signal vocabulary | 3 | Expand phrase-level markers |
| Noisy/timing-drifted labels | 1 | Better labeling protocol |
| Missing data confusing imputation | 1 | Text-confidence gate |
| Model can't read narrative arc | 1 | Sentence-position features |

---

## What Would Fix Most of This

1. **Phrase-level matching** over individual token matching — "relief but pressure", "split between calm and tension", "can't tell what I need"
2. **Text confidence gate** — when text signals are clear, reduce metadata influence proportionally
3. **Narrative arc features** — opening vs closing sentiment of entry (first 50% of words vs last 50%)
4. **Better label timing** — label after writing, not before
5. **More training data** — 120 records is genuinely too small; the model is overfitting on ambience and metadata patterns
