# SignBridge — Honest Status Report (Aug 8 2026)

This document is the single source of truth for the current state of the project. All claims here are backed by runnable commands or existing output files. Nothing is marked "done" unless it has actually been tested.

---

## Architecture Overview

SignBridge uses a three-stage pipeline:

1. **Keypoint Extraction** — MediaPipe Holistic JS (client-side, browser, no server round-trip for extraction)
2. **Sign Recognition** — AI4Bharat OpenHands ST-GCN encoder → 256-dim embedding → cosine-KNN lookup against prebuilt INCLUDE train-split index
3. **Grammar Correction** — Rule-based ISL gloss reordering (T5 fine-tuning scripts exist but no trained checkpoint is deployed)

---

## Current Accuracy

**KNN on ST-GCN encoder embeddings, evaluated on the full INCLUDE test split (813 samples, 263 classes):**

```
Top-1 Accuracy: 404/813  (49.7%)
Top-3 Accuracy: 585/813  (72.0%)
```

Source: `knn_test_output.txt` (run Aug 8 2026, `data/scripts/evaluate_knn.py`, all 813 available test pkl files)

**Important caveats:**
- The ST-GCN checkpoint (`epoch=215`) did not fully converge — raw decoder top-1 accuracy is ~16%. The KNN mitigation raises this to ~50%.
- The KNN index was built from 816 train-split embeddings (across 263 classes). More training samples per class would likely improve accuracy further.
- The frame-length fix (32-frame hard-cap removed; frontend buffer raised to 120) applies to the live webcam path only — these numbers are from dataset pkl files which are already full-length clips.
- A 30-class demo-scoped vocabulary would show higher accuracy (run: `python3 data/scripts/evaluate_knn.py --restrict-classes 30`).

---

## API Reference

**POST `/predict`**

Request:
```json
{
  "keypoints": [ { "pose": [...], "leftHand": [...], "rightHand": [...] } ],
  "mode": "continuous"
}
```

Response (actual Pydantic model — `backend/app/models.py`):
```json
{
  "glosses": ["HOME", "I", "GO"],
  "sentence": "I am going home.",
  "confidence": 0.72,
  "top_k": [["HOME", 0.72], ["SCHOOL", 0.14], ["..."]],
  "processing_time_ms": 42.1
}
```

**GET `/health`**

Response (actual Pydantic model — `backend/app/models.py`):
```json
{
  "status": "ok",
  "models_loaded": true,
  "recognizer_type": "OpenHands ST-GCN Neural Engine (263 INCLUDE Classes)",
  "neural_path_active": true,
  "encoder_loaded": true,
  "knn_loaded": true,
  "class_count": 263
}
```

Note: `neural_path_active` is `false` and `recognizer_type` changes to `"Fallback Motion-Hash"` if the encoder failed to load. `encoder_loaded` and `knn_loaded` are introspected from actual runtime state, not hardcoded.

---

## What Is Actually Deployed / Working

| Component | Status | Notes |
|---|---|---|
| MediaPipe keypoint extraction (browser) | ✅ Working | CDN-based, tested live |
| ST-GCN encoder load | ✅ Working locally | Requires `openhands_repo` on path |
| KNN cosine similarity lookup | ✅ Working locally | Requires `knn_embeddings.npy` + `knn_labels.npy` |
| Motion-hash fallback | ✅ Always available | Deterministic, not a real recognizer |
| Grammar rule-based fallback | ✅ Working | Handles known gloss patterns |
| Fine-tuned T5 grammar model | ❌ Not deployed | No checkpoint; training data being fixed |
| Docker container (clean build) | ⚠️ Needs openhands deps | `requirements.txt` and Dockerfile updated |
| Continuous segmentation | ✅ Working | `models/segmenter.py` |
| `/health` endpoint | ✅ Introspective | Reports real encoder/KNN load state |

---

## Known Issues / In Progress

- **Frame-length mismatch (fixed):** Pipeline previously hard-capped clips at 32 frames; model expects 120 via `PoseUniformSubsampling`. Hard-cap removed; `normalize_sequence(target_frames=120)` now used.
- **Grammar training data (fixed):** `build_grammar_pairs.py` had hardcoded third-person-singular verbs regardless of subject ("They is going", "I needs help"). Verb conjugation added; CSV regenerated.
- **KNN index reproducibility:** `data/scripts/build_knn_index.py` now exists to regenerate `knn_embeddings.npy` and `knn_labels.npy` from the INCLUDE train split.
- **Docker cold-start:** On first container start without pre-built KNN index, set `BUILD_KNN_ON_START=true` to auto-build (slow; ~30 min on CPU). Pre-building is recommended.

---

## Timing

End-to-end latency (local, CPU inference):
- Measured: ~400–600 ms per `/predict` call (CPU, 32-frame clips)
- Target: sub-500 ms for live demo use — achievable on GPU or with smaller clip windows
- **No sub-460ms claim is made** until measured on target hardware with the corrected 120-frame pipeline

---

*Last updated: Aug 8 2026. All numbers are from actual test runs.*
