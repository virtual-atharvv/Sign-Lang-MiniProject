# SignBridge: Grammar-Aware Real-Time Indian Sign Language (ISL) Translator

SignBridge is a web-based real-time Indian Sign Language (ISL) translator that captures webcam video, extracts pose keypoints client-side, recognizes ISL glosses using pretrained pose models (AI4Bharat/OpenHands), and translates topic-comment gloss sequences into grammatically fluent spoken sentences (English/Hindi text + text-to-speech).

## Key Features
- **Client-Side Landmark Extraction**: MediaPipe Holistic JS extracts 75 keypoints in-browser for privacy & low latency.
- **Pretrained Sign Recognition**: Powered by OpenHands Transformer/ST-GCN models trained on the INCLUDE dataset.
- **Grammar Correction Engine**: Rule-based ISL gloss reordering engine (T5 fine-tuning scripts in `models/grammar_correction/` for future training; no trained checkpoint is currently deployed).
- **Deployable Architecture**: Frontend on Cloudflare Pages, backend on Hugging Face Spaces (FastAPI), with Cloudflare Worker API gateway and `cloudflared` local demo-day fallback. See `status.md` for current deployment readiness.

## Project Structure
- `data/`: Dataset scripts, pose metadata, and downloaded checkpoints.
- `models/`: OpenHands recognizer wrappers, keypoint converter, continuous segmenter, and T5 grammar fine-tuning scripts.
- `backend/`: FastAPI backend service exposing real-time prediction endpoints.
- `frontend/`: Single-page static web application for live webcam translation.
- `worker/`: Cloudflare Worker API gateway code.
- `data/scripts/evaluate_knn.py`: Full test-set evaluation script (top-1 and top-3 accuracy).
