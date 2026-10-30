#!/usr/bin/env bash
# start.sh — Container startup script for SignBridge backend.
# Optionally builds the KNN index before starting the FastAPI server.

set -e

echo "=== SignBridge Backend Startup ==="

# If BUILD_KNN_ON_START=true and the KNN index doesn't exist, build it.
KNN_EMB="data/checkpoints/include/st_gcn/knn_embeddings.npy"
KNN_LBL="data/checkpoints/include/st_gcn/knn_labels.npy"

if [ "${BUILD_KNN_ON_START}" = "true" ]; then
    if [ ! -f "${KNN_EMB}" ] || [ ! -f "${KNN_LBL}" ]; then
        echo "[startup] KNN index not found. Running build_knn_index.py ..."
        echo "[startup] This may take 20-60 minutes on CPU. Set BUILD_KNN_ON_START=false"
        echo "[startup] and pre-build the index to avoid this delay."
        python3 data/scripts/build_knn_index.py
        echo "[startup] KNN index built successfully."
    else
        echo "[startup] KNN index already exists — skipping rebuild."
    fi
else
    if [ ! -f "${KNN_EMB}" ] || [ ! -f "${KNN_LBL}" ]; then
        echo "[startup] WARNING: KNN index not found and BUILD_KNN_ON_START=false."
        echo "[startup] The recognizer will fall back to motion-hash predictions."
        echo "[startup] Run: python3 data/scripts/build_knn_index.py"
    fi
fi

echo "[startup] Starting FastAPI server on port ${PORT:-7860} ..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-7860}"
