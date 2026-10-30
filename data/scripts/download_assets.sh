#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$BASE_DIR/data"

echo "=== SignBridge Asset Downloader ==="
echo "Target Directory: $DATA_DIR"

mkdir -p "$DATA_DIR/include_poses"
mkdir -p "$DATA_DIR/include_metadata"
mkdir -p "$DATA_DIR/checkpoints"

# 1. Clone official AI4Bharat/OpenHands repository
echo "[1/4] Checking AI4Bharat/OpenHands Repository..."
if [ ! -d "$BASE_DIR/openhands_repo" ]; then
    echo "Cloning AI4Bharat/OpenHands from GitHub..."
    git clone https://github.com/AI4Bharat/OpenHands.git "$BASE_DIR/openhands_repo"
else
    echo "AI4Bharat/OpenHands repository already cloned."
fi

# 2. Download INCLUDE metadata
echo "[2/4] Downloading INCLUDE Metadata..."
if [ ! -f "$DATA_DIR/include_metadata/include_metadata.zip" ]; then
    curl -L "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_metadata.zip" \
         -o "$DATA_DIR/include_metadata/include_metadata.zip"
    unzip -q -o "$DATA_DIR/include_metadata/include_metadata.zip" -d "$DATA_DIR/include_metadata/"
    echo "INCLUDE metadata extracted."
else
    echo "INCLUDE metadata already exists."
fi

# 3. Download OpenHands Checkpoints for INCLUDE
echo "[3/4] Downloading OpenHands Checkpoints for INCLUDE..."
# BERT checkpoint (primary)
if [ ! -f "$DATA_DIR/checkpoints/include_bert.zip" ]; then
    echo "Downloading INCLUDE BERT checkpoint..."
    curl -L "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_bert.zip" \
         -o "$DATA_DIR/checkpoints/include_bert.zip"
    unzip -q -o "$DATA_DIR/checkpoints/include_bert.zip" -d "$DATA_DIR/checkpoints/"
fi

# ST-GCN checkpoint (secondary/fallback)
if [ ! -f "$DATA_DIR/checkpoints/include_stgcn.zip" ]; then
    echo "Downloading INCLUDE ST-GCN checkpoint..."
    curl -L "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_stgcn.zip" \
         -o "$DATA_DIR/checkpoints/include_stgcn.zip"
    unzip -q -o "$DATA_DIR/checkpoints/include_stgcn.zip" -d "$DATA_DIR/checkpoints/"
fi
echo "Checkpoints downloaded and extracted."

# 4. Download INCLUDE Pose Data
echo "[4/4] Downloading INCLUDE Pose Data (~0.6 GB)..."
if [ ! -f "$DATA_DIR/include_poses/INCLUDE.zip" ]; then
    curl -L "https://zenodo.org/record/6674324/files/INCLUDE.zip?download=1" \
         -o "$DATA_DIR/include_poses/INCLUDE.zip"
    unzip -q -o "$DATA_DIR/include_poses/INCLUDE.zip" -d "$DATA_DIR/include_poses/"
    echo "INCLUDE pose data extracted."
else
    echo "INCLUDE pose data already exists."
fi

if [ -f "$DATA_DIR/include_poses/INCLUDE/Pose_Signs.zip" ] && [ ! -d "$DATA_DIR/include_poses/INCLUDE/Pose_Signs" ]; then
    echo "Extracting inner Pose_Signs.zip..."
    unzip -q -o "$DATA_DIR/include_poses/INCLUDE/Pose_Signs.zip" -d "$DATA_DIR/include_poses/INCLUDE/"
    unzip -q -o "$DATA_DIR/include_poses/INCLUDE/Train_Test_Split.zip" -d "$DATA_DIR/include_poses/INCLUDE/"
    echo "Pose_Signs unzipped successfully."
fi

echo "=== All assets downloaded successfully! ==="

# 5. Install OpenHands as a local package and build KNN index
echo "[5/5] Installing OpenHands from openhands_repo ..."
pip install -q -e "$BASE_DIR/openhands_repo/" && echo "OpenHands installed." || echo "WARNING: pip install failed — check Python environment."

echo ""
echo "=== IMPORTANT: KNN Index Build ==="
echo "The KNN index (knn_embeddings.npy + knn_labels.npy) must be built once"
echo "from the INCLUDE training data. Without it, the recognizer falls back"
echo "to the motion-hash placeholder."
echo ""
echo "Run:"
echo "  cd $BASE_DIR && python3 data/scripts/build_knn_index.py"
echo ""
echo "This takes ~20-60 minutes on CPU. Do it once and commit the result"
echo "(or store it as an artifact alongside the checkpoint)."

if [ "${AUTO_BUILD_KNN:-false}" = "true" ]; then
    echo "AUTO_BUILD_KNN=true — starting KNN index build now ..."
    cd "$BASE_DIR" && python3 data/scripts/build_knn_index.py
fi

