#!/usr/bin/env python3
"""
build_knn_index.py — Build the KNN embedding index from the INCLUDE train split.

This is the MISSING PIECE for reproducible deployment:
- Reads train_include.csv to find all training pkl files
- Runs each sample through the ST-GCN encoder + transform chain
- Writes knn_embeddings.npy, knn_labels.npy, class_centroids.npy to the checkpoint dir

Usage (from project root):
    python3 data/scripts/build_knn_index.py

Prerequisites:
    - openhands_repo cloned  (run: data/scripts/download_assets.sh)
    - data/include_poses/INCLUDE/Pose_Signs/ populated
    - data/checkpoints/include/st_gcn/epoch=215-step=23543.ckpt present

Output:
    data/checkpoints/include/st_gcn/knn_embeddings.npy   — (N, 256) float32
    data/checkpoints/include/st_gcn/knn_labels.npy       — (N,)     int32
    data/checkpoints/include/st_gcn/class_centroids.npy  — (C, 256) float32
"""

import os
import sys
import csv
import pickle
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
OPENHANDS_REPO = os.path.join(PROJECT_ROOT, "openhands_repo")

if OPENHANDS_REPO not in sys.path and os.path.exists(OPENHANDS_REPO):
    sys.path.insert(0, OPENHANDS_REPO)
sys.path.insert(0, PROJECT_ROOT)

CKPT_DIR        = os.path.join(PROJECT_ROOT, "data/checkpoints/include/st_gcn")
CHECKPOINT_PATH = os.path.join(CKPT_DIR, "epoch=215-step=23543.ckpt")
CONFIG_PATH     = os.path.join(CKPT_DIR, "config.yaml")
# train_include.csv columns: Category, Word, Video, FilePath  (FilePath is .MOV, map to .pkl)
TRAIN_CSV       = os.path.join(PROJECT_ROOT, "data/include_metadata/Train_Test_Split/train_include.csv")
POSE_ROOT       = os.path.join(PROJECT_ROOT, "data/include_poses/INCLUDE/Pose_Signs")

OUT_EMBEDDINGS  = os.path.join(CKPT_DIR, "knn_embeddings.npy")
OUT_LABELS      = os.path.join(CKPT_DIR, "knn_labels.npy")
OUT_CENTROIDS   = os.path.join(CKPT_DIR, "class_centroids.npy")


def check_prerequisites():
    missing = []
    if not os.path.exists(OPENHANDS_REPO):
        missing.append(f"openhands_repo not found. Run: data/scripts/download_assets.sh")
    if not os.path.exists(CHECKPOINT_PATH):
        missing.append(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(CONFIG_PATH):
        missing.append(f"Config not found: {CONFIG_PATH}")
    if not os.path.exists(TRAIN_CSV):
        missing.append(f"Train CSV not found: {TRAIN_CSV}")
    if not os.path.isdir(POSE_ROOT):
        missing.append(f"Pose data not found: {POSE_ROOT}")
    if missing:
        print("ERROR — Prerequisites not met:")
        for m in missing:
            print(f"  ✗ {m}")
        sys.exit(1)
    print("✓ All prerequisites found.")


def parse_label_id(word: str) -> int:
    """
    Parse the integer label ID from the Word field.
    Examples: '1. loud' → 1,  '4.sad' → 4,  '94. good' → 94
    """
    return int(word.split('.')[0].strip())


def load_train_samples() -> list[tuple[int, str]]:
    """
    Parse train_include.csv and return list of (label_id, full_pkl_path).

    CSV format (with header):
        Category, Word, Video, FilePath
        Adjectives, 4.sad, MVI_9568.MOV, Adjectives/4. sad/MVI_9568.MOV

    .MOV files map to .pkl files in the same directory tree under POSE_ROOT.
    """
    samples = []
    with open(TRAIN_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word     = row['Word'].strip()
            filepath = row['FilePath'].strip()
            try:
                label_id = parse_label_id(word)
            except ValueError:
                continue
            # Map .MOV → .pkl
            pkl_rel  = os.path.splitext(filepath)[0] + '.pkl'
            full_path = os.path.join(POSE_ROOT, pkl_rel)
            samples.append((label_id, full_path))
    return samples


def load_model_and_transforms():
    import torch
    import omegaconf
    from openhands.apis.inference import InferenceModel
    from openhands.datasets.pose_transforms import (
        Compose, PoseSelect, PoseUniformSubsampling, CenterAndScaleNormalize
    )

    _orig_load = torch.load
    def _safe_load(*a, **kw):
        kw['weights_only'] = False
        return _orig_load(*a, **kw)
    torch.load = _safe_load

    cfg = omegaconf.OmegaConf.load(CONFIG_PATH)
    cfg.pretrained = CHECKPOINT_PATH
    cfg.data.valid_pipeline.dataset.root_dir   = POSE_ROOT
    cfg.data.valid_pipeline.dataset.split_file = TRAIN_CSV
    cfg.data['test_pipeline'] = cfg.data['valid_pipeline']

    print("Loading ST-GCN model …")
    model = InferenceModel(cfg=cfg, stage='test')
    ckpt  = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.model.eval()

    encoder      = model.model.encoder
    id_to_gloss  = model.datamodule.test_dataset.id_to_gloss
    print(f"  Model loaded: {len(id_to_gloss)} classes")

    transforms = Compose([
        PoseSelect(preset='mediapipe_holistic_minimal_27'),
        PoseUniformSubsampling(num_frames=120),
        CenterAndScaleNormalize(
            reference_points_preset='shoulder_mediapipe_holistic_minimal_27'
        ),
    ])
    return encoder, transforms, id_to_gloss


def embed_pkl(path: str, encoder, transforms) -> np.ndarray:
    """Load a .pkl pose file and return a 256-dim embedding. Input is in pixel coords."""
    import torch
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    kps    = raw['keypoints'][:, :, :2].astype('float32')          # (T, V, 2) pixel coords
    tensor = torch.tensor(kps, dtype=torch.float32).permute(2, 0, 1)  # (2, T, V)
    out    = transforms({'frames': tensor})
    inp    = out['frames'].unsqueeze(0)                             # (1, 2, 120, 27)
    with torch.no_grad():
        emb = encoder(inp).squeeze(0).numpy().astype('float32')    # (256,)
    return emb


def build_index():
    check_prerequisites()
    encoder, transforms, id_to_gloss = load_model_and_transforms()

    samples = load_train_samples()
    print(f"\nFound {len(samples)} entries in {TRAIN_CSV}")

    embeddings: list[np.ndarray] = []
    labels: list[int]            = []
    skipped = errors = 0

    for i, (label_id, path) in enumerate(samples):
        if not os.path.exists(path):
            skipped += 1
            if skipped <= 5:
                print(f"  SKIP (missing): {path}")
            continue
        try:
            emb = embed_pkl(path, encoder, transforms)
            embeddings.append(emb)
            labels.append(label_id)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR: {path} → {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(samples)}  "
                  f"embedded: {len(embeddings)}, skipped: {skipped}, errors: {errors}")

    if not embeddings:
        print("ERROR: No embeddings generated. Check POSE_ROOT and CSV paths.")
        sys.exit(1)

    emb_arr   = np.array(embeddings, dtype=np.float32)
    label_arr = np.array(labels,     dtype=np.int32)

    print(f"\n✓ Embeddings: {emb_arr.shape}")
    print(f"✓ Labels:     {label_arr.shape}  ({len(set(labels))} unique classes)")
    print(f"  Skipped: {skipped}   Errors: {errors}")

    # Class centroids
    class_ids = sorted(set(labels))
    max_id    = max(class_ids)
    centroids = np.zeros((max_id + 1, emb_arr.shape[1]), dtype=np.float32)
    for cid in class_ids:
        mask = label_arr == cid
        centroids[cid] = emb_arr[mask].mean(axis=0)

    np.save(OUT_EMBEDDINGS, emb_arr)
    np.save(OUT_LABELS,     label_arr)
    np.save(OUT_CENTROIDS,  centroids)

    print(f"\n✅ Saved to {CKPT_DIR}/")
    print(f"   knn_embeddings.npy  ({emb_arr.nbytes // 1024} KB)")
    print(f"   knn_labels.npy      ({label_arr.nbytes // 1024} KB)")
    print(f"   class_centroids.npy ({centroids.nbytes // 1024} KB)")
    print("\nRestart the backend to pick up the updated KNN index.")


if __name__ == "__main__":
    build_index()
