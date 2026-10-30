#!/usr/bin/env python3
"""
evaluate_knn.py — Honest KNN accuracy evaluation on the INCLUDE test split.

Usage (from project root):
    python3 data/scripts/evaluate_knn.py
    python3 data/scripts/evaluate_knn.py --restrict-classes 30   # demo-scoped vocab

Reports top-1 and top-3 accuracy across all available test-split pkl files.
Writes full per-sample log to knn_test_output.txt.
"""

import os
import sys
import csv
import pickle
import argparse
import numpy as np
from collections import defaultdict

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
OPENHANDS_REPO = os.path.join(PROJECT_ROOT, "openhands_repo")

if OPENHANDS_REPO not in sys.path and os.path.exists(OPENHANDS_REPO):
    sys.path.insert(0, OPENHANDS_REPO)
sys.path.insert(0, PROJECT_ROOT)

CKPT_DIR        = os.path.join(PROJECT_ROOT, "data/checkpoints/include/st_gcn")
CHECKPOINT_PATH = os.path.join(CKPT_DIR, "epoch=215-step=23543.ckpt")
CONFIG_PATH     = os.path.join(CKPT_DIR, "config.yaml")
KNN_EMB_PATH    = os.path.join(CKPT_DIR, "knn_embeddings.npy")
KNN_LBL_PATH    = os.path.join(CKPT_DIR, "knn_labels.npy")
# test_include.csv columns: Category, Word, Video, FilePath
TEST_CSV        = os.path.join(PROJECT_ROOT, "data/include_metadata/Train_Test_Split/test_include.csv")
POSE_ROOT       = os.path.join(PROJECT_ROOT, "data/include_poses/INCLUDE/Pose_Signs")
OUTPUT_FILE     = os.path.join(PROJECT_ROOT, "knn_test_output.txt")


def parse_label_id(word: str) -> int:
    """'1. loud' → 1,  '4.sad' → 4,  '94. good' → 94"""
    return int(word.split('.')[0].strip())


def load_test_samples() -> list[tuple[int, str, str]]:
    """
    Returns list of (label_id, gloss_word, full_pkl_path) for every test CSV row.
    FilePath column (.MOV) is mapped to .pkl in POSE_ROOT.
    """
    samples = []
    with open(TEST_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word     = row['Word'].strip()
            filepath = row['FilePath'].strip()
            try:
                label_id = parse_label_id(word)
            except ValueError:
                continue
            pkl_rel   = os.path.splitext(filepath)[0] + '.pkl'
            full_path = os.path.join(POSE_ROOT, pkl_rel)
            samples.append((label_id, word, full_path))
    return samples


def load_model_and_transforms():
    import torch, omegaconf
    from openhands.apis.inference import InferenceModel
    from openhands.datasets.pose_transforms import (
        Compose, PoseSelect, PoseUniformSubsampling, CenterAndScaleNormalize
    )

    _orig = torch.load
    def _safe(*a, **kw):
        kw['weights_only'] = False
        return _orig(*a, **kw)
    torch.load = _safe

    cfg = omegaconf.OmegaConf.load(CONFIG_PATH)
    cfg.pretrained = CHECKPOINT_PATH
    cfg.data.valid_pipeline.dataset.root_dir   = POSE_ROOT
    cfg.data.valid_pipeline.dataset.split_file = TEST_CSV
    cfg.data['test_pipeline'] = cfg.data['valid_pipeline']

    model = InferenceModel(cfg=cfg, stage='test')
    ckpt  = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.model.eval()

    encoder     = model.model.encoder
    id_to_gloss = model.datamodule.test_dataset.id_to_gloss

    transforms = Compose([
        PoseSelect(preset='mediapipe_holistic_minimal_27'),
        PoseUniformSubsampling(num_frames=120),
        CenterAndScaleNormalize(
            reference_points_preset='shoulder_mediapipe_holistic_minimal_27'
        ),
    ])
    return encoder, transforms, id_to_gloss


def embed_pkl(path, encoder, transforms) -> np.ndarray:
    import torch
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    kps    = raw['keypoints'][:, :, :2].astype('float32')
    tensor = torch.tensor(kps, dtype=torch.float32).permute(2, 0, 1)
    out    = transforms({'frames': tensor})
    inp    = out['frames'].unsqueeze(0)
    with torch.no_grad():
        emb = encoder(inp).squeeze(0).numpy().astype('float32')
    return emb


def knn_predict(emb, knn_embs_n, knn_labels, id_to_gloss, k=5):
    norm  = np.linalg.norm(emb)
    emb_n = emb / max(norm, 1e-8)
    sims  = knn_embs_n @ emb_n
    top_k = np.argsort(sims)[::-1][:k]

    votes = defaultdict(float)
    for idx in top_k:
        votes[int(knn_labels[idx])] += max(float(sims[idx]), 0.0)

    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    total        = sum(w for _, w in sorted_votes) or 1.0

    top1_id    = sorted_votes[0][0]
    top1_name  = id_to_gloss.get(top1_id, f"CLASS_{top1_id}")
    top1_conf  = sorted_votes[0][1] / total
    top3_ids   = [v[0] for v in sorted_votes[:3]]
    top3_names = [id_to_gloss.get(i, f"CLASS_{i}") for i in top3_ids]
    nbrs       = [(id_to_gloss.get(int(knn_labels[idx]), '?'), float(sims[idx]))
                  for idx in top_k[:3]]
    return top1_name, top1_conf, top3_names, nbrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restrict-classes", type=int, default=0,
                        help="Restrict to N most-represented classes (0 = all)")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    for path, desc in [
        (KNN_EMB_PATH,    "KNN embeddings  → run build_knn_index.py first"),
        (KNN_LBL_PATH,    "KNN labels      → run build_knn_index.py first"),
        (CHECKPOINT_PATH, "ST-GCN checkpoint"),
        (TEST_CSV,        "Test split CSV"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: Missing {desc}:\n  {path}")
            sys.exit(1)

    print("Loading model …")
    encoder, transforms, id_to_gloss = load_model_and_transforms()
    print(f"  {len(id_to_gloss)} classes")

    print("Loading KNN index …")
    knn_embs   = np.load(KNN_EMB_PATH).astype('float32')
    knn_labels = np.load(KNN_LBL_PATH).astype('int32')
    norms      = np.linalg.norm(knn_embs, axis=1, keepdims=True)
    knn_embs_n = knn_embs / np.maximum(norms, 1e-8)
    print(f"  Index: {knn_embs.shape[0]} samples, {knn_embs.shape[1]}-dim")

    print("Loading test samples …")
    all_samples = load_test_samples()
    available   = [(lid, w, p) for lid, w, p in all_samples if os.path.exists(p)]
    missing_ct  = len(all_samples) - len(available)
    print(f"  CSV rows: {len(all_samples)}   Available on disk: {len(available)}   Missing: {missing_ct}")

    if not available:
        print("ERROR: No test pkl files found. Check POSE_ROOT path.")
        sys.exit(1)

    if args.restrict_classes > 0:
        from collections import Counter
        class_counts = Counter(lid for lid, _, _ in available)
        top_classes  = set(cid for cid, _ in class_counts.most_common(args.restrict_classes))
        available    = [(lid, w, p) for lid, w, p in available if lid in top_classes]
        print(f"  Restricted to {args.restrict_classes} classes: {len(available)} samples")

    lines        = []
    top1_correct = top3_correct = total = 0

    header = (
        f"KNN Evaluation — {len(available)} test samples, k={args.k}\n"
        f"Model: {os.path.basename(CHECKPOINT_PATH)}\n"
        f"Index: {knn_embs.shape[0]} train embeddings, {knn_embs.shape[1]}-dim\n"
    )
    if args.restrict_classes > 0:
        header += f"Vocabulary restricted to {args.restrict_classes} most common classes\n"
    header += ""
    lines.append(header)

    for i, (label_id, word, path) in enumerate(available):
        try:
            emb = embed_pkl(path, encoder, transforms)
            pred, conf, top3, nbrs = knn_predict(emb, knn_embs_n, knn_labels, id_to_gloss, args.k)
        except Exception as e:
            lines.append(f"  [ERR ] {word} | {os.path.basename(path)} | {e}")
            continue

        gt_clean   = word.split('. ')[-1].upper()
        pred_clean = pred.split('. ')[-1].upper()
        top3_clean = [t.split('. ')[-1].upper() for t in top3]

        is_top1 = gt_clean == pred_clean
        is_top3 = gt_clean in top3_clean
        top1_correct += is_top1
        top3_correct += is_top3
        total        += 1

        sym = 'OK  ' if is_top1 else ('TOP3' if is_top3 else 'FAIL')
        lines.append(f"  [{sym}] GT={word} | Pred={pred_clean} | conf={conf:.3f}")
        for nbr_g, nbr_s in nbrs:
            lines.append(f"         nbr: {nbr_g} sim={nbr_s:.4f}")

        if (i + 1) % 25 == 0:
            pct = 100 * top1_correct / total if total else 0
            print(f"  {i+1}/{len(available)}  Top-1: {top1_correct}/{total} ({pct:.1f}%)")

    summary = (
        f"\nTop-1 Accuracy: {top1_correct}/{total}  "
        f"({100*top1_correct/total:.1f}%)\n"
        f"Top-3 Accuracy: {top3_correct}/{total}  "
        f"({100*top3_correct/total:.1f}%)\n"
        f"\nNOTE: Results from test split only; no training data in evaluation.\n"
    )
    if missing_ct > 0:
        summary += f"NOTE: {missing_ct} test file(s) not on disk — excluded.\n"

    lines.append(summary)
    report = "\n".join(lines)

    print("\n" + "─" * 60)
    print(report)

    with open(OUTPUT_FILE, 'w') as f:
        f.write(report)
    print(f"✅ Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
