import sys, numpy as np, torch, omegaconf, pickle, os
sys.path.insert(0, '/home/elliot/MiniProject')
sys.path.insert(0, '/home/elliot/MiniProject/openhands_repo')

orig = torch.load
def safe(*a,**kw): kw['weights_only']=False; return orig(*a,**kw)
torch.load = safe

cfg = omegaconf.OmegaConf.load('data/checkpoints/include/st_gcn/config.yaml')
cfg.pretrained = 'data/checkpoints/include/st_gcn/epoch=215-step=23543.ckpt'
cfg.data.valid_pipeline.dataset.root_dir = 'data/include_poses/INCLUDE/Pose_Signs'
cfg.data.valid_pipeline.dataset.split_file = 'data/include_metadata/Train_Test_Split/test_include.csv'
cfg.data['test_pipeline'] = cfg.data['valid_pipeline']

from openhands.apis.inference import InferenceModel
from openhands.datasets.pose_transforms import Compose, PoseSelect, PoseUniformSubsampling, CenterAndScaleNormalize

model = InferenceModel(cfg=cfg, stage='test')
ckpt = torch.load('data/checkpoints/include/st_gcn/epoch=215-step=23543.ckpt', map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=True)
model.model.eval()
enc = model.model.encoder
id_to_gloss = model.datamodule.test_dataset.id_to_gloss

transforms = Compose([
    PoseSelect(preset='mediapipe_holistic_minimal_27'),
    PoseUniformSubsampling(num_frames=120),
    CenterAndScaleNormalize(reference_points_preset='shoulder_mediapipe_holistic_minimal_27'),
])

knn_embs = np.load('data/checkpoints/include/st_gcn/knn_embeddings.npy').astype('float32')
knn_labels = np.load('data/checkpoints/include/st_gcn/knn_labels.npy').astype('int32')
norms = np.linalg.norm(knn_embs, axis=1, keepdims=True)
knn_embs_n = knn_embs / np.maximum(norms, 1e-8)

def embed_pkl(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    kps = raw['keypoints'][:,:,:2].astype('float32')
    tensor = torch.tensor(kps, dtype=torch.float32).permute(2,0,1)
    out = transforms({'frames': tensor})
    inp = out['frames'].unsqueeze(0)
    with torch.no_grad():
        emb = enc(inp).squeeze(0).numpy().astype('float32')
    return emb

def knn_predict(emb, k=5):
    emb_n = emb / max(np.linalg.norm(emb), 1e-8)
    sims = knn_embs_n @ emb_n
    top5_idx = np.argsort(sims)[::-1][:k]
    from collections import defaultdict
    votes = defaultdict(float)
    for idx in top5_idx:
        votes[knn_labels[idx]] += max(sims[idx], 0)
    winner = max(votes, key=votes.get)
    total = sum(votes.values()) or 1
    return id_to_gloss.get(winner, '?'), votes[winner]/total, top5_idx, sims

# Test with ACTUAL test-split files (ones in test_include.csv)
test_cases = [
    ('3. happy', 'data/include_poses/INCLUDE/Pose_Signs/Adjectives/3. happy/MVI_9542.pkl'),
    ('4. sad', 'data/include_poses/INCLUDE/Pose_Signs/Adjectives/4. sad/MVI_9565.pkl'),
    ('94. good', 'data/include_poses/INCLUDE/Pose_Signs/Adjectives/94. good/MVI_9268.pkl'),
    ('95. bad', 'data/include_poses/INCLUDE/Pose_Signs/Adjectives/95. bad/MVI_9278.pkl'),
    ('1. loud', 'data/include_poses/INCLUDE/Pose_Signs/Adjectives/1. loud/MVI_9289.pkl'),
]
print('Testing KNN on actual test-set pkl files:')
correct = 0
for label, path in test_cases:
    if not os.path.exists(path):
        print(f'  SKIP (not found): {path}')
        continue
    emb = embed_pkl(path)
    pred, conf, top_idx, sims = knn_predict(emb)
    match = label.split('. ')[-1].lower() in pred.lower() or pred.lower() in label.lower()
    correct += match
    sym = 'OK' if match else 'FAIL'
    pred_clean = pred.split('. ')[-1].upper()
    print(f'  [{sym}] GT={label} | Pred={pred_clean} | conf={conf:.3f}')
    # Show top5 neighbours
    for ri in top_idx[:3]:
        g = id_to_gloss.get(knn_labels[ri], '?')
        print(f'       nbr: [{knn_labels[ri]}] {g} sim={sims[ri]:.4f}')

print(f'\nAccuracy: {correct}/{len(test_cases)}')
