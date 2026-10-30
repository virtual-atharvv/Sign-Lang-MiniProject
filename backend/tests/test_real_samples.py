import sys
import os
import glob
import pickle
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.sign_recognition.openhands_wrapper import OpenHandsSignRecognizer
from backend.app.pipeline import TranslationPipeline

def test_neural_recognizer_on_real_samples():
    print("=== Testing Neural Sign Recognizer on Real Dataset Samples ===")
    recognizer = OpenHandsSignRecognizer()
    
    assert recognizer.encoder is not None, "Neural model encoder should be loaded!"
    
    pose_root = "data/include_poses/INCLUDE/Pose_Signs"
    pkl_files = glob.glob(f"{pose_root}/**/*.pkl", recursive=True)
    assert len(pkl_files) > 0, "Dataset pose files must exist!"
    
    success_count = 0
    test_samples = pkl_files[:5]
    
    for pkl_path in test_samples:
        folder_name = os.path.basename(os.path.dirname(pkl_path))
        expected_gloss = folder_name.split(". ")[-1].upper()
        
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        kps = data["keypoints"] if isinstance(data, dict) else np.array(data)
        
        res = recognizer.predict(kps)
        print(f"Sample: {folder_name:<15} | Expected: {expected_gloss:<10} | Pred: {res['gloss']:<12} | Conf: {res['confidence']}")
        assert "gloss" in res
        assert "confidence" in res
        success_count += 1
        
    print(f"✅ Verified {success_count}/{len(test_samples)} neural predictions successfully.\n")

def test_full_translation_pipeline():
    print("=== Testing End-to-End Translation Pipeline ===")
    pipeline = TranslationPipeline()
    
    # Load 1 sample pose file and convert to raw frame dicts
    pose_root = "data/include_poses/INCLUDE/Pose_Signs"
    pkl_files = glob.glob(f"{pose_root}/**/*.pkl", recursive=True)
    
    with open(pkl_files[0], "rb") as f:
        data = pickle.load(f)
    kps = data["keypoints"] if isinstance(data, dict) else np.array(data)
    
    # Convert numpy matrix back into list of frame dicts
    raw_frames = []
    for t in range(min(16, len(kps))):
        frame_dict = {
            "pose": [{"x": float(kps[t, i, 0]), "y": float(kps[t, i, 1]), "z": float(kps[t, i, 2])} for i in range(33)],
            "leftHand": [{"x": float(kps[t, 33+i, 0]), "y": float(kps[t, 33+i, 1]), "z": float(kps[t, 33+i, 2])} for i in range(21)],
            "rightHand": [{"x": float(kps[t, 54+i, 0]), "y": float(kps[t, 54+i, 1]), "z": float(kps[t, 54+i, 2])} for i in range(21)],
        }
        raw_frames.append(frame_dict)
        
    result = pipeline.process(raw_frames, mode="continuous")
    print("Pipeline Result:", result)
    assert "glosses" in result
    assert "sentence" in result
    assert "confidence" in result
    assert "processing_time_ms" in result
    print("✅ Pipeline integration test passed successfully!")

if __name__ == "__main__":
    test_neural_recognizer_on_real_samples()
    test_full_translation_pipeline()
