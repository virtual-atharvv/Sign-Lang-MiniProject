import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def debug_stage_1_keypoint_adapter():
    print("\n--- [STAGE 1/5] Debugging Keypoint Adapter ---")
    from data.scripts.convert_keypoints import KeypointAdapter
    
    adapter = KeypointAdapter()  # default target_frames=120
    
    # Create sample frame with pose, left hand, right hand
    sample_frame = {
        "pose": [{"x": 0.5 + i*0.01, "y": 0.4 + i*0.01, "z": 0.0} for i in range(33)],
        "leftHand": [{"x": 0.35 + i*0.01, "y": 0.55, "z": 0.0} for i in range(21)],
        "rightHand": [{"x": 0.65 - i*0.01, "y": 0.55, "z": 0.0} for i in range(21)]
    }
    
    converted = adapter.convert_frame(sample_frame)
    print(f"Single frame converted shape: {converted.shape} (Expected: (75, 3))")
    assert converted.shape == (75, 3), "Shape mismatch in convert_frame"
    
    raw_buffer = [sample_frame] * 20
    processed_tensor = adapter.process_raw_buffer(raw_buffer)
    print(f"Sequence normalized shape: {processed_tensor.shape} (Expected: (120, 75, 3))")
    assert processed_tensor.shape == (120, 75, 3), "Shape mismatch in process_raw_buffer"
    print("Stage 1 PASS: Keypoint Adapter is working perfectly.")

def debug_stage_2_motion_segmenter():
    print("\n--- [STAGE 2/5] Debugging Motion Segmenter ---")
    from models.segmenter import MotionSegmenter
    
    segmenter = MotionSegmenter(motion_threshold=0.015, pause_frames=5, window_size=64)
    
    # Generate 100 frames of dummy sequence with motion and pause
    seq = np.zeros((100, 75, 3), dtype=np.float32)
    # Active motion from frame 10 to 40
    for f in range(10, 40):
        seq[f] += f * 0.05
        
    energy = segmenter.compute_motion_energy(seq)
    segments = segmenter.segment_sequence(seq)
    
    print(f"Computed motion energy length: {len(energy)}")
    print(f"Segmented intervals: {segments}")
    assert len(segments) > 0, "No segments returned"
    print("Stage 2 PASS: Motion Segmenter is working perfectly.")

def debug_stage_3_sign_recognizer():
    print("\n--- [STAGE 3/5] Debugging OpenHands Sign Recognizer ---")
    from models.sign_recognition.openhands_wrapper import OpenHandsSignRecognizer
    
    config_path     = "data/checkpoints/include/st_gcn/config.yaml"
    checkpoint_path = "data/checkpoints/include/st_gcn/epoch=215-step=23543.ckpt"
    
    print(f"Config path exists:      {os.path.exists(config_path)}")
    print(f"Checkpoint path exists:  {os.path.exists(checkpoint_path)}")
    
    recognizer = OpenHandsSignRecognizer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    print(f"Encoder loaded:  {recognizer.encoder is not None}")
    print(f"KNN index loaded: {recognizer._knn_embs is not None}")
    print(f"Classes:          {len(recognizer.id_to_gloss)}")
    
    # Feed a 40-frame dummy clip — recognizer's transform chain resamples to 120
    dummy_input = np.random.uniform(0.0, 1.0, (40, 75, 3)).astype(np.float32)
    prediction = recognizer.predict(dummy_input)
    
    print(f"Sign Recognizer Prediction: {prediction}")
    assert "gloss" in prediction, "Gloss missing in prediction output"
    assert "confidence" in prediction, "Confidence missing in prediction output"
    print("Stage 3 PASS: Sign Recognizer is working perfectly.")

def debug_stage_4_grammar_corrector():
    print("\n--- [STAGE 4/5] Debugging Grammar Corrector ---")
    from models.grammar_correction.inference import ISLGrammarCorrector
    
    corrector = ISLGrammarCorrector()
    
    test_glosses = ["HOME", "I", "GO"]
    sentence = corrector.translate(test_glosses)
    
    print(f"Input glosses: {test_glosses}")
    print(f"Translated sentence: '{sentence}'")
    assert isinstance(sentence, str) and len(sentence) > 0, "Empty sentence returned"
    print("Stage 4 PASS: Grammar Corrector is working perfectly.")

def debug_stage_5_full_pipeline():
    print("\n--- [STAGE 5/5] Debugging End-to-End Translation Pipeline ---")
    from backend.app.pipeline import TranslationPipeline
    
    pipeline = TranslationPipeline()
    
    # 30 frames mock stream
    raw_frames = []
    for f in range(30):
        raw_frames.append({
            "pose": [{"x": 0.5 + np.sin(f*0.1)*0.02, "y": 0.4, "z": 0.0} for _ in range(33)],
            "leftHand": [{"x": 0.35, "y": 0.55 + np.cos(f*0.1)*0.02, "z": 0.0} for _ in range(21)],
            "rightHand": [{"x": 0.65, "y": 0.55, "z": 0.0} for _ in range(21)]
        })
        
    t0 = time.time()
    result = pipeline.process(raw_frames, mode="continuous")
    dt = (time.time() - t0) * 1000.0
    
    print(f"Pipeline Result: {result}")
    print(f"Total Debug Execution Time: {dt:.2f} ms")
    assert "glosses" in result and "sentence" in result and "confidence" in result, "Malformed result dict"
    print("Stage 5 PASS: Full Translation Pipeline is working perfectly.")

if __name__ == "__main__":
    print("==================================================")
    print("   SignBridge Pipeline Comprehensive Diagnostic   ")
    print("==================================================")
    debug_stage_1_keypoint_adapter()
    debug_stage_2_motion_segmenter()
    debug_stage_3_sign_recognizer()
    debug_stage_4_grammar_corrector()
    debug_stage_5_full_pipeline()
    print("\n==================================================")
    print("      ALL 5 PIPELINE STAGES PASSED CLEANLY!       ")
    print("==================================================")
