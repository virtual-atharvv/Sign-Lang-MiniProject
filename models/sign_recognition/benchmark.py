import os
import time
import argparse
import omegaconf

def benchmark_checkpoint(config_path: str, checkpoint_path: str):
    print(f"=== Benchmarking Checkpoint: {checkpoint_path} ===")
    start_time = time.time()
    
    if not os.path.exists(config_path):
        print(f"Config path {config_path} does not exist.")
        return
        
    cfg = omegaconf.OmegaConf.load(config_path)
    cfg.pretrained = checkpoint_path
    
    try:
        from openhands.apis.inference import InferenceModel
        model = InferenceModel(cfg=cfg)
        model.init_from_checkpoint_if_available()
        
        if hasattr(cfg.data, 'test_pipeline') and cfg.data.test_pipeline.dataset.inference_mode:
            results = model.test_inference()
        else:
            results = model.compute_test_accuracy()
            
        elapsed = time.time() - start_time
        print(f"Benchmark finished in {elapsed:.2f}s.")
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error during benchmark: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenHands Checkpoints")
    parser.add_argument("--config", default="models/sign_recognition/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default="data/checkpoints/include_bert.ckpt", help="Path to checkpoint .ckpt")
    args = parser.parse_args()
    
    benchmark_checkpoint(args.config, args.checkpoint)
