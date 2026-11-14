import subprocess
import sys
import time
from pathlib import Path

def run_experiment(config_file):
    cmd = [sys.executable, "src/train.py", "--config", config_file]
    print(f"\n{'='*80}")
    print(f"Starting training for {config_file}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        # Run process synchronously
        subprocess.run(cmd, check=True)
        elapsed = (time.time() - start_time) / 3600
        print(f"\n[SUCCESS] {config_file} completed in {elapsed:.2f} hours.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {config_file} failed with exit code {e.returncode}.")
        raise e

def main():
    # List of configs to run for the full 100-epoch ablation study.
    # Assumes the configs are set to `epochs: 100` (which they are).
    configs = [
        "configs/e2_tiling_p2.yaml",
        "configs/e3_fusion_neck.yaml",
        "configs/e4_boundary.yaml",
        "configs/e5a_shapeiou.yaml",
        "configs/e5b_mpdiou.yaml",
        "configs/e6_context_refine.yaml",
    ]
    
    total_start = time.time()
    for config in configs:
        run_experiment(config)
        
    total_elapsed = (time.time() - total_start) / 3600
    print(f"\n{'*'*80}")
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY in {total_elapsed:.2f} hours.")
    print(f"{'*'*80}")

if __name__ == "__main__":
    main()
