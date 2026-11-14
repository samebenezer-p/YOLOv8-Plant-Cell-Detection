"""
Install CUDA-enabled PyTorch and project dependencies into .venv_gpu,
ensuring all temp files and caches are directed to Drive E (avoiding Drive C exhaustion).
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_GPU_PYTHON = PROJECT_ROOT / ".venv_gpu" / "Scripts" / "python.exe"

# Redirect temp dirs and pip cache to Drive E
TMP_DIR = Path("E:/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
PIP_CACHE = Path("E:/pip_cache")
PIP_CACHE.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["TEMP"] = str(TMP_DIR)
env["TMP"] = str(TMP_DIR)
env["PIP_CACHE_DIR"] = str(PIP_CACHE)

def run_pip(args):
    cmd = [
        str(VENV_GPU_PYTHON), "-m", "pip",
        "--cache-dir", str(PIP_CACHE),
    ] + args
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

def main():
    print(f"Target python: {VENV_GPU_PYTHON}")
    print(f"TEMP dir: {TMP_DIR}")
    print(f"PIP cache: {PIP_CACHE}")

    # 1. Upgrade pip
    run_pip(["install", "--upgrade", "pip"])

    # 2. Install torch & torchvision with CUDA 12.4
    run_pip([
        "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ])

    # 3. Install ultralytics and project requirements
    run_pip([
        "install",
        "ultralytics",
        "thop",
        "albumentations",
        "scikit-learn",
        "opencv-python",
        "pandas",
        "matplotlib",
        "seaborn",
        "PyYAML"
    ])

    print("\n--- Installation Complete! Verifying environment ---")
    verify_script = (
        "import torch; "
        "print('torch.__version__:', torch.__version__); "
        "print('torch.cuda.is_available():', torch.cuda.is_available()); "
        "print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    )
    subprocess.run([str(VENV_GPU_PYTHON), "-c", verify_script], env=env, check=True)

if __name__ == "__main__":
    main()
