"""
Resumable downloader and installer for PyTorch CUDA 12.4 in .venv_gpu.
Directly downloads the wheels to E:/wheels/ with range resume,
installs dependencies, and verifies E1 forward pass on GPU.
"""

import os
import sys
import time
import urllib.request
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WHEELS_DIR = Path("E:/wheels")
WHEELS_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = Path("E:/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
PIP_CACHE = Path("E:/pip_cache")
PIP_CACHE.mkdir(parents=True, exist_ok=True)

VENV_GPU_PYTHON = PROJECT_ROOT / ".venv_gpu" / "Scripts" / "python.exe"

env = os.environ.copy()
env["TEMP"] = str(TMP_DIR)
env["TMP"] = str(TMP_DIR)
env["PIP_CACHE_DIR"] = str(PIP_CACHE)

WHEEL_URLS = [
    ("https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl", "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl"),
    ("https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl", "torchvision-0.21.0+cu124-cp312-cp312-win_amd64.whl"),
]

def download_resumable(url, local_path, max_retries=20):
    print(f"\nTarget: {local_path.name}")
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    for attempt in range(1, max_retries + 1):
        existing_size = local_path.stat().st_size if local_path.exists() else 0
        if existing_size >= total_size and total_size > 0:
            print(f"Already fully downloaded: {local_path.name} ({existing_size:,} bytes)")
            return

        print(f"[Attempt {attempt}/{max_retries}] Downloading from byte {existing_size:,}/{total_size:,} ({existing_size/total_size*100:.1f}%) ...")
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response, open(local_path, "ab" if existing_size > 0 else "wb") as f:
                last_print = time.time()
                downloaded = existing_size
                while True:
                    chunk = response.read(1024 * 1024)  # 1 MB chunk
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_print > 5.0:
                        speed = (downloaded - existing_size) / (time.time() - last_print) / (1024 * 1024)
                        print(f"  {downloaded:,}/{total_size:,} bytes ({downloaded/total_size*100:.1f}%) - {speed:.2f} MB/s")
                        last_print = time.time()
                        existing_size = downloaded

            final_size = local_path.stat().st_size
            if final_size >= total_size:
                print(f"Successfully downloaded {local_path.name} ({final_size:,} bytes)")
                return
        except Exception as e:
            print(f"  Network error ({e}), retrying in 3 seconds...")
            time.sleep(3)

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")

def main():
    print("=== Step 1: Resumable Download of PyTorch & Torchvision Wheels ===")
    downloaded_wheels = []
    for url, filename in WHEEL_URLS:
        target_path = WHEELS_DIR / filename
        download_resumable(url, target_path)
        downloaded_wheels.append(str(target_path))

    print("\n=== Step 2: Install PyTorch & Torchvision into .venv_gpu ===")
    cmd_install_torch = [
        str(VENV_GPU_PYTHON), "-m", "pip",
        "--cache-dir", str(PIP_CACHE),
        "install", "--no-deps"
    ] + downloaded_wheels
    print(f"Running: {' '.join(cmd_install_torch)}")
    subprocess.run(cmd_install_torch, env=env, check=True)

    print("\n=== Step 3: Install Remaining Project Requirements ===")
    # Dependencies required by torch, torchvision, ultralytics
    cmd_install_reqs = [
        str(VENV_GPU_PYTHON), "-m", "pip",
        "--cache-dir", str(PIP_CACHE),
        "install",
        "filelock", "typing_extensions", "networkx", "jinja2", "fsspec", "sympy", "mpmath",
        "numpy", "pillow",
        "ultralytics",
        "thop",
        "albumentations",
        "scikit-learn",
        "opencv-python",
        "pandas",
        "matplotlib",
        "seaborn",
        "PyYAML"
    ]
    print(f"Running requirements install...")
    subprocess.run(cmd_install_reqs, env=env, check=True)

    print("\n=== Step 4: Verification of GPU & E1 Forward Pass ===")
    verify_script = """
import os
import sys
from pathlib import Path
import torch

print(f"torch.__version__: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA is NOT available!"

device = torch.device("cuda:0")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Run a one-batch forward pass of E1 on the GPU
from ultralytics import YOLO
model = YOLO("configs/yolov8n_nc3.yaml")
model.to(device)
model.model.eval()

# Measure initial memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

x = torch.randn(1, 3, 640, 640, device=device)
with torch.no_grad():
    y = model.model(x)

peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
reserved_vram_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

print(f"Forward pass completed successfully!")
print(f"Output shape/type: {type(y)} len={len(y) if isinstance(y, (list, tuple)) else 'N/A'}")
print(f"Peak VRAM allocated: {peak_vram_mb:.2f} MB")
print(f"Peak VRAM reserved:  {reserved_vram_mb:.2f} MB")
"""
    verify_file = PROJECT_ROOT / "verify_gpu_e1.py"
    verify_file.write_text(verify_script, encoding="utf-8")
    subprocess.run([str(VENV_GPU_PYTHON), str(verify_file)], env=env, check=True)

if __name__ == "__main__":
    main()
