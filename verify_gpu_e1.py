
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
