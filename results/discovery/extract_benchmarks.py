"""
extract_benchmarks.py
=====================
Extracts latency, FPS, and VRAM benchmark measurements across all models
from results/real/<stage>.json.
"""

import json
from pathlib import Path

stages = ['E1', 'E2', 'E3', 'E4', 'E5a', 'E5b', 'E8']
res_dir = Path("results/real")

print("=" * 110)
print(f"{'Stage':<5} | {'Params (Inf)':<12} | {'GFLOPs':<8} | {'B1 FP32 (ms/FPS/MB)':<23} | {'B1 FP16 (ms/FPS/MB)':<23} | {'B16 FP16 (ms/FPS/MB)':<23}")
print("=" * 110)

all_data = {}
for s in stages:
    f = res_dir / f"{s}.json"
    if not f.exists():
        print(f"{s:<5} | NOT FOUND")
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    b = d.get("benchmark", {})
    inf_params = d.get("params_inference", 0)
    gflops = d.get("gflops", 0)

    b1_fp32 = f"{b.get('b1_fp32_latency_ms', 0):.2f}ms / {b.get('b1_fp32_fps', 0):.1f} / {b.get('b1_fp32_vram_mb', 0):.1f}MB"
    b1_fp16 = f"{b.get('b1_fp16_latency_ms', 0):.2f}ms / {b.get('b1_fp16_fps', 0):.1f} / {b.get('b1_fp16_vram_mb', 0):.1f}MB"
    b16_fp32 = f"{b.get('b16_fp32_latency_ms', 0):.2f}ms / {b.get('b16_fp32_fps', 0):.1f} / {b.get('b16_fp32_vram_mb', 0):.1f}MB"
    b16_fp16 = f"{b.get('b16_fp16_latency_ms', 0):.2f}ms / {b.get('b16_fp16_fps', 0):.1f} / {b.get('b16_fp16_vram_mb', 0):.1f}MB"

    all_data[s] = {
        "params_inference": inf_params,
        "gflops": gflops,
        "b1_fp32": b1_fp32,
        "b1_fp16": b1_fp16,
        "b16_fp32": b16_fp32,
        "b16_fp16": b16_fp16,
        "raw_benchmark": b,
    }
    print(f"{s:<5} | {inf_params:<12,} | {gflops:<8.2f} | {b1_fp32:<23} | {b1_fp16:<23} | {b16_fp16:<23}")

print("=" * 110)

# Check for ONNX / TensorRT exports
onnx_files = list(Path(".").glob("**/*.onnx"))
trt_files = list(Path(".").glob("**/*.engine")) + list(Path(".").glob("**/*.trt"))
print(f"\nONNX Models Found: {len(onnx_files)} -> {onnx_files}")
print(f"TensorRT Engines Found: {len(trt_files)} -> {trt_files}")
