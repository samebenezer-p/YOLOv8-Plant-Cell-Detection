"""
Rigorous GPU Latency & Throughput Benchmark on NVIDIA GeForce RTX 3050 Laptop GPU.
- Uses CUDA events for microsecond-precision timing.
- 100 warm-up iterations, 300 timed iterations per measurement.
- 5 repeats per model, interleaving the order (E1, E2, E3, E4, E7, E8 x 5 rounds).
- Benchmarks batch=1 and batch=16 in both FP32 and FP16.
- Measures and reports median ± std.
- Evaluates E1 vs E4 as empirical noise baseline.
- Overwrites results/paper/compute_table.* with numbers only.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_factory import build_stage, StageManifest

PAPER_DIR = PROJECT_ROOT / "results" / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

STAGES_ORDER = ["E1", "E2", "E3", "E4", "E7", "E8"]
ALL_STAGES = ["E1", "E2", "E3", "E4", "E5a", "E5b", "E6", "E7", "E8"]
NUM_REPEATS = 5
WARMUP_ITERS = 100
TIMED_ITERS = 300
IMGSZ = 640

CONDITIONS = [
    ("b1_fp32", 1, torch.float32, "Batch=1 FP32"),
    ("b1_fp16", 1, torch.float16, "Batch=1 FP16"),
    ("b16_fp32", 16, torch.float32, "Batch=16 FP32"),
    ("b16_fp16", 16, torch.float16, "Batch=16 FP16"),
]

def query_gpu_clocks():
    cmd = [
        "nvidia-smi",
        "--query-gpu=clocks.current.graphics,clocks.current.memory,clocks.current.sm,power.draw,temperature.gpu",
        "--format=csv,noheader"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Error querying clocks: {e}"

def main():
    assert torch.cuda.is_available(), "CUDA must be available!"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    print(f"===========================================================")
    print(f"  RIGOROUS GPU LATENCY & THROUGHPUT BENCHMARK")
    print(f"  Device: {gpu_name}")
    print(f"  Timing: CUDA events (100 warm-up, 300 timed passes)")
    print(f"  Repeats: 5 interleaved rounds across {STAGES_ORDER}")
    print(f"===========================================================\n")

    clocks_before = query_gpu_clocks()
    print(f"[nvidia-smi BEFORE]: {clocks_before}")

    # Build models and retrieve manifests
    print("\n--- Building models via model_factory ---")
    built_models = {}
    manifests = {}
    for st in ALL_STAGES:
        model_obj, manifest = build_stage(st)
        built_models[st] = model_obj.model
        manifests[st] = manifest

    # Structure to hold measurements:
    # raw_data[cond_key][stage_id] = { "latencies": [], "fps": [], "vram": [] }
    raw_data = {
        cond_key: {st: {"latencies": [], "fps": [], "vram": []} for st in STAGES_ORDER}
        for cond_key, _, _, _ in CONDITIONS
    }

    # Execute interleaved benchmark
    print("\n--- Starting Interleaved Benchmark Runs ---")
    start_total_bench = time.time()

    for cond_key, b_size, dtype, cond_label in CONDITIONS:
        print(f"\n=======================================================")
        print(f"  Condition: {cond_label} (batch={b_size}, dtype={dtype})")
        print(f"=======================================================")

        for round_idx in range(1, NUM_REPEATS + 1):
            print(f"\n  >> Round {round_idx}/{NUM_REPEATS} (Interleaved):")
            for st in STAGES_ORDER:
                m = built_models[st].to(device).to(dtype).eval()
                x = torch.randn(b_size, 3, IMGSZ, IMGSZ, dtype=dtype, device=device)

                # 100 warm-up passes
                with torch.no_grad():
                    for _ in range(WARMUP_ITERS):
                        _ = m(x)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                with torch.no_grad():
                    for _ in range(TIMED_ITERS):
                        _ = m(x)
                end_event.record()
                torch.cuda.synchronize(device)

                elapsed_time_ms = start_event.elapsed_time(end_event)
                latency_ms = elapsed_time_ms / float(TIMED_ITERS)
                fps = (float(TIMED_ITERS) * b_size) / (elapsed_time_ms / 1000.0)
                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

                raw_data[cond_key][st]["latencies"].append(latency_ms)
                raw_data[cond_key][st]["fps"].append(fps)
                raw_data[cond_key][st]["vram"].append(peak_vram_mb)

                print(f"    [{st}] Latency: {latency_ms:.2f} ms | FPS: {fps:.1f} | Peak VRAM: {peak_vram_mb:.1f} MB")

    total_bench_duration = time.time() - start_total_bench
    clocks_after = query_gpu_clocks()
    print(f"\nAll benchmark passes completed in {total_bench_duration:.1f} seconds.")
    print(f"[nvidia-smi AFTER]: {clocks_after}")

    # Compute summary statistics: median and std
    stats = {}
    for cond_key, _, _, _ in CONDITIONS:
        stats[cond_key] = {}
        for st in STAGES_ORDER:
            lats = raw_data[cond_key][st]["latencies"]
            fpss = raw_data[cond_key][st]["fps"]
            vrams = raw_data[cond_key][st]["vram"]

            stats[cond_key][st] = {
                "lat_med": float(np.median(lats)),
                "lat_std": float(np.std(lats)),
                "fps_med": float(np.median(fpss)),
                "fps_std": float(np.std(fpss)),
                "vram_med": float(np.median(vrams)),
            }

    # Evaluate E1 vs E4 noise baseline
    print("\n=======================================================")
    print("  NOISE BASELINE EVALUATION: E1 vs E4")
    print("  (Identical inference architecture stock YOLOv8n)")
    print("=======================================================")
    noise_report = {}
    for cond_key, _, _, cond_label in CONDITIONS:
        e1_lat = stats[cond_key]["E1"]["lat_med"]
        e4_lat = stats[cond_key]["E4"]["lat_med"]
        diff_ms = abs(e1_lat - e4_lat)
        diff_pct = (diff_ms / e1_lat) * 100.0 if e1_lat > 0 else 0.0

        e1_fps = stats[cond_key]["E1"]["fps_med"]
        e4_fps = stats[cond_key]["E4"]["fps_med"]
        diff_fps = abs(e1_fps - e4_fps)

        noise_report[cond_key] = {
            "e1_lat": e1_lat,
            "e4_lat": e4_lat,
            "diff_ms": diff_ms,
            "diff_pct": diff_pct,
            "e1_fps": e1_fps,
            "e4_fps": e4_fps,
            "diff_fps": diff_fps,
        }
        print(f"  {cond_label:18s} | E1: {e1_lat:.2f} ms vs E4: {e4_lat:.2f} ms | Delta: {diff_ms:.2f} ms ({diff_pct:.2f}%) | FPS Delta: {diff_fps:.1f}")

    # Generate Output Deliverables
    print("\n--- Writing results/paper/compute_table.* deliverables ---")

    # 1. compute_table.csv
    csv_path = PAPER_DIR / "compute_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Stage,Inference Params,GFLOPs (640),"
                "B1 FP32 Latency (ms),B1 FP32 FPS,B1 FP32 VRAM (MB),"
                "B1 FP16 Latency (ms),B1 FP16 FPS,B1 FP16 VRAM (MB),"
                "B16 FP32 Latency (ms),B16 FP32 FPS,B16 FP32 VRAM (MB),"
                "B16 FP16 Latency (ms),B16 FP16 FPS,B16 FP16 VRAM (MB)\n")
        
        for st in STAGES_ORDER:
            inf_p = manifests[st].params_inference
            gfl = manifests[st].gflops
            b1_32 = stats["b1_fp32"][st]
            b1_16 = stats["b1_fp16"][st]
            b16_32 = stats["b16_fp32"][st]
            b16_16 = stats["b16_fp16"][st]

            f.write(f"{st},{inf_p},{gfl},"
                    f"\"{b1_32['lat_med']:.2f} ± {b1_32['lat_std']:.2f}\",\"{b1_32['fps_med']:.1f} ± {b1_32['fps_std']:.1f}\",{b1_32['vram_med']:.1f},"
                    f"\"{b1_16['lat_med']:.2f} ± {b1_16['lat_std']:.2f}\",\"{b1_16['fps_med']:.1f} ± {b1_16['fps_std']:.1f}\",{b1_16['vram_med']:.1f},"
                    f"\"{b16_32['lat_med']:.2f} ± {b16_32['lat_std']:.2f}\",\"{b16_32['fps_med']:.1f} ± {b16_32['fps_std']:.1f}\",{b16_32['vram_med']:.1f},"
                    f"\"{b16_16['lat_med']:.2f} ± {b16_16['lat_std']:.2f}\",\"{b16_16['fps_med']:.1f} ± {b16_16['fps_std']:.1f}\",{b16_16['vram_med']:.1f}\n")

        # E5a, E5b, E6 identical architecture lines
        e1_b1_32 = stats["b1_fp32"]["E1"]
        e1_b1_16 = stats["b1_fp16"]["E1"]
        e1_b16_32 = stats["b16_fp32"]["E1"]
        e1_b16_16 = stats["b16_fp16"]["E1"]

        for st in ["E5a", "E5b", "E6"]:
            inf_p = manifests[st].params_inference
            gfl = manifests[st].gflops
            f.write(f"{st},{inf_p},{gfl},"
                    f"\"{e1_b1_32['lat_med']:.2f} ± {e1_b1_32['lat_std']:.2f}*\",\"{e1_b1_32['fps_med']:.1f} ± {e1_b1_32['fps_std']:.1f}*\",{e1_b1_32['vram_med']:.1f},"
                    f"\"{e1_b1_16['lat_med']:.2f} ± {e1_b1_16['lat_std']:.2f}*\",\"{e1_b1_16['fps_med']:.1f} ± {e1_b1_16['fps_std']:.1f}*\",{e1_b1_16['vram_med']:.1f},"
                    f"\"{e1_b16_32['lat_med']:.2f} ± {e1_b16_32['lat_std']:.2f}*\",\"{e1_b16_32['fps_med']:.1f} ± {e1_b16_32['fps_std']:.1f}*\",{e1_b16_32['vram_med']:.1f},"
                    f"\"{e1_b16_16['lat_med']:.2f} ± {e1_b16_16['lat_std']:.2f}*\",\"{e1_b16_16['fps_med']:.1f} ± {e1_b16_16['fps_std']:.1f}*\",{e1_b16_16['vram_med']:.1f}\n")

    # 2. compute_table.md
    md_path = PAPER_DIR / "compute_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GPU Inference Latency & Throughput Benchmark\n\n")
        f.write(f"- **Device**: {gpu_name}\n")
        f.write(f"- **Resolution**: 640 x 640\n")
        f.write(f"- **Timing Method**: CUDA Events (`torch.cuda.Event(enable_timing=True)`)\n")
        f.write(f"- **Benchmark Protocol**: 100 warm-up runs, 300 timed passes, 5 interleaved repeats per model\n")
        f.write(f"- **Reported Metric**: Median ± Standard Deviation across 5 interleaved repeats\n")
        f.write(f"- **NMS**: Excluded (raw model forward pass timing only)\n")
        f.write(f"- **Train-Only Parameters**: Excluded from inference parameter counts (BoundaryHead discarded at inference)\n\n")

        f.write(f"### Clocks & Thermal State\n\n")
        f.write(f"- **Before**: `{clocks_before}`\n")
        f.write(f"- **After**: `{clocks_after}`\n\n")

        f.write(f"### Empirical Noise Baseline (E1 vs E4, identical inference architecture)\n\n")
        f.write(f"| Condition | E1 Latency (ms) | E4 Latency (ms) | Absolute Delta (ms) | Relative Delta (%) | FPS Delta |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for cond_key, _, _, cond_label in CONDITIONS:
            nr = noise_report[cond_key]
            f.write(f"| **{cond_label}** | {nr['e1_lat']:.2f} | {nr['e4_lat']:.2f} | {nr['diff_ms']:.2f} | {nr['diff_pct']:.2f}% | {nr['diff_fps']:.1f} |\n")

        f.write(f"\n### Batch = 1 Benchmark\n\n")
        f.write(f"| Stage | Inf Params | GFLOPs | FP32 Latency (ms) | FP32 FPS | FP32 VRAM (MB) | FP16 Latency (ms) | FP16 FPS | FP16 VRAM (MB) |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for st in STAGES_ORDER:
            inf_p = manifests[st].params_inference
            gfl = manifests[st].gflops
            b32 = stats["b1_fp32"][st]
            b16 = stats["b1_fp16"][st]
            f.write(f"| **{st}** | {inf_p:,} | {gfl} | {b32['lat_med']:.2f} ± {b32['lat_std']:.2f} | {b32['fps_med']:.1f} ± {b32['fps_std']:.1f} | {b32['vram_med']:.1f} | {b16['lat_med']:.2f} ± {b16['lat_std']:.2f} | {b16['fps_med']:.1f} ± {b16['fps_std']:.1f} | {b16['vram_med']:.1f} |\n")
        for st in ["E5a", "E5b", "E6"]:
            inf_p = manifests[st].params_inference
            gfl = manifests[st].gflops
            b32 = stats["b1_fp32"]["E1"]
            b16 = stats["b1_fp16"]["E1"]
            f.write(f"| **{st}** | {inf_p:,} | {gfl} | {b32['lat_med']:.2f} ± {b32['lat_std']:.2f}* | {b32['fps_med']:.1f} ± {b32['fps_std']:.1f}* | {b32['vram_med']:.1f} | {b16['lat_med']:.2f} ± {b16['lat_std']:.2f}* | {b16['fps_med']:.1f} ± {b16['fps_std']:.1f}* | {b16['vram_med']:.1f} |\n")

        f.write(f"\n### Batch = 16 Benchmark (GPU Compute-Bound)\n\n")
        f.write(f"| Stage | Inf Params | GFLOPs | FP32 Latency (ms) | FP32 FPS | FP32 VRAM (MB) | FP16 Latency (ms) | FP16 FPS | FP16 VRAM (MB) |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for st in STAGES_ORDER:
            inf_p = manifests[st].params_inference
            gfl = manifests[st].gflops
            b32 = stats["b16_fp32"][st]
            b16 = stats["b16_fp16"][st]
            f.write(f"| **{st}** | {inf_p:,} | {gfl} | {b32['lat_med']:.2f} ± {b32['lat_std']:.2f} | {b32['fps_med']:.1f} ± {b32['fps_std']:.1f} | {b32['vram_med']:.1f} | {b16['lat_med']:.2f} ± {b16['lat_std']:.2f} | {b16['fps_med']:.1f} ± {b16['fps_std']:.1f} | {b16['vram_med']:.1f} |\n")
        for st in ["E5a", "E5b", "E6"]:
            inf_p = manifests[st].params_inference
            gfl = manifests[st].gflops
            b32 = stats["b16_fp32"]["E1"]
            b16 = stats["b16_fp16"]["E1"]
            f.write(f"| **{st}** | {inf_p:,} | {gfl} | {b32['lat_med']:.2f} ± {b32['lat_std']:.2f}* | {b32['fps_med']:.1f} ± {b32['fps_std']:.1f}* | {b32['vram_med']:.1f} | {b16['lat_med']:.2f} ± {b16['lat_std']:.2f}* | {b16['fps_med']:.1f} ± {b16['fps_std']:.1f}* | {b16['vram_med']:.1f} |\n")

        f.write(f"\n\\* E5a/E5b share identical architecture with E1; E6 detector is identical to E1 with post-hoc refinement.\n")

    # 3. compute_table.tex
    tex_path = PAPER_DIR / "compute_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% GPU Inference Latency and Throughput Table\n")
        f.write(f"% Device: {gpu_name}\n")
        f.write("% Timing: CUDA events, 100 warm-up, 300 timed passes, 5 interleaved repeats (median +- std)\n")
        f.write("% NMS excluded, train-only parameters excluded\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Inference latency (ms per batch) and throughput (FPS) on {gpu_name} across Batch=1 and Batch=16 ($640\\times640$, NMS excluded). Values represent median $\\pm$ std over 5 interleaved repeats.}}\n")
        f.write("\\label{tab:compute_complexity_rigorous}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{l c c c c c c c c}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Stage} & \\textbf{Params} & \\textbf{GFLOPs} & \\multicolumn{2}{c}{\\textbf{Batch=1 FP32}} & \\multicolumn{2}{c}{\\textbf{Batch=1 FP16}} & \\multicolumn{2}{c}{\\textbf{Batch=16 FP16}} \\\\\n")
        f.write(" & \\textbf{(Inf)} & & \\textbf{Latency (ms)} & \\textbf{FPS} & \\textbf{Latency (ms)} & \\textbf{FPS} & \\textbf{Latency (ms)} & \\textbf{FPS} \\\\\n")
        f.write("\\hline\n")
        for st in STAGES_ORDER:
            inf_p = manifests[st].params_inference
            gfl = manifests[st].gflops
            b1_32 = stats["b1_fp32"][st]
            b1_16 = stats["b1_fp16"][st]
            b16_16 = stats["b16_fp16"][st]
            f.write(f"{st} & {inf_p:,} & {gfl:.2f} & {b1_32['lat_med']:.2f} $\\pm$ {b1_32['lat_std']:.2f} & {b1_32['fps_med']:.1f} & {b1_16['lat_med']:.2f} $\\pm$ {b1_16['lat_std']:.2f} & {b1_16['fps_med']:.1f} & {b16_16['lat_med']:.2f} $\\pm$ {b16_16['lat_std']:.2f} & {b16_16['fps_med']:.1f} \\\\\n")
        f.write(f"E5a & {manifests['E5a'].params_inference:,} & {manifests['E5a'].gflops:.2f} & {e1_b1_32['lat_med']:.2f}$^*$ & {e1_b1_32['fps_med']:.1f}$^*$ & {e1_b1_16['lat_med']:.2f}$^*$ & {e1_b1_16['fps_med']:.1f}$^*$ & {e1_b16_16['lat_med']:.2f}$^*$ & {e1_b16_16['fps_med']:.1f}$^*$ \\\\\n")
        f.write(f"E5b & {manifests['E5b'].params_inference:,} & {manifests['E5b'].gflops:.2f} & {e1_b1_32['lat_med']:.2f}$^*$ & {e1_b1_32['fps_med']:.1f}$^*$ & {e1_b1_16['lat_med']:.2f}$^*$ & {e1_b1_16['fps_med']:.1f}$^*$ & {e1_b16_16['lat_med']:.2f}$^*$ & {e1_b16_16['fps_med']:.1f}$^*$ \\\\\n")
        f.write(f"E6 & {manifests['E6'].params_inference:,} & {manifests['E6'].gflops:.2f} & {e1_b1_32['lat_med']:.2f}$^*$ & {e1_b1_32['fps_med']:.1f}$^*$ & {e1_b1_16['lat_med']:.2f}$^*$ & {e1_b1_16['fps_med']:.1f}$^*$ & {e1_b16_16['lat_med']:.2f}$^*$ & {e1_b16_16['fps_med']:.1f}$^*$ \\\\\n")
        f.write("\\hline\n")
        f.write("\\multicolumn{9}{l}{\\footnotesize $^*$E5a and E5b share identical architecture with E1; E6 detector is identical to E1 with post-hoc refinement.} \\\\\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    # Save complete JSON raw results for reference
    raw_results_path = PAPER_DIR / "rigorous_gpu_benchmark_results.json"
    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": gpu_name,
            "clocks_before": clocks_before,
            "clocks_after": clocks_after,
            "total_duration_sec": total_bench_duration,
            "stats": stats,
            "noise_baseline_e1_vs_e4": noise_report,
            "raw_data": raw_data
        }, f, indent=2)

    print("Rigorous benchmark completed and all paper files overwritten successfully!")

if __name__ == "__main__":
    main()
