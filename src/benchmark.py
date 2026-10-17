"""
Benchmark module for measuring computational complexity.

Measures parameters, GFLOPs, inference latency, FPS, and model file size.
Reports boundary-branch cost as train-time only, and tiling/context-
refinement cost as inference-time additions with separate line items.

Usage:
    python src/benchmark.py --model results/e1_baseline/seed_42/weights/best.pt \
        --imgsz 512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def benchmark_model(
    model_path: str,
    imgsz: int = 512,
    device: Optional[str] = None,
    warmup: int = 10,
    iterations: int = 100,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Benchmark a YOLOv8 model for computational complexity.

    Parameters
    ----------
    model_path : str
        Path to the model weights (.pt file).
    imgsz : int
        Input image size (square).
    device : str, optional
        Torch device. Auto-detects if None.
    warmup : int
        Number of warmup inference passes (not timed).
    iterations : int
        Number of timed inference passes.
    batch_size : int
        Batch size for inference timing.

    Returns
    -------
    dict
        Benchmark results with keys:
        - ``params``: total parameter count
        - ``params_M``: params in millions
        - ``gflops``: estimated GFLOPs
        - ``latency_mean_ms``: mean inference latency per image (ms)
        - ``latency_std_ms``: std of latency
        - ``fps``: frames per second
        - ``model_size_mb``: model file size in MB
        - ``device``: device used
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results: Dict[str, Any] = {
        "model_path": model_path,
        "imgsz": imgsz,
        "device": device,
        "batch_size": batch_size,
    }

    # Model file size
    if os.path.isfile(model_path):
        size_bytes = os.path.getsize(model_path)
        results["model_size_mb"] = round(size_bytes / (1024 * 1024), 2)
    else:
        results["model_size_mb"] = None

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)

        # Parameter count
        params = sum(p.numel() for p in model.model.parameters())
        results["params"] = params
        results["params_M"] = round(params / 1e6, 2)

        # GFLOPs estimation
        results["gflops"] = _estimate_gflops(model, imgsz, device)

        # Latency measurement
        latency_stats = _measure_latency(
            model, imgsz, device, warmup, iterations, batch_size
        )
        results.update(latency_stats)

    except Exception as e:
        print(f"WARNING: Benchmark error: {e}")
        results["error"] = str(e)

    return results


def _estimate_gflops(
    model: Any,
    imgsz: int,
    device: str,
) -> Optional[float]:
    """Estimate GFLOPs using the Ultralytics profiler or thop.

    Parameters
    ----------
    model : YOLO
        Ultralytics YOLO model.
    imgsz : int
        Input image size.
    device : str
        Torch device.

    Returns
    -------
    float or None
        Estimated GFLOPs, or None if estimation fails.
    """
    try:
        # Try thop first
        from thop import profile

        dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
        torch_model = model.model.to(device)
        torch_model.eval()

        flops, _ = profile(torch_model, inputs=(dummy,), verbose=False)
        return round(flops / 1e9, 2)
    except Exception:
        pass

    try:
        # Fallback: use Ultralytics info
        info = model.info(verbose=False)
        if isinstance(info, tuple) and len(info) >= 2:
            return round(info[1], 2)
    except Exception:
        pass

    return None


def _measure_latency(
    model: Any,
    imgsz: int,
    device: str,
    warmup: int,
    iterations: int,
    batch_size: int,
) -> Dict[str, float]:
    """Measure inference latency.

    Parameters
    ----------
    model : YOLO
        Ultralytics YOLO model.
    imgsz : int
        Input image size.
    device : str
        Torch device.
    warmup : int
        Warmup iterations.
    iterations : int
        Timed iterations.
    batch_size : int
        Batch size.

    Returns
    -------
    dict
        ``{"latency_mean_ms", "latency_std_ms", "fps"}``.
    """
    # Create dummy input
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

    # Timed runs
    latencies: list = []
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "latency_mean_ms": round(mean_ms, 2),
        "latency_std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
    }


def benchmark_overhead(
    category: str,
    description: str,
    latency_ms: float,
    is_train_only: bool = False,
) -> Dict[str, Any]:
    """Create a structured overhead line item for the benchmark report.

    Parameters
    ----------
    category : str
        E.g., "boundary_branch", "tiling", "context_refinement".
    description : str
        Human-readable description.
    latency_ms : float
        Overhead in milliseconds.
    is_train_only : bool
        Whether this cost applies only at training time.

    Returns
    -------
    dict
        Overhead line item.
    """
    return {
        "category": category,
        "description": description,
        "latency_ms": latency_ms,
        "is_train_only": is_train_only,
        "is_inference_addition": not is_train_only,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a YOLOv8 model (params, GFLOPs, latency, FPS)"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model .pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: alongside model)")
    args = parser.parse_args()

    results = benchmark_model(
        model_path=args.model,
        imgsz=args.imgsz,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    # Print summary
    print(f"\n{'='*50}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k:>20s}: {v}")

    # Save
    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.model).parent.parent / "benchmark.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nBenchmark saved to {output_path}")


if __name__ == "__main__":
    main()
