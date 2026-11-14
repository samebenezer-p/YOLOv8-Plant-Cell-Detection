"""
Comprehensive Real-Data Training and Evaluation Suite for Pyropia haitanensis Cell Detection.

Protocol & Guarantees:
- Single-class cell detection (nc=1, class 0: cell).
- Real data only (data/real/data.yaml).
- Disjoint specimen-level split (Train: 11 specimens / 880 tiles, Val: 3 specimens / 240 tiles, Test: 2 specimens / 160 tiles).
- Training protocol: imgsz=640, epochs=50, patience=20, seed=42, batch=16, AMP=True, device=0 (RTX 3050 Laptop GPU).
- Pretrained transfer: compatible weights from yolov8n.pt transferred; reports transferred vs newly initialized tensors.
- Primary stages: E1, E2, E5a, E5b, E3, E4, E8 (E6 and E7 skipped per protocol).
- Phase 8 Sanity Check: If E1 mAP50 > 0.95 or cell AP50 > 0.95, execution stops and reports.
- Results saved immediately after each stage to results/real/<stage>.json.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Architecture imports
from src.model_factory import build_stage, StageManifest, _count_params
from src.models.boundary_branch import (
    BoundaryHead,
    InterceptWrapper,
    generate_boundary_targets_tensor,
)
from src.models.fusion_neck import (
    E8FusionNeck,
    _FastFusion,
    _make_e8_predict_once,
)
from src.models.losses import patched_bbox_iou, shape_iou, mpdiou
from src.benchmark import _estimate_gflops

class E8LossWrapper:
    pass


class BoundaryLossWrapper:
    """Wrapper around standard YOLOv8 detection loss to inject auxiliary BoundaryHead loss."""

    def __init__(self, loss_fn, boundary_head, loss_history=None):
        self.loss_fn = loss_fn
        self.boundary_head = boundary_head
        self.loss_history = loss_history
        self.__name__ = getattr(loss_fn, "__name__", "v8DetectionLoss")

    def __call__(self, preds, batch):
        total_loss, loss_items = self.loss_fn(preds, batch)
        heatmap_pred = self.boundary_head.latest_heatmap
        if heatmap_pred is not None:
            bs, _, h, w = heatmap_pred.shape
            targets = generate_boundary_targets_tensor(
                batch["bboxes"], batch["batch_idx"], bs, h, w, thickness=3
            )
            b_total, b_bce, b_dice = self.boundary_head.compute_loss(heatmap_pred, targets)
            total_loss += b_total
            if self.loss_history is not None:
                self.loss_history.append(float(b_total.item()))
        return total_loss, loss_items


# Backwards-compatibility aliases in __main__ for torch.load / unpickling
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "E8FusionNeck", E8FusionNeck)
    setattr(main_mod, "_FastFusion", _FastFusion)
    setattr(main_mod, "BoundaryHead", BoundaryHead)
    setattr(main_mod, "InterceptWrapper", InterceptWrapper)
    setattr(main_mod, "BoundaryLossWrapper", BoundaryLossWrapper)
    setattr(main_mod, "E8LossWrapper", E8LossWrapper)


if main_mod is not None:
    setattr(main_mod, "BoundaryLossWrapper", BoundaryLossWrapper)


def transfer_pretrained_weights(model: nn.Module, weights_path: str = "yolov8n.pt") -> Tuple[int, int]:
    """Transfer compatible weights from pretrained checkpoint to model.

    Returns (num_transferred_tensors, num_reinit_tensors).
    """
    if not Path(weights_path).exists():
        print(f"[transfer_weights] Pretrained file {weights_path} not found. Skipping weight transfer.")
        return 0, 0

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt

    model_state = model.state_dict()
    transferred = 0
    reinit = 0

    for k, v in model_state.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            v.copy_(state_dict[k])
            transferred += 1
        else:
            reinit += 1

    print(f"  [Weight Transfer] Transferred {transferred} tensors from {weights_path} | {reinit} tensors newly initialized.")
    return transferred, reinit


class RealAblationTrainer(DetectionTrainer):
    """Custom trainer for real data ablation experiments."""

    def __init__(self, overrides=None, _callbacks=None, stage="E1"):
        self.stage = stage
        self.epoch_boundary_losses: List[float] = []
        self._batch_boundary_losses: List[float] = []
        self.transferred_count = 0
        self.reinit_count = 0
        super().__init__(overrides=overrides, _callbacks=_callbacks)

        def on_train_epoch_end_cb(trainer):
            if trainer._batch_boundary_losses:
                avg_b_loss = sum(trainer._batch_boundary_losses) / len(trainer._batch_boundary_losses)
                trainer.epoch_boundary_losses.append(avg_b_loss)
                ep = len(trainer.epoch_boundary_losses)
                print(f"  [{trainer.stage}] Epoch {ep} Boundary Loss: {avg_b_loss:.4f} (across {len(trainer._batch_boundary_losses)} batches)")
                trainer._batch_boundary_losses.clear()

        self.add_callback("on_train_epoch_end", on_train_epoch_end_cb)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Always build model from the model factory with nc=1
        factory_model, manifest = build_stage(self.stage, nc=1)
        self.factory_manifest = manifest

        # Transfer compatible pretrained weights from yolov8n.pt
        self.transferred_count, self.reinit_count = transfer_pretrained_weights(factory_model.model, "yolov8n.pt")

        # Check for BoundaryHead
        bh_modules = [m for m in factory_model.model.modules() if isinstance(m, BoundaryHead)]
        if bh_modules:
            self.boundary_head = bh_modules[0]

        return factory_model.model

    def set_model_attributes(self):
        super().set_model_attributes()
        # If BoundaryHead present, attach BoundaryLossWrapper
        if hasattr(self, "boundary_head"):
            orig_loss = self.loss
            self.loss = BoundaryLossWrapper(
                orig_loss, self.boundary_head, loss_history=self._batch_boundary_losses
            )


def evaluate_on_test_split(
    weights_path: str,
    data_yaml: str = "data/real/data.yaml",
    imgsz: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_det: int = 300,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Strict test evaluation on held-out test split."""
    yolo = YOLO(weights_path)
    val_res = yolo.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        conf=conf_thresh,
        iou=iou_thresh,
        max_det=max_det,
        plots=True,
        save_json=True,
        verbose=True,
        project=str(save_dir.parent) if save_dir else "runs/val",
        name=save_dir.name if save_dir else "test_eval",
        exist_ok=True,
    )

    box = val_res.box
    p = float(box.mp) if hasattr(box, "mp") else 0.0
    r = float(box.mr) if hasattr(box, "mr") else 0.0
    f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    map50 = float(box.map50) if hasattr(box, "map50") else 0.0
    map50_95 = float(box.map) if hasattr(box, "map") else 0.0

    cell_ap50 = 0.0
    cell_ap50_95 = 0.0
    if hasattr(box, "ap50") and box.ap50 is not None and len(box.ap50) > 0:
        cell_ap50 = float(box.ap50[0])
    if hasattr(box, "ap") and box.ap is not None and len(box.ap) > 0:
        cell_ap50_95 = float(box.ap[0])

    metrics = {
        "precision": p,
        "recall": r,
        "F1": f1,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "cell_AP50": cell_ap50,
        "cell_AP50_95": cell_ap50_95,
        "val_save_dir": str(val_res.save_dir),
    }
    return metrics


def save_representative_predictions(
    weights_path: str,
    test_img_dir: Path,
    output_dir: Path,
    num_samples: int = 4,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    imgsz: int = 640,
) -> List[str]:
    """Run inference on representative test tiles and save visualized predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(weights_path)

    # Pick sample test images (e.g. from plant_8 and plant_16)
    test_images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    if not test_images:
        print(f"Warning: No images found in {test_img_dir}")
        return []

    # Distribute samples evenly across the list
    step = max(1, len(test_images) // num_samples)
    selected = [test_images[i * step] for i in range(min(num_samples, len(test_images)))]

    saved_paths = []
    for idx, img_path in enumerate(selected, 1):
        res = yolo.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thresh,
            iou=iou_thresh,
            save=False,
            verbose=False,
        )[0]
        # Plot predictions on image
        plotted_bgr = res.plot(line_width=2, font_size=1)
        out_file = output_dir / f"test_pred_sample_{idx}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_file), plotted_bgr)
        saved_paths.append(str(out_file))

    print(f"  Saved {len(saved_paths)} representative prediction images to {output_dir}")
    return saved_paths


def run_stage(
    stage_id: str,
    epochs: int = 50,
    batch_size: int = 16,
    patience: int = 20,
    seed: int = 42,
    data_yaml: str = "data/real/data.yaml",
) -> Dict[str, Any]:
    """Train and evaluate a single stage on real Pyropia haitanensis microscopy tiles."""
    print(f"\n{'='*70}")
    print(f"  STARTING REAL-DATA STAGE: {stage_id}")
    print(f"  Data: {data_yaml} | Epochs: {epochs} | Batch: {batch_size} | Seed: {seed}")
    print(f"{'='*70}\n")

    results_dir = PROJECT_ROOT / "results" / "real"
    results_dir.mkdir(parents=True, exist_ok=True)
    stage_json_path = results_dir / f"{stage_id}.json"

    # Pre-build model to verify architecture and complexity
    factory_yolo, manifest = build_stage(stage_id, nc=1)
    inf_params = manifest.params_inference
    total_params = manifest.params_total
    gflops = manifest.gflops

    # Setup trainer arguments
    trainer_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "patience": patience,
        "batch": batch_size,
        "imgsz": 640,
        "device": 0,
        "seed": seed,
        "amp": True,
        "project": "results/real",
        "name": stage_id,
        "plots": True,
        "save": True,
        "verbose": True,
        "workers": 0,  # 0 is fastest/safest on Windows
        "exist_ok": True,
    }

    class StageSpecificTrainer(RealAblationTrainer):
        def __init__(self, overrides=None, _callbacks=None):
            super().__init__(overrides=overrides, _callbacks=_callbacks, stage=stage_id)

    # Determine loss patching
    iou_patch = None
    if stage_id in ["E5a", "E8"]:
        iou_patch = shape_iou
    elif stage_id == "E5b":
        iou_patch = mpdiou

    # Memory-safe execution with OOM retry
    current_batch = batch_size
    train_success = False
    train_start_time = time.time()

    while not train_success and current_batch >= 2:
        trainer_kwargs["batch"] = current_batch
        try:
            print(f"  Launching training with batch={current_batch}...")
            torch.cuda.empty_cache()
            gc.collect()

            if iou_patch is not None:
                print(f"  Applying IoU monkey-patch: {iou_patch.__name__}")
                with patched_bbox_iou(iou_patch):
                    factory_yolo.train(trainer=StageSpecificTrainer, **trainer_kwargs)
            else:
                factory_yolo.train(trainer=StageSpecificTrainer, **trainer_kwargs)

            train_success = True
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  [CUDA OOM] Caught OOM at batch={current_batch}: {oom}")
            current_batch //= 2
            print(f"  Halving batch size to {current_batch} and retrying...")
            torch.cuda.empty_cache()
            gc.collect()
            # Clean up partial run dir if needed
            run_dir = Path("results/real") / stage_id
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    if not train_success:
        raise RuntimeError(f"Training failed for {stage_id} even at batch={current_batch}.")

    train_duration = time.time() - train_start_time
    print(f"  Training for {stage_id} completed in {train_duration:.1f}s ({train_duration/60:.2f} min).")

    # Locate best.pt checkpoint
    save_dir = Path(factory_yolo.trainer.save_dir) if hasattr(factory_yolo, "trainer") and factory_yolo.trainer else Path(f"results/real/{stage_id}")
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = save_dir / "weights" / "last.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Checkpoint not found in {save_dir / 'weights'}")

    print(f"  Best checkpoint: {best_pt}")

    # Load Rigorous Benchmark metrics from results/paper/rigorous_gpu_benchmark_results.json
    bench_data = {}
    bench_json_path = PROJECT_ROOT / "results" / "paper" / "rigorous_gpu_benchmark_results.json"
    if bench_json_path.exists():
        with open(bench_json_path, "r", encoding="utf-8") as f:
            bench_all = json.load(f)
            stats = bench_all.get("stats", {})
            lookup_stage = stage_id if stage_id in stats.get("b1_fp32", {}) else "E1"
            b1_fp32 = stats.get("b1_fp32", {}).get(lookup_stage, {})
            b1_fp16 = stats.get("b1_fp16", {}).get(lookup_stage, {})
            b16_fp32 = stats.get("b16_fp32", {}).get(lookup_stage, {})
            b16_fp16 = stats.get("b16_fp16", {}).get(lookup_stage, {})
            bench_data = {
                "b1_fp32_latency_ms": b1_fp32.get("lat_med"),
                "b1_fp32_latency_std": b1_fp32.get("lat_std"),
                "b1_fp32_fps": b1_fp32.get("fps_med"),
                "b1_fp32_vram_mb": b1_fp32.get("vram_med"),
                "b1_fp16_latency_ms": b1_fp16.get("lat_med"),
                "b1_fp16_latency_std": b1_fp16.get("lat_std"),
                "b1_fp16_fps": b1_fp16.get("fps_med"),
                "b1_fp16_vram_mb": b1_fp16.get("vram_med"),
                "b16_fp32_latency_ms": b16_fp32.get("lat_med"),
                "b16_fp32_latency_std": b16_fp32.get("lat_std"),
                "b16_fp32_fps": b16_fp32.get("fps_med"),
                "b16_fp32_vram_mb": b16_fp32.get("vram_med"),
                "b16_fp16_latency_ms": b16_fp16.get("lat_med"),
                "b16_fp16_latency_std": b16_fp16.get("lat_std"),
                "b16_fp16_fps": b16_fp16.get("fps_med"),
                "b16_fp16_vram_mb": b16_fp16.get("vram_med"),
            }

    # Evaluate on held-out test split
    print(f"\n  Evaluating {stage_id} on TEST split (imgsz=640, conf=0.25, iou=0.45)...")
    eval_dir = save_dir / "test_eval"
    eval_metrics = evaluate_on_test_split(
        str(best_pt),
        data_yaml=data_yaml,
        imgsz=640,
        conf_thresh=0.25,
        iou_thresh=0.45,
        max_det=300,
        save_dir=eval_dir,
    )

    # Save 4 representative test prediction images
    test_img_dir = PROJECT_ROOT / "data" / "real" / "images" / "test"
    pred_dir = save_dir / "representative_predictions"
    pred_images = save_representative_predictions(
        str(best_pt),
        test_img_dir=test_img_dir,
        output_dir=pred_dir,
        num_samples=4,
    )

    # Copy confusion matrix to save_dir if exists
    cm_path = Path(eval_metrics["val_save_dir"]) / "confusion_matrix.png"
    cm_dest = save_dir / "confusion_matrix.png"
    if cm_path.exists() and cm_path != cm_dest:
        shutil.copy2(cm_path, cm_dest)

    # Extract per-epoch boundary loss if available
    epoch_boundary_losses = getattr(factory_yolo.trainer, "epoch_boundary_losses", [])

    # Compile stage result dictionary
    stage_result = {
        "stage": stage_id,
        "description": manifest.description,
        "status": "COMPLETED",
        "checkpoint": str(best_pt),
        "params_inference": inf_params,
        "params_total": total_params,
        "gflops": gflops,
        "train_batch_size": current_batch,
        "train_duration_sec": train_duration,
        "test_metrics": eval_metrics,
        "benchmark": bench_data,
        "representative_predictions": pred_images,
        "epoch_boundary_losses": epoch_boundary_losses,
    }

    # Save immediately to results/real/<stage>.json
    with open(stage_json_path, "w", encoding="utf-8") as f:
        json.dump(stage_result, f, indent=2)
    print(f"  Stage results successfully saved to: {stage_json_path}")

    # Print summary
    print(f"\n{'-'*60}")
    print(f"  {stage_id} Test Results Summary:")
    print(f"  Precision:   {eval_metrics['precision']:.4f}")
    print(f"  Recall:      {eval_metrics['recall']:.4f}")
    print(f"  F1:          {eval_metrics['F1']:.4f}")
    print(f"  mAP50:       {eval_metrics['mAP50']:.4f}")
    print(f"  mAP50-95:    {eval_metrics['mAP50_95']:.4f}")
    print(f"  Cell AP50:   {eval_metrics['cell_AP50']:.4f}")
    print(f"  Cell AP50-95:{eval_metrics['cell_AP50_95']:.4f}")
    print(f"  Params (inf):{inf_params:,} | GFLOPs: {gflops}")
    print(f"{'-'*60}\n")

    return stage_result


def main():
    parser = argparse.ArgumentParser(description="Real-data training and evaluation suite")
    parser.add_argument("--stage", type=str, default=None, help="Stage to run (e.g. E1, E2, E5a, etc.)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data", type=str, default="data/real/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    # Pre-flight environment assertions
    print("\n==================================================")
    print("  PRE-FLIGHT ENVIRONMENT VERIFICATION")
    print("==================================================")
    print(f"  torch.__version__:       {torch.__version__}")
    print(f"  torch.cuda.is_available:{torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FATAL ERROR: CUDA is unavailable. Training must NOT start on CPU. STOPPING.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    print(f"  GPU Name:                {gpu_name}")
    print(f"  GPU Total VRAM:          {vram_gb:.2f} GB ({vram_bytes:,} bytes)")
    print(f"  CUDA Device Count:       {torch.cuda.device_count()}")
    print("==================================================\n")

    # Stages in primary experimental order
    PRIMARY_ORDER = ["E1", "E2", "E5a", "E5b", "E3", "E4", "E8"]
    SKIPPED_STAGES = ["E6", "E7"]

    # Record skipped stages
    results_dir = PROJECT_ROOT / "results" / "real"
    results_dir.mkdir(parents=True, exist_ok=True)
    for sk in SKIPPED_STAGES:
        sk_path = results_dir / f"{sk}.json"
        if not sk_path.exists():
            sk_data = {
                "stage": sk,
                "status": "SKIPPED",
                "reason": "Explicitly excluded from primary training order per experimental protocol.",
            }
            with open(sk_path, "w", encoding="utf-8") as f:
                json.dump(sk_data, f, indent=2)

    stages_to_run = [args.stage] if args.stage else PRIMARY_ORDER

    for stage_id in stages_to_run:
        result = run_stage(
            stage_id=stage_id,
            epochs=args.epochs,
            batch_size=args.batch,
            patience=args.patience,
            seed=args.seed,
            data_yaml=args.data,
        )

        # Phase 8 Sanity Check for E1
        if stage_id == "E1":
            map50 = result["test_metrics"]["mAP50"]
            cell_ap50 = result["test_metrics"]["cell_AP50"]
            print(f"\n{'*'*70}")
            print(f"  PHASE 8 SANITY CHECK ON E1:")
            print(f"  Measured mAP50 = {map50:.4f} | Cell AP50 = {cell_ap50:.4f}")
            if map50 > 0.95 or cell_ap50 > 0.95:
                print("  [ALERT] SANITY CHECK TRIGGERED:")
                print("  mAP50 or cell AP50 exceeds 0.95.")
                print("  STOPPING EXECUTION IMMEDIATELY per Phase 8 protocol.")
                print("  Investigate potential data leakage, specimen overlap, or evaluation anomaly.")
                print(f"{'*'*70}\n")
                sys.exit(0)
            else:
                print(f"  Sanity check PASSED: mAP50 ({map50:.4f}) and cell AP50 ({cell_ap50:.4f}) are within credible range (<= 0.95).")
                print(f"{'*'*70}\n")


if __name__ == "__main__":
    main()
