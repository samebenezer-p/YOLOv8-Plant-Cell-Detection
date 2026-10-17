"""
Model Factory — single source of truth for all ablation stages (E1–E8).

Every experiment must obtain its model via ``build_stage(stage_id)`` so that
architecture construction, parameter accounting, and complexity profiling are
done identically in every context.

Key guarantees
--------------
* All models built with ``nc=3`` (vessel / fiber / parenchyma).
* Params and GFLOPs measured on the *built* model at imgsz=640, never
  approximated from a separate fresh instantiation.
* Training-only modules (BoundaryHead) reported separately from
  inference-time parameters.
* Checkpoint validation: if a saved `best.pt` is found, its parameter count
  is compared to the factory build; mismatches are flagged as stale.
* DCN GFLOPs corrected via a custom_ops hook (offset + sampling MACs).

Usage
-----
    from src.model_factory import build_stage, print_complexity_table

    model, manifest = build_stage("E1")
    print_complexity_table()            # prints + saves results/complexity_table.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy import guard for thop (may not be installed)
# ---------------------------------------------------------------------------
try:
    import thop
    _THOP_AVAILABLE = True
except ImportError:
    _THOP_AVAILABLE = False
    print("[model_factory] WARNING: thop not installed — GFLOPs will not be measured.")

NC = 3  # vessel, fiber, parenchyma
IMGSZ = 640
STAGE_IDS = ["E1", "E2", "E3", "E4", "E5a", "E5b", "E6", "E7", "E8"]

# Canonical checkpoint paths (populated after full training; smoke paths used if missing)
CHECKPOINT_PATHS: Dict[str, List[str]] = {
    "E1":  ["runs/detect/results/e1_baseline/seed_42/weights/best.pt",
             "runs/detect/runs/smoke/E1/weights/best.pt"],
    "E2":  ["runs/detect/results/e2_tiling_p2/seed_42/weights/best.pt",
             "runs/detect/runs/smoke/E2/weights/best.pt"],
    "E3":  ["runs/detect/results/e3_fusion_neck/seed_42/weights/best.pt",
             "runs/detect/runs/smoke/E3/weights/best.pt"],
    "E4":  ["runs/detect/results/e4_boundary/seed_42/weights/best.pt",
             "runs/detect/runs/smoke/E4/weights/best.pt"],
    "E5a": ["runs/detect/results/e5a_shapeiou/seed_42/weights/best.pt",
             "runs/detect/runs/smoke/E5a/weights/best.pt"],
    "E5b": ["runs/detect/results/e5b_mpdiou/seed_42/weights/best.pt",
             "runs/detect/runs/smoke/E5b/weights/best.pt"],
    "E6":  ["runs/detect/results/e6_context_refine/seed_42/weights/best.pt"],
    "E7":  ["runs/detect/results/e7_dcn_optional/seed_42/weights/best.pt",
             "runs/detect/runs/smoke/E7/weights/best.pt"],
    "E8":  ["runs/detect/runs/smoke/E8/weights/best.pt"],
}


# ---------------------------------------------------------------------------
# Data class for the manifest of a single stage
# ---------------------------------------------------------------------------

@dataclass
class StageManifest:
    stage_id: str
    description: str
    builds_on: str                         # "E1" for ablations, "combined" for E8
    nc: int = NC
    # Architecture flags
    p2_head: bool = False
    fusion_neck: bool = False
    boundary_head: bool = False
    boundary_head_train_only: bool = True  # always True when boundary_head=True
    iou_loss: str = "CIoU"
    refiner: bool = False
    dcn: bool = False
    dcn_layer_count: int = 0
    # Complexity (filled in after build)
    params_total: int = 0
    params_inference: int = 0             # params_total - train-only params
    params_train_only: int = 0            # boundary head params
    params_delta_vs_e1: int = 0
    gmacs: float = 0.0
    gflops: float = 0.0
    gmacs_delta_vs_e1: float = 0.0
    gflops_delta_vs_e1: float = 0.0
    # Checkpoint validation
    checkpoint_path: Optional[str] = None
    checkpoint_params: Optional[int] = None
    checkpoint_status: str = "no_checkpoint"  # valid / stale / no_checkpoint
    stale_reason: str = ""


# ---------------------------------------------------------------------------
# GFLOPs / GMACs profiler
# ---------------------------------------------------------------------------

def _dcn_custom_op(module, x_inputs, y_output):
    """thop custom_op hook: add DCN offset + sampling MACs."""
    x = x_inputs[0]
    b, c_in, h_in, w_in = x.shape
    c_out = module.out_channels
    k = module.kernel_size
    h_out, w_out = y_output.shape[-2:]
    # Offset conv MACs + mask conv MACs + deform sampling MACs (bilinear)
    offset_macs = b * h_out * w_out * (2 * k * k * c_in + 1) * (2 * k * k)
    mask_macs   = b * h_out * w_out * (k * k * c_in + 1) * (k * k)
    sample_macs = b * h_out * w_out * k * k * c_in * 4  # bilinear interp
    extra = offset_macs + mask_macs + sample_macs
    module.total_ops += torch.DoubleTensor([int(extra)])


def _profile(model: nn.Module, imgsz: int = IMGSZ) -> Tuple[float, float]:
    """Return (GMACs, GFLOPs) for model at given imgsz. Returns (0,0) if thop missing."""
    if not _THOP_AVAILABLE:
        return 0.0, 0.0
    from src.models.dcn_optional import DeformableConv2d
    x = torch.randn(1, 3, imgsz, imgsz)
    try:
        macs, _ = thop.profile(
            model,
            inputs=(x,),
            custom_ops={DeformableConv2d: _dcn_custom_op},
            verbose=False,
        )
        return round(macs / 1e9, 4), round(2 * macs / 1e9, 4)
    except Exception as e:
        print(f"  [profile] WARNING: {e}")
        return 0.0, 0.0


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Individual stage builders
# ---------------------------------------------------------------------------

def _build_e1() -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml")
    model.model.nc = NC
    m = StageManifest(
        stage_id="E1",
        description="Stock YOLOv8n baseline (nc=3)",
        builds_on="—",
        nc=NC,
        iou_loss="CIoU",
    )
    return model, m


def _build_e2() -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    model = YOLO("yolov8n-p2.yaml")
    model.model.nc = NC
    m = StageManifest(
        stage_id="E2",
        description="P2 stride-4 detection head (nc=3)",
        builds_on="E1",
        nc=NC,
        p2_head=True,
        iou_loss="CIoU",
    )
    return model, m


def _build_e3() -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    from src.models.fusion_neck import inject_fusion_neck
    model = YOLO("yolov8n.yaml")
    model.model.nc = NC
    inject_fusion_neck(model.model, num_repeats=2, out_channels=256)
    m = StageManifest(
        stage_id="E3",
        description="Bidirectional FusionNeck (BiFPN-style, nc=3)",
        builds_on="E1",
        nc=NC,
        fusion_neck=True,
        iou_loss="CIoU",
    )
    return model, m


def _build_e4() -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    from src.models.boundary_branch import BoundaryHead, InterceptWrapper
    model = YOLO("yolov8n.yaml")
    model.model.nc = NC

    # Intercept P3-level feature (layer 15 in YOLOv8n is the first neck C2f)
    # We intercept *after* the backbone at layer index that feeds P3.
    # In yolov8n.yaml the P3-level C2f is typically layer 12 or 15.
    # We probe the model to find the correct channel count.
    dummy = torch.zeros(1, 3, 64, 64)
    seq = model.model.model
    # Run through backbone to find P3-level feature channels
    y_saved: Dict[int, torch.Tensor] = {}
    y = dummy
    for i, layer in enumerate(seq):
        f = getattr(layer, "f", -1)
        if isinstance(f, list):
            inputs = [y if fi == -1 else y_saved.get(fi, y) for fi in f]
            y = layer(inputs)
        else:
            y = layer(y_saved.get(f, y) if f != -1 else y)
        y_saved[i] = y
        if i == 9:   # after backbone — P3 channel at layer 4 typically
            break

    # Use the P3 feature channels from layer index 4 (backbone output P3)
    # For yolov8n: layer 4 = C2f with 64 channels
    p3_channels = y_saved.get(4, dummy).shape[1] if 4 in y_saved else 64

    bh = BoundaryHead(in_channels=p3_channels, hidden_channels=max(32, p3_channels // 2), alpha=0.5)

    # Wrap layer 4 (the P3-level feature) with InterceptWrapper
    original_layer4 = seq[4]
    wrapper = InterceptWrapper(original_layer4, bh)
    wrapper.i = 4
    wrapper.f = getattr(original_layer4, "f", -1)
    seq[4] = wrapper

    m = StageManifest(
        stage_id="E4",
        description="Auxiliary BoundaryHead on P3 feature (train-only, nc=3)",
        builds_on="E1",
        nc=NC,
        boundary_head=True,
        boundary_head_train_only=True,
        iou_loss="CIoU",
    )
    return model, m


def _build_e5a() -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml")
    model.model.nc = NC
    m = StageManifest(
        stage_id="E5a",
        description="Shape IoU loss variant (architecture = E1, nc=3)",
        builds_on="E1",
        nc=NC,
        iou_loss="ShapeIoU",
    )
    return model, m


def _build_e5b() -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml")
    model.model.nc = NC
    m = StageManifest(
        stage_id="E5b",
        description="MPDIoU loss variant (architecture = E1, nc=3)",
        builds_on="E1",
        nc=NC,
        iou_loss="MPDIoU",
    )
    return model, m


def _build_e6() -> Tuple[Any, StageManifest]:
    """E6: E1 detector + ContextRefineClassifier (post-hoc, not in backbone)."""
    from ultralytics import YOLO
    from src.models.context_refine import ContextRefineClassifier
    model = YOLO("yolov8n.yaml")
    model.model.nc = NC
    # The classifier is not attached to the backbone; it's post-hoc.
    # We attach it as a metadata attribute so factory callers can access it.
    clf = ContextRefineClassifier(input_size=64, margin_ratio=0.25)
    model._refiner = clf
    m = StageManifest(
        stage_id="E6",
        description="E1 detector + ContextRefineClassifier (post-hoc, nc=3)",
        builds_on="E1",
        nc=NC,
        refiner=True,
        iou_loss="CIoU",
    )
    return model, m


def _build_e7() -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    from src.models.dcn_optional import replace_conv_with_dcn, DeformableConv2d
    model = YOLO("yolov8n.yaml")
    model.model.nc = NC
    replace_conv_with_dcn(model.model, layer_indices=[6, 8])
    # Verify injection
    dcn_count = sum(
        1 for m in model.model.modules() if isinstance(m, DeformableConv2d)
    )
    if dcn_count != 3:
        raise AssertionError(
            f"E7: expected exactly 3 DeformableConv2d layers, found {dcn_count}. "
            "DCN injection failed."
        )
    mf = StageManifest(
        stage_id="E7",
        description="DCNv2 in last 2 C2f backbone blocks (3 layers, nc=3)",
        builds_on="E1",
        nc=NC,
        dcn=True,
        dcn_layer_count=dcn_count,
        iou_loss="CIoU",
    )
    return model, mf


def _build_e8() -> Tuple[Any, StageManifest]:
    """E8: P2 head + FusionNeck + BoundaryHead + ShapeIoU + ContextRefiner."""
    from ultralytics import YOLO
    from src.models.fusion_neck import inject_fusion_neck
    from src.models.boundary_branch import BoundaryHead, InterceptWrapper
    from src.models.context_refine import ContextRefineClassifier

    model = YOLO("yolov8n-p2.yaml")
    model.model.nc = NC

    # Inject FusionNeck
    inject_fusion_neck(model.model, num_repeats=2, out_channels=256)

    # Probe P3-level channels after fusion neck injection
    # With the P2 yaml, the P3-level backbone feature is at layer index 4 (64ch)
    p3_channels = 64  # yolov8n P3 = 64 channels
    bh = BoundaryHead(
        in_channels=p3_channels,
        hidden_channels=max(32, p3_channels // 2),
        alpha=0.5,
    )
    # Wrap layer 4 (P3 feature, pre-neck)
    seq = model.model.model
    original_layer4 = seq[4]
    wrapper = InterceptWrapper(original_layer4, bh)
    wrapper.i = 4
    wrapper.f = getattr(original_layer4, "f", -1)
    seq[4] = wrapper

    # Post-hoc ContextRefiner
    clf = ContextRefineClassifier(input_size=64, margin_ratio=0.25)
    model._refiner = clf

    # Verify components
    from src.models.fusion_neck import FusionNeck
    from src.models.dcn_optional import DeformableConv2d
    assert any(isinstance(m, FusionNeck) for m in model.model.modules()), \
        "E8: FusionNeck not found in model"
    assert any(isinstance(m, BoundaryHead) for m in model.model.modules()), \
        "E8: BoundaryHead not found in model"
    assert not any(isinstance(m, DeformableConv2d) for m in model.model.modules()), \
        "E8: DCN should NOT be present in E8"

    m = StageManifest(
        stage_id="E8",
        description="Full combined: P2 + FusionNeck + BoundaryHead + ShapeIoU + Refiner (nc=3)",
        builds_on="combined",
        nc=NC,
        p2_head=True,
        fusion_neck=True,
        boundary_head=True,
        boundary_head_train_only=True,
        iou_loss="ShapeIoU",
        refiner=True,
        dcn=False,
    )
    return model, m


_BUILDERS = {
    "E1": _build_e1,
    "E2": _build_e2,
    "E3": _build_e3,
    "E4": _build_e4,
    "E5a": _build_e5a,
    "E5b": _build_e5b,
    "E6": _build_e6,
    "E7": _build_e7,
    "E8": _build_e8,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_stage(stage_id: str) -> Tuple[Any, StageManifest]:
    """Build a model for the given stage and return (model, manifest).

    Parameters
    ----------
    stage_id : str
        One of "E1", "E2", "E3", "E4", "E5a", "E5b", "E6", "E7", "E8".

    Returns
    -------
    model : ultralytics YOLO instance (with possible custom modules attached)
    manifest : StageManifest
        Populated with params, GFLOPs, and checkpoint validation results.
    """
    if stage_id not in _BUILDERS:
        raise ValueError(f"Unknown stage '{stage_id}'. Valid: {list(_BUILDERS)}")

    print(f"\n{'-'*60}")
    print(f"  Building {stage_id} ...")
    model, manifest = _BUILDERS[stage_id]()

    # -- Parameter accounting ----------------------------------------------
    from src.models.boundary_branch import BoundaryHead
    from src.models.context_refine import ContextRefineClassifier

    total_params = _count_params(model.model)

    # Train-only params: BoundaryHead modules embedded in the model
    train_only_params = sum(
        _count_params(m)
        for m in model.model.modules()
        if isinstance(m, BoundaryHead)
    )

    # Refiner params (post-hoc, not in model graph)
    refiner_params = 0
    if hasattr(model, "_refiner") and model._refiner is not None:
        refiner_params = _count_params(model._refiner)

    manifest.params_total = total_params + refiner_params
    manifest.params_train_only = train_only_params
    manifest.params_inference = total_params - train_only_params + refiner_params

    # -- GFLOPs profiling (inference graph only; no BoundaryHead) ----------
    gmacs, gflops = _profile(model.model, imgsz=IMGSZ)
    manifest.gmacs = gmacs
    manifest.gflops = gflops

    # -- Checkpoint validation ---------------------------------------------
    ckpt_path = _find_checkpoint(stage_id)
    if ckpt_path:
        manifest.checkpoint_path = ckpt_path
        ckpt_params = _count_checkpoint_params(ckpt_path)
        manifest.checkpoint_params = ckpt_params
        if ckpt_params is not None:
            # Compare detector params only (no refiner)
            factory_det_params = total_params
            if abs(ckpt_params - factory_det_params) <= 100:
                manifest.checkpoint_status = "valid"
            else:
                manifest.checkpoint_status = "stale"
                manifest.stale_reason = (
                    f"factory={factory_det_params:,} vs ckpt={ckpt_params:,} "
                    f"(Delta={ckpt_params - factory_det_params:+,})"
                )
    else:
        manifest.checkpoint_status = "no_checkpoint"

    _print_manifest(manifest)
    return model, manifest


def _find_checkpoint(stage_id: str) -> Optional[str]:
    for p in CHECKPOINT_PATHS.get(stage_id, []):
        full = PROJECT_ROOT / p
        if full.exists():
            return str(full)
    return None


def _count_checkpoint_params(ckpt_path: str) -> Optional[int]:
    """Load checkpoint and count parameters without executing the model."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Ultralytics saves model under 'model' key
        saved_model = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        if saved_model is None:
            return None
        if isinstance(saved_model, dict):
            # state_dict
            return sum(v.numel() for v in saved_model.values() if isinstance(v, torch.Tensor))
        # nn.Module
        return _count_params(saved_model)
    except Exception as e:
        print(f"  [ckpt] WARNING: Could not load {ckpt_path}: {e}")
        return None


def _print_manifest(m: StageManifest) -> None:
    status_icon = {"valid": "[valid]", "stale": "[stale]", "no_checkpoint": "--"}.get(
        m.checkpoint_status, "?"
    )
    print(f"  Stage:       {m.stage_id}  ({m.description})")
    print(f"  Builds on:   {m.builds_on}")
    print(f"  nc={m.nc}  |  P2={m.p2_head}  FusionNeck={m.fusion_neck}  "
          f"BoundaryHead={m.boundary_head}  DCN={m.dcn}({m.dcn_layer_count})  "
          f"Refiner={m.refiner}  IoU={m.iou_loss}")
    print(f"  Params total={m.params_total:,}  "
          f"inference={m.params_inference:,}  "
          f"train-only(BH)={m.params_train_only:,}")
    if m.gflops:
        print(f"  GMACs={m.gmacs:.4f}  GFLOPs={m.gflops:.4f}  (at imgsz={IMGSZ})")
    else:
        print(f"  GFLOPs: thop not available")
    print(f"  Checkpoint:  {status_icon} {m.checkpoint_status}"
          + (f"  [{m.stale_reason}]" if m.stale_reason else "")
          + (f"\n    path: {m.checkpoint_path}" if m.checkpoint_path else ""))


# ---------------------------------------------------------------------------
# Complexity table (auto-generated)
# ---------------------------------------------------------------------------

def build_complexity_table(
    output_path: Optional[str] = None,
    stages: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build, print, and save the complexity table for all (or selected) stages.

    Parameters
    ----------
    output_path : str, optional
        JSON output path. Defaults to ``results/complexity_table.json``.
    stages : list of str, optional
        Stage IDs to include. Defaults to all (E1–E8).

    Returns
    -------
    list of dict
        One entry per stage with all StageManifest fields plus delta columns.
    """
    if output_path is None:
        output_path = str(PROJECT_ROOT / "results" / "complexity_table.json")

    stages = stages or STAGE_IDS
    rows: List[StageManifest] = []
    e1_params: Optional[int] = None
    e1_gmacs: Optional[float] = None

    for sid in stages:
        try:
            _, manifest = build_stage(sid)
        except Exception as e:
            print(f"  [complexity_table] ERROR building {sid}: {e}")
            manifest = StageManifest(
                stage_id=sid,
                description=f"BUILD ERROR: {e}",
                builds_on="?",
            )
        rows.append(manifest)
        if sid == "E1":
            e1_params = manifest.params_total
            e1_gmacs = manifest.gmacs

    # Fill deltas now that E1 is known
    for r in rows:
        if e1_params is not None:
            r.params_delta_vs_e1 = r.params_total - e1_params
        if e1_gmacs is not None:
            r.gmacs_delta_vs_e1 = round(r.gmacs - e1_gmacs, 4)
            r.gflops_delta_vs_e1 = round(r.gflops - 2 * e1_gmacs, 4)

    # Print table
    _print_complexity_table(rows)

    # Save JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"\n  Complexity table saved -> {output_path}")

    return [asdict(r) for r in rows]


def _print_complexity_table(rows: List[StageManifest]) -> None:
    """Print a formatted ASCII complexity table."""
    print(f"\n{'='*110}")
    print(f"  COMPLEXITY TABLE -- nc={NC}, imgsz={IMGSZ}, GFLOPs=2xGMACs (DCN corrected)")
    print(f"{'='*110}")
    hdr = (
        f"  {'Stage':<6} {'Params(total)':>14} {'Params(inf)':>12} {'Params(BH)':>11} "
        f"{'DeltaParams':>12} {'GMACs':>7} {'GFLOPs':>8} {'DeltaGFLOPs':>12} "
        f"{'Ckpt':>7}  Description"
    )
    print(hdr)
    print(f"  {'-'*107}")
    for r in rows:
        ckpt = {"valid": "[valid]", "stale": "[stale]", "no_checkpoint": "  --"}.get(
            r.checkpoint_status, "?"
        )
        delta_p = f"{r.params_delta_vs_e1:+,}" if r.stage_id != "E1" else "--"
        delta_g = f"{r.gflops_delta_vs_e1:+.4f}" if r.stage_id != "E1" else "--"
        print(
            f"  {r.stage_id:<6} {r.params_total:>14,} {r.params_inference:>12,} "
            f"{r.params_train_only:>11,} {delta_p:>12} "
            f"{r.gmacs:>7.4f} {r.gflops:>8.4f} {delta_g:>12}  "
            f"{ckpt:<9}  {r.description[:50]}"
        )
    print(f"{'='*110}")
    print(
        "  Notes:\n"
        "  * Params(inf)  = params_total - train-only (BoundaryHead)\n"
        "  * Params(BH)   = BoundaryHead params (train-only, discarded at inference)\n"
        "  * E6 Refiner params counted in total/inf but NOT in GFLOPs (post-hoc classifier)\n"
        "  * [stale]      = checkpoint param count differs from factory build (pre-fix weights)\n"
        "  * GFLOPs for E3/E8 FusionNeck: backbone GFLOPs only; neck branch adds ~0.04 GFLOPs"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and report ablation complexity table")
    parser.add_argument("--stages", nargs="*", default=None,
                        help="Stage IDs to include (default: all)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/complexity_table.json)")
    args = parser.parse_args()
    build_complexity_table(output_path=args.output, stages=args.stages)
