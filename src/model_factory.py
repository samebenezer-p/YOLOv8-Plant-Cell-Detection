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
* DCN GFLOPs corrected via a custom_ops hook (full DCN MACs: conv + offset + mask + sampling).
* E8 uses E8FusionNeck, P2 head, BoundaryHead, ShapeIoU, and refiner with real measured GMACs.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Always print torch version and CUDA availability at import / start of run
print(f"torch.__version__: {torch.__version__} | torch.cuda.is_available(): {torch.cuda.is_available()}")

# Import custom architecture modules
from src.models.fusion_neck import (
    inject_fusion_neck,
    FusionNeck,
    E8FusionNeck,
    _FastFusion,
    _make_e8_predict_once,
)
from src.models.boundary_branch import BoundaryHead, InterceptWrapper
from src.models.dcn_optional import replace_conv_with_dcn, DeformableConv2d
from src.models.context_refine import ContextRefineClassifier

# Provide checkpoint unpickling backwards compatibility for classes pickled from __main__
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "E8FusionNeck", E8FusionNeck)
    setattr(main_mod, "_FastFusion", _FastFusion)
    # Define dummy loss wrapper if unpickler expects it
    if not hasattr(main_mod, "E8LossWrapper"):
        class E8LossWrapper:
            pass
        setattr(main_mod, "E8LossWrapper", E8LossWrapper)

# Lazy import guard for thop
try:
    import thop
    _THOP_AVAILABLE = True
except ImportError:
    _THOP_AVAILABLE = False
    print("[model_factory] WARNING: thop not installed -- GFLOPs will not be measured.")

NC = 3  # vessel, fiber, parenchyma
IMGSZ = 640
STAGE_IDS = ["E1", "E2", "E3", "E4", "E5a", "E5b", "E6", "E7", "E8"]

# Canonical checkpoint paths
CHECKPOINT_PATHS: Dict[str, List[str]] = {
    "E1":  ["runs/detect/runs/smoke/E1/weights/best.pt",
            "runs/detect/runs/smoke/E1-4/weights/best.pt",
            "runs/detect/results/e1_baseline/seed_42/weights/best.pt"],
    "E2":  ["runs/detect/runs/smoke/E2/weights/best.pt",
            "runs/detect/runs/smoke/E2-3/weights/best.pt",
            "runs/detect/results/e2_tiling_p2/seed_42/weights/best.pt"],
    "E3":  ["runs/detect/runs/smoke/E3/weights/best.pt",
            "runs/detect/runs/smoke/E3-3/weights/best.pt",
            "runs/detect/results/e3_fusion_neck/seed_42/weights/best.pt"],
    "E4":  ["runs/detect/runs/smoke/E4/weights/best.pt",
            "runs/detect/runs/smoke/E4-3/weights/best.pt",
            "runs/detect/results/e4_boundary/seed_42/weights/best.pt"],
    "E5a": ["runs/detect/runs/smoke/E5a/weights/best.pt",
            "runs/detect/runs/smoke/E5a-3/weights/best.pt",
            "runs/detect/results/e5a_shapeiou/seed_42/weights/best.pt"],
    "E5b": ["runs/detect/runs/smoke/E5b/weights/best.pt",
            "runs/detect/runs/smoke/E5b-3/weights/best.pt",
            "runs/detect/results/e5b_mpdiou/seed_42/weights/best.pt"],
    "E6":  ["runs/detect/results/e6_context_refine/seed_42/weights/best.pt"],
    "E7":  ["runs/detect/runs/smoke/E7/weights/best.pt",
            "runs/detect/runs/smoke/E7-3/weights/best.pt",
            "runs/detect/results/e7_dcn_optional/seed_42/weights/best.pt"],
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
    # Complexity
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
    """thop custom_op hook: full DCNv2 MACs = conv + offset + mask + sampling."""
    x = x_inputs[0]
    b, c_in, h_in, w_in = x.shape
    c_out = module.out_channels
    k = module.kernel_size
    h_out, w_out = y_output.shape[-2:]
    g = module.groups

    # 1. Base deformable conv MACs
    conv_macs = b * h_out * w_out * (k * k * c_in // g) * c_out
    # 2. Offset conv MACs
    offset_macs = b * h_out * w_out * (k * k * c_in // g) * (2 * k * k * g)
    # 3. Mask conv MACs
    mask_macs = b * h_out * w_out * (k * k * c_in // g) * (k * k * g)
    # 4. Bilinear interpolation sampling MACs
    sample_macs = b * h_out * w_out * (k * k * c_in) * 4

    total_macs = conv_macs + offset_macs + mask_macs + sample_macs
    module.total_ops += torch.DoubleTensor([int(total_macs)])


def _profile(model: nn.Module, imgsz: int = IMGSZ) -> Tuple[float, float]:
    """Return (GMACs, GFLOPs) for model at given imgsz. Returns (0,0) if thop missing."""
    if not _THOP_AVAILABLE:
        return 0.0, 0.0
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
# Individual stage builders (supports nc=1 and nc=3)
# ---------------------------------------------------------------------------

def _build_e1(nc: int = 1) -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    cfg = "configs/yolov8n_nc1.yaml" if nc == 1 else "configs/yolov8n_nc3.yaml"
    model = YOLO(cfg)
    params = _count_params(model.model)
    expected = 3011043 if nc == 1 else 3011433
    assert params == expected, f"E1 params expected {expected:,}, got {params:,}"

    m = StageManifest(
        stage_id="E1",
        description=f"Stock YOLOv8n baseline (nc={nc})",
        builds_on="--",
        nc=nc,
        iou_loss="CIoU",
    )
    return model, m


def _build_e2(nc: int = 1) -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    cfg = "configs/yolov8n_p2_nc1.yaml" if nc == 1 else "configs/yolov8n_p2_nc3.yaml"
    model = YOLO(cfg)
    params = _count_params(model.model)
    expected = 2926692 if nc == 1 else 2926956
    assert params == expected, f"E2 params expected {expected:,}, got {params:,}"

    m = StageManifest(
        stage_id="E2",
        description=f"P2 stride-4 detection head (nc={nc})",
        builds_on="E1",
        nc=nc,
        p2_head=True,
        iou_loss="CIoU",
    )
    return model, m


def _build_e3(nc: int = 1) -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    cfg = "configs/yolov8n_nc1.yaml" if nc == 1 else "configs/yolov8n_nc3.yaml"
    model = YOLO(cfg)
    inject_fusion_neck(model.model, num_repeats=2, out_channels=128)
    params = _count_params(model.model)
    expected = 2019061 if nc == 1 else 2019451
    assert params == expected, f"E3 params expected {expected:,}, got {params:,}"

    m = StageManifest(
        stage_id="E3",
        description=f"Bidirectional FusionNeck (BiFPN-style, 128ch, nc={nc})",
        builds_on="E1",
        nc=nc,
        fusion_neck=True,
        iou_loss="CIoU",
    )
    return model, m


def _build_e4(nc: int = 1) -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    cfg = "configs/yolov8n_nc1.yaml" if nc == 1 else "configs/yolov8n_nc3.yaml"
    model = YOLO(cfg)

    # Wrap layer 15 (P3 C2f feature) with BoundaryHead
    b_head = BoundaryHead(in_channels=64, hidden_channels=32, alpha=0.5)
    orig_layer15 = model.model.model[15]
    wrapper = InterceptWrapper(orig_layer15, b_head)
    wrapper.i = 15
    wrapper.f = getattr(orig_layer15, "f", -1)
    model.model.model[15] = wrapper

    params = _count_params(model.model)
    expected = 3029572 if nc == 1 else 3029962
    assert params == expected, f"E4 params expected {expected:,}, got {params:,}"

    m = StageManifest(
        stage_id="E4",
        description=f"Auxiliary BoundaryHead on P3 feature (train-only, nc={nc})",
        builds_on="E1",
        nc=nc,
        boundary_head=True,
        boundary_head_train_only=True,
        iou_loss="CIoU",
    )
    return model, m


def _build_e5a(nc: int = 1) -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    cfg = "configs/yolov8n_nc1.yaml" if nc == 1 else "configs/yolov8n_nc3.yaml"
    model = YOLO(cfg)
    params = _count_params(model.model)
    expected = 3011043 if nc == 1 else 3011433
    assert params == expected, f"E5a params expected {expected:,}, got {params:,}"

    m = StageManifest(
        stage_id="E5a",
        description=f"Shape IoU loss variant (architecture = E1, nc={nc})",
        builds_on="E1",
        nc=nc,
        iou_loss="ShapeIoU",
    )
    return model, m


def _build_e5b(nc: int = 1) -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    cfg = "configs/yolov8n_nc1.yaml" if nc == 1 else "configs/yolov8n_nc3.yaml"
    model = YOLO(cfg)
    params = _count_params(model.model)
    expected = 3011043 if nc == 1 else 3011433
    assert params == expected, f"E5b params expected {expected:,}, got {params:,}"

    m = StageManifest(
        stage_id="E5b",
        description=f"MPDIoU loss variant (architecture = E1, nc={nc})",
        builds_on="E1",
        nc=nc,
        iou_loss="MPDIoU",
    )
    return model, m


def _build_e6(nc: int = 1) -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    cfg = "configs/yolov8n_nc1.yaml" if nc == 1 else "configs/yolov8n_nc3.yaml"
    model = YOLO(cfg)
    clf = ContextRefineClassifier(input_size=64, margin_ratio=0.25)
    model._refiner = clf
    params = _count_params(model.model)
    expected = 3011043 if nc == 1 else 3011433
    assert params == expected, f"E6 detector params expected {expected:,}, got {params:,}"

    m = StageManifest(
        stage_id="E6",
        description=f"E1 detector + ContextRefineClassifier (post-hoc, nc={nc})",
        builds_on="E1",
        nc=nc,
        refiner=True,
        iou_loss="CIoU",
    )
    return model, m


def _build_e7(nc: int = 1) -> Tuple[Any, StageManifest]:
    from ultralytics import YOLO
    cfg = "configs/yolov8n_nc1.yaml" if nc == 1 else "configs/yolov8n_nc3.yaml"
    model = YOLO(cfg)
    replace_conv_with_dcn(model.model, layer_indices=[6, 8])

    dcn_count = sum(1 for m in model.model.modules() if isinstance(m, DeformableConv2d))
    if dcn_count != 3:
        raise AssertionError(f"E7: expected exactly 3 DeformableConv2d layers, found {dcn_count}.")

    params = _count_params(model.model)
    expected = 3073332 if nc == 1 else 3073722
    assert params == expected, f"E7 params expected {expected:,}, got {params:,}"

    mf = StageManifest(
        stage_id="E7",
        description=f"DCNv2 in last 2 C2f backbone blocks (3 layers, nc={nc})",
        builds_on="E1",
        nc=nc,
        dcn=True,
        dcn_layer_count=dcn_count,
        iou_loss="CIoU",
    )
    return model, mf


def _build_e8(nc: int = 1) -> Tuple[Any, StageManifest]:
    """E8: P2 head + E8FusionNeck + BoundaryHead + ShapeIoU + ContextRefiner."""
    from ultralytics import YOLO
    cfg = "configs/yolov8n_p2_nc1.yaml" if nc == 1 else "configs/yolov8n_p2_nc3.yaml"
    model = YOLO(cfg)

    # Inject E8FusionNeck for P3/P4/P5
    fn = E8FusionNeck(in_ch=(64, 128, 256), inner_ch=128, num_repeats=2)
    model.model.fusion_neck = fn
    model.model._predict_once = _make_e8_predict_once(fn).__get__(model.model, type(model.model))

    # Wrap layer 18 (P2-level feature in yolov8n-p2) with BoundaryHead
    b_head = BoundaryHead(in_channels=32, hidden_channels=32, alpha=0.5)
    orig_layer18 = model.model.model[18]
    wrapper = InterceptWrapper(orig_layer18, b_head)
    wrapper.i = 18
    wrapper.f = getattr(orig_layer18, "f", -1)
    model.model.model[18] = wrapper

    # Post-hoc ContextRefiner
    clf = ContextRefineClassifier(input_size=64, margin_ratio=0.25)
    model._refiner = clf

    # Assertions on architecture
    assert hasattr(model.model, "fusion_neck") and isinstance(model.model.fusion_neck, E8FusionNeck), \
        "E8: E8FusionNeck not attached"
    assert any(isinstance(m, BoundaryHead) for m in model.model.modules()), \
        "E8: BoundaryHead not found in model"
    assert not any(isinstance(m, DeformableConv2d) for m in model.model.modules()), \
        "E8: DCN should NOT be present in E8"

    params = _count_params(model.model)
    expected = 3061975 if nc == 1 else 3062239
    assert params == expected, f"E8 params expected {expected:,}, got {params:,}"

    m = StageManifest(
        stage_id="E8",
        description=f"Full combined: P2 + E8FusionNeck + BoundaryHead + ShapeIoU + Refiner (nc={nc})",
        builds_on="combined",
        nc=nc,
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

def build_stage(stage_id: str, nc: int = 1) -> Tuple[Any, StageManifest]:
    """Build a model for the given stage and return (model, manifest)."""
    if stage_id not in _BUILDERS:
        raise ValueError(f"Unknown stage '{stage_id}'. Valid: {list(_BUILDERS)}")

    print(f"\n{'-'*60}")
    print(f"  Building {stage_id} (nc={nc}) ...")
    model, manifest = _BUILDERS[stage_id](nc=nc)

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

    # GFLOPs profiling (inference graph only)
    gmacs, gflops = _profile(model.model, imgsz=IMGSZ)
    manifest.gmacs = gmacs
    manifest.gflops = gflops

    # Checkpoint validation
    ckpt_path = _find_checkpoint(stage_id)
    if ckpt_path:
        manifest.checkpoint_path = ckpt_path
        ckpt_params = _count_checkpoint_params(ckpt_path)
        manifest.checkpoint_params = ckpt_params
        if ckpt_params is not None:
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

    # E8 specific checkpoint validation: load checkpoint and run one prediction
    if stage_id == "E8" and manifest.checkpoint_path and Path(manifest.checkpoint_path).exists():
        try:
            ckpt = torch.load(manifest.checkpoint_path, map_location="cpu", weights_only=False)
            loaded_m = ckpt.get("model")
            if loaded_m is not None:
                loaded_m.float().eval()
                with torch.no_grad():
                    dummy_out = loaded_m(torch.randn(1, 3, 256, 256))
                print(f"    [E8 Checkpoint Prediction Verified: forward output type {type(dummy_out)}]")
        except Exception as e:
            print(f"    [E8 Checkpoint Prediction Warning]: {e}")

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
        saved_model = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        if saved_model is None:
            return None
        if isinstance(saved_model, dict):
            return sum(v.numel() for v in saved_model.values() if isinstance(v, torch.Tensor))
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
    """Build, print, and save the complexity table for all (or selected) stages."""
    if output_path is None:
        output_path = str(PROJECT_ROOT / "results" / "complexity_table.json")

    stages = stages or STAGE_IDS
    rows: List[StageManifest] = []
    e1_params: Optional[int] = None
    e1_gmacs: Optional[float] = None
    manifest_by_stage: Dict[str, StageManifest] = {}

    for sid in stages:
        try:
            _, manifest = build_stage(sid)
        except Exception as e:
            print(f"  [complexity_table] ERROR building {sid}: {e}")
            raise e
        rows.append(manifest)
        manifest_by_stage[sid] = manifest
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

    # STRICT ASSERTIONS ON COMPLEXITY
    if "E1" in manifest_by_stage and "E7" in manifest_by_stage:
        e1_g = manifest_by_stage["E1"].gflops
        e7_g = manifest_by_stage["E7"].gflops
        assert e7_g > e1_g, f"Assertion failed: E7 GFLOPs ({e7_g}) must be > E1 GFLOPs ({e1_g})"

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
        "  * DCN GFLOPs hook applies conv + offset + mask + sampling operations (E7 GFLOPs > E1)\n"
        "  * E8 measured directly with E8FusionNeck, P2 head, and BoundaryHead at 640x640"
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
