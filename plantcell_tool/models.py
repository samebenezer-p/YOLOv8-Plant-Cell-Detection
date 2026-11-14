"""
Model loading and inspection utilities for plant-cell detection.

Supports standard experimental checkpoints (E1..E8) and arbitrary .pt checkpoints.
Handles unpickling patch requirements for custom modules (BoundaryHead, InterceptWrapper)
and dynamically re-binds _predict_once for E8 BiFPN feature routing.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from ultralytics import YOLO

DOCUMENTED_THRESHOLDS: Dict[str, float] = {
    "E1": 0.108,
    "E3": 0.114,
    "E5a": 0.063,
}

MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "E1": {
        "description": "Baseline YOLOv8n (CIoU loss)",
        "params": 3_005_843,
        "gflops": 8.19,
        "map50": 0.4846,
        "best_f1_conf": 0.108,
        "recommended": False,
        "note": "Baseline architecture without custom neck or auxiliary boundary loss.",
    },
    "E2": {
        "description": "YOLOv8n + Shape-IoU (Ablation)",
        "params": 3_005_843,
        "gflops": 8.19,
        "map50": 0.5283,
        "best_f1_conf": None,
        "recommended": False,
        "note": "Ablation study variant using Shape-IoU bounding box regression.",
    },
    "E3": {
        "description": "P2BiFPN / FusionNeck (Fastest / Smallest)",
        "params": 2_015_781,
        "gflops": 7.28,
        "map50": 0.6210,
        "best_f1_conf": 0.114,
        "recommended": False,
        "note": "Lightweight BiFPN fusion neck; lowest parameter count and highest throughput.",
    },
    "E4": {
        "description": "YOLOv8n + Boundary Head (Ablation)",
        "params": 3_036_723,
        "gflops": 8.24,
        "map50": 0.5055,
        "best_f1_conf": None,
        "recommended": False,
        "note": "Auxiliary boundary prediction head attached at layer 15.",
    },
    "E5a": {
        "description": "YOLOv8n + Shape-IoU (Default Recommended)",
        "params": 3_005_843,
        "gflops": 8.19,
        "map50": 0.6324,
        "best_f1_conf": 0.063,
        "recommended": True,
        "note": "Default model: achieves highest test mAP50 (0.6324) and highest max-F1 (0.6244).",
    },
    "E5b": {
        "description": "YOLOv8n + MPDIoU (Ablation)",
        "params": 3_005_843,
        "gflops": 8.19,
        "map50": 0.5367,
        "best_f1_conf": None,
        "recommended": False,
        "note": "Ablation variant using MPDIoU bounding box regression.",
    },
    "E8": {
        "description": "Full Composite (P2BiFPN + Boundary Head + MPDIoU)",
        "params": 3_056_455,
        "gflops": 9.87,
        "map50": 0.6043,
        "best_f1_conf": None,
        "recommended": False,
        "note": "Composite architecture requiring custom _predict_once re-binding. Not recommended as default.",
    },
}


def _find_repo_root() -> Path:
    """Find repository root by looking for src/ and runs/ upward from current file."""
    cur = Path(__file__).resolve().parent
    for p in [cur, *cur.parents]:
        if (p / "runs").is_dir() and (p / "src").is_dir():
            return p
    return Path.cwd()


def _patch_main_namespace() -> None:
    """Expose custom classes on __main__ for unpickling models E4 and E8."""
    import src.models.fusion_neck as fn
    import src.models.boundary_branch as bb

    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        setattr(main_mod, "E8FusionNeck", fn.E8FusionNeck)
        setattr(main_mod, "_FastFusion", fn._FastFusion)
        setattr(main_mod, "BoundaryHead", bb.BoundaryHead)
        setattr(main_mod, "InterceptWrapper", bb.InterceptWrapper)

        if not hasattr(main_mod, "E8LossWrapper"):
            class E8LossWrapper:
                pass
            setattr(main_mod, "E8LossWrapper", E8LossWrapper)

        if not hasattr(main_mod, "BoundaryLossWrapper"):
            class BoundaryLossWrapper:
                pass
            setattr(main_mod, "BoundaryLossWrapper", BoundaryLossWrapper)


def resolve_checkpoint_path(name_or_path: Union[str, Path]) -> Tuple[str, Path]:
    """Resolve a model name (e.g. 'E5a') or file path to an absolute Path and canonical name."""
    repo_root = _find_repo_root()
    raw_str = str(name_or_path).strip()
    name_upper = raw_str.upper()

    # Direct name mapping
    name_map = {"E1": "E1", "E2": "E2", "E3": "E3", "E4": "E4", "E5A": "E5a", "E5B": "E5b", "E8": "E8"}
    if name_upper in name_map:
        canon_name = name_map[name_upper]
        ckpt = repo_root / "runs" / "detect" / "results" / "real" / canon_name / "weights" / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint for model {canon_name} not found at {ckpt}")
        return canon_name, ckpt

    # File path passed
    path = Path(raw_str)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {raw_str} (resolved to {path})")

    # Try to infer model name from path parts
    canon_name = path.stem
    for part in path.parts:
        if part.upper() in name_map:
            canon_name = name_map[part.upper()]
            break

    return canon_name, path


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_info(name_or_path: Union[str, Path]) -> Dict[str, Any]:
    """Return model metadata: path, size_bytes, sha256, params, map50, and best_f1_conf."""
    canon_name, ckpt_path = resolve_checkpoint_path(name_or_path)
    size_bytes = ckpt_path.stat().st_size
    sha256_hash = compute_sha256(ckpt_path)

    base_meta = MODEL_METADATA.get(canon_name, {})
    best_f1_conf = DOCUMENTED_THRESHOLDS.get(canon_name, base_meta.get("best_f1_conf", None))

    info = {
        "name": canon_name,
        "checkpoint_path": str(ckpt_path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "sha256": sha256_hash,
        "params": base_meta.get("params"),
        "gflops": base_meta.get("gflops"),
        "tile_map50": base_meta.get("map50"),
        "best_f1_conf": best_f1_conf,
        "description": base_meta.get("description", "Custom checkpoint"),
        "note": base_meta.get("note", ""),
    }
    return info


def load_model(
    name_or_path: Union[str, Path] = "E5a",
    device: str = "auto",
    half: bool = False,
) -> Tuple[YOLO, str]:
    """Load YOLO model with unpickling patch and E8 re-binding if applicable.

    Parameters
    ----------
    name_or_path : str or Path
        Standard model name ('E1', 'E2', 'E3', 'E4', 'E5a', 'E5b', 'E8') or .pt path.
    device : str
        Target device ('auto', 'cuda', 'cpu', or device index).
    half : bool
        Whether to use half precision (FP16).

    Returns
    -------
    (YOLO, str)
        Tuple of (loaded model, canonical model name).
    """
    canon_name, ckpt_path = resolve_checkpoint_path(name_or_path)

    # Patch namespace inside function, NOT on module import
    _patch_main_namespace()

    import src.models.fusion_neck as fn

    # Determine device
    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    yolo = YOLO(str(ckpt_path))

    # Handle E8 custom forward pass rebinding
    m = yolo.model
    is_e8 = (canon_name == "E8") or (hasattr(m, "fusion_neck") and isinstance(m.fusion_neck, fn.E8FusionNeck))

    if is_e8:
        if not hasattr(m, "fusion_neck"):
            raise RuntimeError(f"Model is identified as E8 but missing required 'fusion_neck' attribute.")
        bound_func = fn._make_e8_predict_once(m.fusion_neck).__get__(m, type(m))
        m._predict_once = bound_func

        # Assert that _predict_once is successfully rebound to custom closure
        co_filename = getattr(m._predict_once.__func__.__code__, "co_filename", "")
        if "fusion_neck" not in co_filename:
            raise RuntimeError(
                f"Assertion failed: E8 model _predict_once was not rebound to fusion_neck closure. "
                f"Resolved function code in: {co_filename}"
            )

    yolo.to(target_device)

    if half and target_device != "cpu" and torch.cuda.is_available():
        yolo.model.half()

    return yolo, canon_name
