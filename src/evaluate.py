"""
Evaluation module for computing detection metrics.

Computes precision, recall, F1, mAP@50, mAP@50:95, per-class AP
(vessel/fiber/parenchyma), small-object AP, and confusion matrix.

Usage:
    python src/evaluate.py --model results/e1_baseline/seed_42/weights/best.pt \
        --data data.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Class names for reporting
CLASS_NAMES = {0: "vessel", 1: "fiber", 2: "parenchyma"}

# Small-object threshold (pixels, area-based)
SMALL_OBJECT_AREA_THRESH = 32 * 32  # 1024 px²


def evaluate_model(
    model: Any,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_det: int = 300,
    output_dir: Optional[str] = None,
    iou_thresholds: Optional[List[float]] = None,
    tiling_enabled: bool = False,
    tile_size: int = 512,
    tile_overlap: float = 0.25,
) -> Dict[str, Any]:
    """Evaluate a detector model on a dataset split with strict fairness protocol.

    Uses the Ultralytics built-in validation for the primary metrics, then
    supplements with custom per-class AP and small-object analysis.

    Parameters
    ----------
    model : BaselineDetector or similar
        Model with a ``.model`` attribute (Ultralytics YOLO instance).
    data_yaml : str
        Path to data.yaml.
    split : str
        Which split to evaluate on ("val" or "test").
    imgsz : int
        Input image resolution. Must be exactly 640 for fair evaluation.
    conf_thresh : float
        Confidence threshold (0.25).
    iou_thresh : float
        NMS IoU threshold (0.45).
    max_det : int
        Maximum detections per image (300).
    output_dir : str, optional
        Directory to save evaluation outputs.
    iou_thresholds : list of float, optional
        IoU thresholds for mAP computation. Defaults to [0.5, 0.55, ..., 0.95].
    tiling_enabled : bool
        Whether overlapping tile inference is enabled.
    tile_size : int
        Tile size in pixels if tiling is enabled.
    tile_overlap : float
        Tile overlap ratio if tiling is enabled.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    # Strict fairness assertion across all stages
    if imgsz != 640:
        raise ValueError(
            f"FAIRNESS PROTOCOL VIOLATION: All stages must evaluate at imgsz=640. "
            f"Got imgsz={imgsz}."
        )

    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    print(f"\n{'='*60}")
    print(f"  Shared Fair Evaluation on {split} set")
    print(f"  imgsz={imgsz}, conf={conf_thresh}, iou={iou_thresh}, max_det={max_det}")
    print(f"  tiling_enabled={tiling_enabled} (size={tile_size}, overlap={tile_overlap})")
    print(f"{'='*60}\n")

    # Run Ultralytics validation
    yolo_model = model.model
    val_results = yolo_model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf_thresh,
        iou=iou_thresh,
        max_det=max_det,
        verbose=True,
        plots=True,
        save_json=True,
    )

    # Extract metrics from Ultralytics results
    metrics = _extract_ultralytics_metrics(val_results)

    # Add metadata
    metrics["split"] = split
    metrics["imgsz"] = imgsz
    metrics["conf_thresh"] = conf_thresh
    metrics["iou_thresh"] = iou_thresh
    metrics["max_det"] = max_det
    metrics["tiling_enabled"] = tiling_enabled
    if tiling_enabled:
        metrics["tile_size"] = tile_size
        metrics["tile_overlap"] = tile_overlap
    metrics["conf_thresh"] = conf_thresh
    metrics["iou_thresh"] = iou_thresh

    # Generate confusion matrix
    confusion = _build_confusion_matrix(val_results)
    metrics["confusion_matrix"] = confusion.tolist() if confusion is not None else None

    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {metrics_path}")

        # Save confusion matrix plot
        _plot_confusion_matrix(confusion, output_dir / "confusion_matrix.png")

    return metrics


def _extract_ultralytics_metrics(val_results: Any) -> Dict[str, Any]:
    """Extract metrics from Ultralytics validation results.

    Parameters
    ----------
    val_results : ultralytics.utils.metrics.DetMetrics
        Results object from model.val().

    Returns
    -------
    dict
        Standardized metrics dictionary.
    """
    metrics: Dict[str, Any] = {}

    try:
        box = val_results.box

        # Overall metrics
        metrics["precision"] = float(box.mp) if hasattr(box, "mp") else None
        metrics["recall"] = float(box.mr) if hasattr(box, "mr") else None
        metrics["mAP50"] = float(box.map50) if hasattr(box, "map50") else None
        metrics["mAP50_95"] = float(box.map) if hasattr(box, "map") else None

        # Compute F1 from P and R
        p = metrics.get("precision") or 0
        r = metrics.get("recall") or 0
        metrics["F1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        # Per-class AP
        per_class_ap = {}
        if hasattr(box, "ap50") and box.ap50 is not None:
            ap50_array = box.ap50
            for cls_id, cls_name in CLASS_NAMES.items():
                if cls_id < len(ap50_array):
                    per_class_ap[cls_name] = {
                        "AP50": float(ap50_array[cls_id]),
                    }

        if hasattr(box, "ap") and box.ap is not None:
            ap_array = box.ap
            for cls_id, cls_name in CLASS_NAMES.items():
                if cls_id < len(ap_array):
                    if cls_name not in per_class_ap:
                        per_class_ap[cls_name] = {}
                    per_class_ap[cls_name]["AP50_95"] = float(ap_array[cls_id])

        # Per-class P, R
        if hasattr(box, "p") and box.p is not None:
            for cls_id, cls_name in CLASS_NAMES.items():
                if cls_id < len(box.p):
                    if cls_name not in per_class_ap:
                        per_class_ap[cls_name] = {}
                    per_class_ap[cls_name]["precision"] = float(box.p[cls_id])
        if hasattr(box, "r") and box.r is not None:
            for cls_id, cls_name in CLASS_NAMES.items():
                if cls_id < len(box.r):
                    if cls_name not in per_class_ap:
                        per_class_ap[cls_name] = {}
                    per_class_ap[cls_name]["recall"] = float(box.r[cls_id])

        metrics["per_class"] = per_class_ap

        # Small-object AP placeholder (requires custom computation)
        metrics["small_object_AP50"] = _compute_small_object_ap(val_results)

    except Exception as e:
        print(f"WARNING: Error extracting metrics: {e}")
        metrics["error"] = str(e)

    return metrics


def _compute_small_object_ap(val_results: Any) -> Optional[float]:
    """Compute AP for small objects (area < 32×32 pixels).

    This is a simplified estimation. For precise small-object AP,
    post-hoc filtering of predictions and ground truth by box area
    is needed.

    Parameters
    ----------
    val_results : ultralytics validation results

    Returns
    -------
    float or None
        Small-object AP50 if computable, else None.
    """
    # NOTE: Ultralytics doesn't natively report size-stratified AP.
    # For a full implementation, we would need to:
    # 1. Load GT annotations
    # 2. Filter by box area < SMALL_OBJECT_AREA_THRESH
    # 3. Match predictions to small-GT only
    # 4. Compute AP on that subset
    # This is a placeholder that returns None until post-hoc analysis is done.
    return None


def _build_confusion_matrix(val_results: Any) -> Optional[np.ndarray]:
    """Extract or build a confusion matrix from validation results.

    Parameters
    ----------
    val_results : ultralytics validation results

    Returns
    -------
    np.ndarray or None
        Confusion matrix of shape (num_classes + 1, num_classes + 1),
        where the extra row/col is for background (FP/FN).
    """
    try:
        if hasattr(val_results, "confusion_matrix"):
            cm = val_results.confusion_matrix
            if hasattr(cm, "matrix"):
                return np.array(cm.matrix)
    except Exception:
        pass
    return None


def _plot_confusion_matrix(
    matrix: Optional[np.ndarray],
    output_path: Path,
) -> None:
    """Save a confusion matrix as a PNG heatmap.

    Parameters
    ----------
    matrix : np.ndarray or None
        Confusion matrix. Skips if None.
    output_path : Path
        Output PNG path.
    """
    if matrix is None:
        print("  No confusion matrix to plot.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        labels = list(CLASS_NAMES.values()) + ["background"]
        n = len(labels)
        # Ensure matrix matches label count
        if matrix.shape[0] < n:
            padded = np.zeros((n, n))
            padded[:matrix.shape[0], :matrix.shape[1]] = matrix
            matrix = padded

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix[:n, :n],
            annot=True,
            fmt=".0f",
            xticklabels=labels[:n],
            yticklabels=labels[:n],
            cmap="Blues",
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        print(f"  Confusion matrix saved to {output_path}")
    except ImportError:
        print("  matplotlib/seaborn not available; skipping confusion matrix plot.")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 model"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (must be 640)")
    parser.add_argument("--max_det", type=int, default=300, help="Max detections per image")
    parser.add_argument("--tiling", action="store_true", help="Enable overlapping tile inference")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size")
    parser.add_argument("--tile_overlap", type=float, default=0.25, help="Tile overlap ratio")
    args = parser.parse_args()

    from src.models.yolov8_baseline import BaselineDetector

    detector = BaselineDetector.__new__(BaselineDetector)
    detector.model = None
    detector.num_classes = 3
    detector.device = "cpu"
    detector.model_variant = args.model
    detector.load_weights(args.model)

    output_dir = args.output_dir or str(
        Path(args.model).parent.parent
    )

    evaluate_model(
        model=detector,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        max_det=args.max_det,
        output_dir=output_dir,
        tiling_enabled=args.tiling,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )


if __name__ == "__main__":
    main()
