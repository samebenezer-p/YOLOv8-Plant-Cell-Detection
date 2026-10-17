"""
Error analysis module — categorizes detection errors per Table IX.

Error categories:
1. Small-object misses — FN where GT box area < threshold
2. Dense-packing errors — FP/FN in regions with high box density
3. Fiber↔parenchyma confusion — class confusion between fibers and parenchyma
4. Boundary localization — detections with IoU ∈ [0.3, 0.5] (poor localization)
5. False positives — detections with no matching GT
6. False negatives — GT boxes with no matching detection

Usage:
    python src/error_analysis.py --model results/e1_baseline/seed_42/weights/best.pt \
        --data data.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "vessel", 1: "fiber", 2: "parenchyma"}

# Error thresholds
SMALL_OBJECT_AREA_PX = 32 * 32  # 1024 px²
DENSE_REGION_K = 5  # number of neighbors for density estimation
DENSE_REGION_RADIUS = 50  # pixel radius for density counting
LOCALIZATION_IOU_LOW = 0.3
LOCALIZATION_IOU_HIGH = 0.5


# ---------------------------------------------------------------------------
# Core error analysis
# ---------------------------------------------------------------------------

def analyze_errors(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    img_size: int = 512,
) -> Dict[str, Any]:
    """Categorize detection errors across a dataset.

    Parameters
    ----------
    predictions : list of dict
        Per-image predictions. Each dict has:
        ``{"image": str, "detections": [{"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int}]}``.
    ground_truths : list of dict
        Per-image ground truths. Each dict has:
        ``{"image": str, "boxes": [{"bbox": [x1,y1,x2,y2], "class_id": int}]}``.
    iou_threshold : float
        IoU threshold for matching predictions to ground truth.
    img_size : int
        Image size (for area normalization).

    Returns
    -------
    dict
        Error analysis results with categorized error counts and examples.
    """
    error_stats = {
        "small_object_misses": {"count": 0, "examples": []},
        "dense_packing_errors": {"count": 0, "examples": []},
        "fiber_parenchyma_confusion": {"count": 0, "examples": []},
        "boundary_localization": {"count": 0, "examples": []},
        "false_positives": {"count": 0, "per_class": defaultdict(int), "examples": []},
        "false_negatives": {"count": 0, "per_class": defaultdict(int), "examples": []},
        "total_predictions": 0,
        "total_ground_truths": 0,
        "total_images": len(predictions),
    }

    for pred_item, gt_item in zip(predictions, ground_truths):
        preds = pred_item.get("detections", [])
        gts = gt_item.get("boxes", [])

        error_stats["total_predictions"] += len(preds)
        error_stats["total_ground_truths"] += len(gts)

        if not gts and not preds:
            continue

        # Convert to arrays for vectorized operations
        pred_boxes = np.array([d["bbox"] for d in preds], dtype=np.float32) if preds else np.zeros((0, 4))
        pred_classes = np.array([d["class_id"] for d in preds], dtype=np.int32) if preds else np.zeros(0, dtype=np.int32)
        pred_confs = np.array([d["confidence"] for d in preds], dtype=np.float32) if preds else np.zeros(0)

        gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4))
        gt_classes = np.array([g["class_id"] for g in gts], dtype=np.int32) if gts else np.zeros(0, dtype=np.int32)

        # Compute IoU matrix
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = _compute_iou_matrix(pred_boxes, gt_boxes)
        else:
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

        # Match predictions to GT (greedy)
        pred_matched = np.full(len(preds), -1, dtype=np.int32)
        gt_matched = np.full(len(gts), -1, dtype=np.int32)

        if iou_matrix.size > 0:
            _greedy_match(iou_matrix, pred_classes, gt_classes,
                         iou_threshold, pred_matched, gt_matched)

        image_name = pred_item.get("image", "unknown")

        # Analyze unmatched predictions (FP)
        for i, det in enumerate(preds):
            if pred_matched[i] < 0:
                cls_name = CLASS_NAMES.get(det["class_id"], str(det["class_id"]))
                error_stats["false_positives"]["count"] += 1
                error_stats["false_positives"]["per_class"][cls_name] += 1

                # Check for boundary localization (near-miss)
                if len(gt_boxes) > 0:
                    max_iou = float(iou_matrix[i].max())
                    if LOCALIZATION_IOU_LOW <= max_iou < LOCALIZATION_IOU_HIGH:
                        best_gt = int(iou_matrix[i].argmax())
                        error_stats["boundary_localization"]["count"] += 1
                        if len(error_stats["boundary_localization"]["examples"]) < 20:
                            error_stats["boundary_localization"]["examples"].append({
                                "image": image_name,
                                "pred_class": cls_name,
                                "gt_class": CLASS_NAMES.get(gt_classes[best_gt], str(gt_classes[best_gt])),
                                "iou": max_iou,
                            })

                # Check for fiber↔parenchyma confusion
                if len(gt_boxes) > 0 and det["class_id"] in (1, 2):
                    best_gt_idx = int(iou_matrix[i].argmax())
                    best_iou = float(iou_matrix[i, best_gt_idx])
                    if best_iou >= iou_threshold:
                        gt_cls = gt_classes[best_gt_idx]
                        if (det["class_id"] == 1 and gt_cls == 2) or \
                           (det["class_id"] == 2 and gt_cls == 1):
                            error_stats["fiber_parenchyma_confusion"]["count"] += 1
                            if len(error_stats["fiber_parenchyma_confusion"]["examples"]) < 20:
                                error_stats["fiber_parenchyma_confusion"]["examples"].append({
                                    "image": image_name,
                                    "pred_class": CLASS_NAMES[det["class_id"]],
                                    "gt_class": CLASS_NAMES[gt_cls],
                                    "iou": best_iou,
                                })

                # Add FP example
                if len(error_stats["false_positives"]["examples"]) < 20:
                    error_stats["false_positives"]["examples"].append({
                        "image": image_name,
                        "class": cls_name,
                        "confidence": float(det["confidence"]),
                    })

        # Analyze unmatched GT (FN)
        for j, gt in enumerate(gts):
            if gt_matched[j] < 0:
                cls_name = CLASS_NAMES.get(gt["class_id"], str(gt["class_id"]))
                error_stats["false_negatives"]["count"] += 1
                error_stats["false_negatives"]["per_class"][cls_name] += 1

                # Check if small object
                bbox = gt["bbox"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < SMALL_OBJECT_AREA_PX:
                    error_stats["small_object_misses"]["count"] += 1
                    if len(error_stats["small_object_misses"]["examples"]) < 20:
                        error_stats["small_object_misses"]["examples"].append({
                            "image": image_name,
                            "class": cls_name,
                            "area_px": float(area),
                        })

                # Check if in dense region
                if _is_dense_region(gt_boxes, j, DENSE_REGION_K, DENSE_REGION_RADIUS):
                    error_stats["dense_packing_errors"]["count"] += 1
                    if len(error_stats["dense_packing_errors"]["examples"]) < 20:
                        error_stats["dense_packing_errors"]["examples"].append({
                            "image": image_name,
                            "class": cls_name,
                        })

                # Add FN example
                if len(error_stats["false_negatives"]["examples"]) < 20:
                    error_stats["false_negatives"]["examples"].append({
                        "image": image_name,
                        "class": cls_name,
                    })

    # Convert defaultdicts for JSON serialization
    error_stats["false_positives"]["per_class"] = dict(error_stats["false_positives"]["per_class"])
    error_stats["false_negatives"]["per_class"] = dict(error_stats["false_negatives"]["per_class"])

    # Summary table
    error_stats["summary_table"] = _build_summary_table(error_stats)

    return error_stats


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Parameters
    ----------
    boxes_a : (M, 4)
    boxes_b : (N, 4)

    Returns
    -------
    np.ndarray
        IoU matrix of shape (M, N).
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def _greedy_match(
    iou_matrix: np.ndarray,
    pred_classes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
    pred_matched: np.ndarray,
    gt_matched: np.ndarray,
) -> None:
    """Greedy matching of predictions to ground truth.

    Matches predictions to GT in order of decreasing IoU, requiring class
    match and IoU ≥ threshold. Modifies pred_matched and gt_matched in-place.
    """
    if iou_matrix.size == 0:
        return

    # Get all (pred, gt) pairs sorted by IoU descending
    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for flat_idx in flat_indices:
        pi = flat_idx // iou_matrix.shape[1]
        gi = flat_idx % iou_matrix.shape[1]

        if iou_matrix[pi, gi] < iou_threshold:
            break  # All remaining have lower IoU

        if pred_matched[pi] >= 0 or gt_matched[gi] >= 0:
            continue  # Already matched

        if pred_classes[pi] == gt_classes[gi]:
            pred_matched[pi] = gi
            gt_matched[gi] = pi


def _is_dense_region(
    gt_boxes: np.ndarray,
    box_idx: int,
    k: int,
    radius: float,
) -> bool:
    """Check if a GT box is in a dense region (many nearby boxes).

    Parameters
    ----------
    gt_boxes : (N, 4) xyxy boxes.
    box_idx : int
        Index of the box to check.
    k : int
        Minimum number of neighbors to be considered dense.
    radius : float
        Pixel radius for neighbor counting.

    Returns
    -------
    bool
    """
    if len(gt_boxes) <= k:
        return False

    # Use box centers
    centers = (gt_boxes[:, :2] + gt_boxes[:, 2:4]) / 2
    target = centers[box_idx]
    distances = np.sqrt(np.sum((centers - target) ** 2, axis=1))
    distances[box_idx] = np.inf  # exclude self
    neighbors_in_radius = np.sum(distances < radius)
    return int(neighbors_in_radius) >= k


def _build_summary_table(error_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a summary table suitable for Table IX formatting.

    Parameters
    ----------
    error_stats : dict
        Error analysis results.

    Returns
    -------
    list of dict
        Rows for the summary table.
    """
    total_gt = error_stats["total_ground_truths"]
    total_pred = error_stats["total_predictions"]

    rows = [
        {
            "category": "Small-object misses",
            "count": error_stats["small_object_misses"]["count"],
            "pct_of_fn": _safe_pct(
                error_stats["small_object_misses"]["count"],
                error_stats["false_negatives"]["count"],
            ),
        },
        {
            "category": "Dense-packing errors",
            "count": error_stats["dense_packing_errors"]["count"],
            "pct_of_fn": _safe_pct(
                error_stats["dense_packing_errors"]["count"],
                error_stats["false_negatives"]["count"],
            ),
        },
        {
            "category": "Fiber↔Parenchyma confusion",
            "count": error_stats["fiber_parenchyma_confusion"]["count"],
            "pct_of_errors": _safe_pct(
                error_stats["fiber_parenchyma_confusion"]["count"],
                error_stats["false_positives"]["count"] + error_stats["false_negatives"]["count"],
            ),
        },
        {
            "category": "Boundary localization",
            "count": error_stats["boundary_localization"]["count"],
            "pct_of_fp": _safe_pct(
                error_stats["boundary_localization"]["count"],
                error_stats["false_positives"]["count"],
            ),
        },
        {
            "category": "Total FP",
            "count": error_stats["false_positives"]["count"],
            "per_class": error_stats["false_positives"]["per_class"],
        },
        {
            "category": "Total FN",
            "count": error_stats["false_negatives"]["count"],
            "per_class": error_stats["false_negatives"]["per_class"],
        },
    ]
    return rows


def _safe_pct(numerator: int, denominator: int) -> Optional[float]:
    """Compute percentage, returning None if denominator is zero."""
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 1)


# ---------------------------------------------------------------------------
# High-level runner using the model
# ---------------------------------------------------------------------------

def run_error_analysis(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run error analysis on a trained model.

    Parameters
    ----------
    model_path : str
        Path to trained model weights.
    data_yaml : str
        Path to data.yaml.
    split : str
        Dataset split.
    conf_thresh : float
        Confidence threshold.
    iou_thresh : float
        IoU threshold for matching.
    output_dir : str, optional
        Output directory.

    Returns
    -------
    dict
        Error analysis results.
    """
    import yaml as _yaml
    from src.models.yolov8_baseline import BaselineDetector

    # Load data config
    with open(data_yaml, "r") as f:
        data_cfg = _yaml.safe_load(f)

    data_root = Path(data_yaml).parent
    if "path" in data_cfg:
        data_root = Path(data_cfg["path"])
        if not data_root.is_absolute():
            data_root = Path(data_yaml).parent / data_root

    img_dir = data_root / data_cfg.get(split, f"images/{split}")
    lbl_dir = data_root / str(data_cfg.get(split, f"images/{split}")).replace("images", "labels")

    # Load model
    detector = BaselineDetector(model_variant=model_path, num_classes=3)

    # Get predictions
    img_paths = sorted([
        str(p) for p in Path(img_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    predictions = []
    ground_truths = []

    for img_path in img_paths:
        # Predict
        preds = detector.predict(img_path, conf=conf_thresh, iou=iou_thresh)
        predictions.append(preds[0] if preds else {"image": img_path, "detections": []})

        # Load GT
        lbl_path = Path(lbl_dir) / (Path(img_path).stem + ".txt")
        gt_boxes = []
        if lbl_path.is_file():
            import cv2
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                labels = np.loadtxt(str(lbl_path), ndmin=2)
                if labels.size > 0:
                    for row in labels:
                        cls, cx, cy, bw, bh = row
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        gt_boxes.append({
                            "bbox": [x1, y1, x2, y2],
                            "class_id": int(cls),
                        })
        ground_truths.append({"image": img_path, "boxes": gt_boxes})

    # Analyze
    results = analyze_errors(predictions, ground_truths, iou_threshold=iou_thresh)

    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "error_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nError analysis saved to {out_path}")

    # Print summary
    print(f"\n{'='*50}")
    print("  ERROR ANALYSIS SUMMARY")
    print(f"{'='*50}")
    for row in results.get("summary_table", []):
        pct_key = [k for k in row if k.startswith("pct")]
        pct = f" ({row[pct_key[0]]}%)" if pct_key and row[pct_key[0]] is not None else ""
        print(f"  {row['category']:>35s}: {row['count']}{pct}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error analysis for detection results"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or str(
        Path(args.model).parent.parent
    )

    run_error_analysis(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
