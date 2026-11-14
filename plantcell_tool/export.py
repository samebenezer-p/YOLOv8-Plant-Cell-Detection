"""
Export utilities for detections: annotated overlay PNG, CSV, YOLO txt, and JSON metadata.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from .infer import DetectionResult
from .models import compute_sha256, resolve_checkpoint_path


def export_overlay(
    image_rgb: np.ndarray,
    result: DetectionResult,
    output_path: Union[str, Path],
    line_thickness: int = 2,
    box_color_rgb: tuple = (30, 215, 96),  # Crisp emerald green
) -> Path:
    """Save annotated overlay PNG with thin boxes, no text per box, and a clean header badge.

    Parameters
    ----------
    image_rgb : np.ndarray
        Full RGB image (H, W, 3).
    result : DetectionResult
        Inference detection result.
    output_path : str or Path
        Destination PNG path.
    line_thickness : int
        Box border thickness in pixels (default: 2).
    box_color_rgb : tuple
        RGB box border color.

    Returns
    -------
    Path
        Path to saved PNG file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV drawing and saving
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw thin boxes without cluttered text per box
    bgr_color = (box_color_rgb[2], box_color_rgb[1], box_color_rgb[0])
    for box in result.boxes_xyxy:
        x1, y1, x2, y2 = map(int, np.round(box))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, line_thickness, cv2.LINE_AA)

    # Draw header badge at top-left
    header_text = (
        f"PlantCell Tool | Model: {result.model_name} | "
        f"Count: {result.count} cells | Conf: {result.conf:.3f} | IoU: {result.iou:.2f}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85 if annotated.shape[1] >= 1920 else 0.55
    thickness = 2 if annotated.shape[1] >= 1920 else 1

    (tw, th), baseline = cv2.getTextSize(header_text, font, font_scale, thickness)
    pad = 12
    badge_x1, badge_y1 = 20, 20
    badge_x2, badge_y2 = badge_x1 + tw + 2 * pad, badge_y1 + th + 2 * pad

    # Draw dark background badge with subtle border
    cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (24, 24, 27), -1)
    cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (63, 63, 70), 1)

    # Draw header text
    text_origin = (badge_x1 + pad, badge_y1 + th + pad - 2)
    cv2.putText(
        annotated,
        header_text,
        text_origin,
        font,
        font_scale,
        (244, 244, 245),
        thickness,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_file), annotated)
    return out_file


def export_csv(
    result: DetectionResult,
    output_path: Union[str, Path],
) -> Path:
    """Export predictions to CSV in full-image pixel coordinates.

    Format: x1,y1,x2,y2,score,width,height
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "y1", "x2", "y2", "score", "width", "height"])
        for box, score in zip(result.boxes_xyxy, result.scores):
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            writer.writerow([
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                f"{score:.4f}",
                f"{w:.2f}",
                f"{h:.2f}",
            ])

    return out_file


def export_yolo(
    result: DetectionResult,
    image_shape: Tuple[int, int],
    output_path: Union[str, Path],
) -> Path:
    """Export predictions to normalized YOLO format txt.

    Format: <class_id> <x_center> <y_center> <width> <height>
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img_h, img_w = image_shape[:2]

    lines = []
    for box, cls_id in zip(result.boxes_xyxy, result.classes):
        x1, y1, x2, y2 = box.tolist()
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        norm_cx = cx / img_w
        norm_cy = cy / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        lines.append(f"{int(cls_id)} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return out_file


def export_json(
    result: DetectionResult,
    image_metadata: Dict[str, Any],
    output_path: Union[str, Path],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export complete prediction metadata, timings, and parameters to JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resolve model checkpoint sha256
    try:
        _, ckpt_path = resolve_checkpoint_path(result.model_name)
        ckpt_sha256 = compute_sha256(ckpt_path)
    except Exception:
        ckpt_sha256 = "UNKNOWN"

    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": result.model_name,
            "checkpoint_sha256": ckpt_sha256,
            "device": result.device,
            "half_precision": result.half,
        },
        "parameters": {
            "conf_threshold": result.conf,
            "iou_threshold": result.iou,
            "tile_size": result.tile_size,
            "overlap": result.overlap,
            "n_tiles": result.n_tiles,
        },
        "image": image_metadata,
        "timing_seconds": result.timing,
        "environment": result.versions,
        "detection_summary": {
            "count": result.count,
        },
        "detections": [
            {
                "x1": round(float(b[0]), 2),
                "y1": round(float(b[1]), 2),
                "x2": round(float(b[2]), 2),
                "y2": round(float(b[3]), 2),
                "score": round(float(s), 4),
                "class_id": int(c),
            }
            for b, s, c in zip(result.boxes_xyxy, result.scores, result.classes)
        ],
    }

    if extra_meta:
        payload["extra"] = extra_meta

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_file
