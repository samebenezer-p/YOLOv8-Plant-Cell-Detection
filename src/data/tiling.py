"""
Overlapping tile generation and cross-tile NMS prediction merging.

High-resolution micrographs are sliced into overlapping tiles for training
and inference. At prediction time, detections from overlapping tiles are
merged via cross-tile NMS to produce full-image predictions.

Usage:
    python -m src.data.tiling --image_dir data/raw --label_dir data/annotations \
        --output_dir data/tiles --tile_size 512 --overlap 0.25
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------------

def compute_tile_coords(
    img_h: int,
    img_w: int,
    tile_size: int,
    overlap: float,
) -> List[Tuple[int, int, int, int]]:
    """Compute (x1, y1, x2, y2) tile coordinates with overlap.

    Parameters
    ----------
    img_h, img_w : int
        Image dimensions.
    tile_size : int
        Square tile side length in pixels.
    overlap : float
        Fractional overlap between adjacent tiles (0.0–0.5 typical).

    Returns
    -------
    list of (x1, y1, x2, y2)
        Tile coordinates in absolute pixels.
    """
    step = int(tile_size * (1.0 - overlap))
    step = max(step, 1)

    tiles: List[Tuple[int, int, int, int]] = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            x1 = max(x2 - tile_size, 0)
            y1 = max(y2 - tile_size, 0)
            tiles.append((x1, y1, x2, y2))
    # Deduplicate (edge tiles may repeat)
    return list(dict.fromkeys(tiles))


def remap_labels_to_tile(
    labels: np.ndarray,
    tile_x1: int,
    tile_y1: int,
    tile_size: int,
    img_w: int,
    img_h: int,
    min_visible_fraction: float = 0.3,
) -> np.ndarray:
    """Remap YOLO labels from full-image to tile coordinates.

    Parameters
    ----------
    labels : np.ndarray
        Shape (N, 5) — [class, cx, cy, w, h] in normalized coordinates.
    tile_x1, tile_y1 : int
        Top-left corner of the tile in absolute pixels.
    tile_size : int
        Tile side length.
    img_w, img_h : int
        Original image dimensions.
    min_visible_fraction : float
        Minimum fraction of box area that must be inside the tile to keep it.

    Returns
    -------
    np.ndarray
        Remapped labels in tile-relative normalized coordinates.
    """
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    # Convert normalized YOLO → absolute pixel coords
    classes = labels[:, 0].astype(int)
    cx_abs = labels[:, 1] * img_w
    cy_abs = labels[:, 2] * img_h
    w_abs = labels[:, 3] * img_w
    h_abs = labels[:, 4] * img_h

    x1_abs = cx_abs - w_abs / 2
    y1_abs = cy_abs - h_abs / 2
    x2_abs = cx_abs + w_abs / 2
    y2_abs = cy_abs + h_abs / 2

    # Clip to tile
    tile_x2 = tile_x1 + tile_size
    tile_y2 = tile_y1 + tile_size

    clipped_x1 = np.clip(x1_abs, tile_x1, tile_x2)
    clipped_y1 = np.clip(y1_abs, tile_y1, tile_y2)
    clipped_x2 = np.clip(x2_abs, tile_x1, tile_x2)
    clipped_y2 = np.clip(y2_abs, tile_y1, tile_y2)

    clipped_w = clipped_x2 - clipped_x1
    clipped_h = clipped_y2 - clipped_y1
    clipped_area = clipped_w * clipped_h
    orig_area = w_abs * h_abs + 1e-6

    # Keep boxes with sufficient visible area
    keep = (clipped_area / orig_area) >= min_visible_fraction
    keep &= (clipped_w > 1) & (clipped_h > 1)  # non-degenerate

    if not np.any(keep):
        return np.zeros((0, 5), dtype=np.float32)

    # Convert to tile-relative normalized coords
    new_cx = (clipped_x1[keep] + clipped_x2[keep]) / 2 - tile_x1
    new_cy = (clipped_y1[keep] + clipped_y2[keep]) / 2 - tile_y1
    new_w = clipped_w[keep]
    new_h = clipped_h[keep]

    result = np.column_stack([
        classes[keep].astype(np.float32),
        new_cx / tile_size,
        new_cy / tile_size,
        new_w / tile_size,
        new_h / tile_size,
    ])
    return result


def generate_tiles(
    image_path: str | Path,
    label_path: Optional[str | Path],
    output_img_dir: str | Path,
    output_lbl_dir: str | Path,
    tile_size: int = 512,
    overlap: float = 0.25,
    min_visible_fraction: float = 0.3,
    pad_value: int = 0,
) -> List[Dict]:
    """Slice a single image into overlapping tiles, remap annotations.

    Parameters
    ----------
    image_path : path-like
        Path to source image.
    label_path : path-like or None
        Path to YOLO label file (class cx cy w h per line). None for
        unlabelled images (inference).
    output_img_dir, output_lbl_dir : path-like
        Directories for tile images and labels.
    tile_size : int
        Square tile side length.
    overlap : float
        Fractional overlap.
    min_visible_fraction : float
        Minimum visible box fraction to keep a label in a tile.
    pad_value : int
        Pixel value for padding edge tiles.

    Returns
    -------
    list of dict
        Each dict: ``{"tile_path": str, "label_path": str,
        "coords": (x1, y1, x2, y2), "source": str}``.
    """
    image_path = Path(image_path)
    output_img_dir = Path(output_img_dir)
    output_lbl_dir = Path(output_lbl_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_h, img_w = img.shape[:2]

    # Load labels
    labels = np.zeros((0, 5), dtype=np.float32)
    if label_path and Path(label_path).is_file():
        raw = np.loadtxt(str(label_path), ndmin=2)
        if raw.size > 0:
            labels = raw.astype(np.float32)

    tile_coords = compute_tile_coords(img_h, img_w, tile_size, overlap)
    stem = image_path.stem
    tile_records: List[Dict] = []

    for idx, (x1, y1, x2, y2) in enumerate(tile_coords):
        # Extract tile (pad if needed)
        tile = img[y1:y2, x1:x2]
        th, tw = tile.shape[:2]
        if th < tile_size or tw < tile_size:
            padded = np.full((tile_size, tile_size, 3), pad_value, dtype=np.uint8)
            padded[:th, :tw] = tile
            tile = padded

        tile_name = f"{stem}_tile{idx:04d}"
        tile_img_path = output_img_dir / f"{tile_name}.jpg"
        tile_lbl_path = output_lbl_dir / f"{tile_name}.txt"

        cv2.imwrite(str(tile_img_path), tile)

        # Remap labels
        tile_labels = remap_labels_to_tile(
            labels, x1, y1, tile_size, img_w, img_h, min_visible_fraction
        )
        if len(tile_labels) > 0:
            np.savetxt(
                str(tile_lbl_path), tile_labels,
                fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"],
            )
        else:
            tile_lbl_path.write_text("")

        tile_records.append({
            "tile_path": str(tile_img_path),
            "label_path": str(tile_lbl_path),
            "coords": (x1, y1, x2, y2),
            "source": str(image_path),
        })

    return tile_records


# ---------------------------------------------------------------------------
# Cross-tile NMS merge
# ---------------------------------------------------------------------------

def merge_tile_predictions(
    tile_preds: List[Dict],
    tile_coords: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    iou_thresh: float = 0.45,
) -> np.ndarray:
    """Merge detections from overlapping tiles via cross-tile NMS.

    Parameters
    ----------
    tile_preds : list of dict
        Each dict has keys ``"boxes"`` (N, 4) xyxy absolute in tile coords,
        ``"scores"`` (N,), ``"classes"`` (N,).
    tile_coords : list of (x1, y1, x2, y2)
        Tile positions in the original image.
    image_shape : (height, width)
        Original image dimensions (used for clipping).
    iou_thresh : float
        IoU threshold for NMS.

    Returns
    -------
    np.ndarray
        Merged detections: (M, 6) — [x1, y1, x2, y2, score, class].
    """
    all_boxes = []
    all_scores = []
    all_classes = []

    for preds, (tx1, ty1, _tx2, _ty2) in zip(tile_preds, tile_coords):
        boxes = np.array(preds["boxes"], dtype=np.float32)
        scores = np.array(preds["scores"], dtype=np.float32)
        classes = np.array(preds["classes"], dtype=np.int32)

        if len(boxes) == 0:
            continue

        # Offset boxes to full-image coordinates
        boxes[:, 0] += tx1
        boxes[:, 1] += ty1
        boxes[:, 2] += tx1
        boxes[:, 3] += ty1

        # Clip to image bounds
        img_h, img_w = image_shape
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)

    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32)

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    # Per-class NMS
    keep_indices = _per_class_nms(all_boxes, all_scores, all_classes, iou_thresh)

    merged = np.column_stack([
        all_boxes[keep_indices],
        all_scores[keep_indices],
        all_classes[keep_indices].astype(np.float32),
    ])
    return merged


def _per_class_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thresh: float,
) -> np.ndarray:
    """Apply NMS independently per class, return kept indices."""
    keep_all: List[int] = []
    for cls_id in np.unique(classes):
        mask = classes == cls_id
        cls_indices = np.where(mask)[0]
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]

        keep_local = _nms(cls_boxes, cls_scores, iou_thresh)
        keep_all.extend(cls_indices[keep_local].tolist())

    return np.array(sorted(keep_all), dtype=np.int64)


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
) -> List[int]:
    """Standard greedy NMS on (N, 4) xyxy boxes."""
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep: List[int] = []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order] - inter + 1e-6)

        suppress_mask = iou > iou_thresh
        for j, idx in enumerate(order):
            if suppress_mask[j]:
                suppressed[idx] = True

    return keep


# ---------------------------------------------------------------------------
# Full-image tiled inference with cross-tile NMS
# ---------------------------------------------------------------------------

def predict_tiled(
    model: Any,
    image: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.25,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_det: int = 300,
) -> np.ndarray:
    """Run overlapping tile inference on an image and merge predictions via cross-tile NMS.

    Parameters
    ----------
    model : YOLO
        Ultralytics YOLO model.
    image : np.ndarray
        Full image (H, W, 3) in BGR or RGB.
    tile_size : int
        Square tile size in pixels.
    overlap : float
        Fractional overlap between tiles.
    conf_thresh : float
        Confidence threshold.
    iou_thresh : float
        Cross-tile NMS IoU threshold.
    max_det : int
        Maximum detections per tile.

    Returns
    -------
    np.ndarray
        Merged detections: (M, 6) — [x1, y1, x2, y2, score, class].
    """
    img_h, img_w = image.shape[:2]
    coords = compute_tile_coords(img_h, img_w, tile_size, overlap)
    tile_preds: List[Dict] = []

    for (x1, y1, x2, y2) in coords:
        tile = image[y1:y2, x1:x2]
        th, tw = tile.shape[:2]
        if th < tile_size or tw < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:th, :tw] = tile
            tile = padded

        results = model.predict(
            tile,
            imgsz=tile_size,
            conf=conf_thresh,
            iou=iou_thresh,
            max_det=max_det,
            verbose=False,
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        tile_preds.append({"boxes": boxes, "scores": scores, "classes": classes})

    return merge_tile_predictions(tile_preds, coords, (img_h, img_w), iou_thresh=iou_thresh)


# ---------------------------------------------------------------------------
# Batch tile generation CLI
# ---------------------------------------------------------------------------

def generate_tiles_batch(
    image_dir: str | Path,
    label_dir: Optional[str | Path],
    output_dir: str | Path,
    tile_size: int = 512,
    overlap: float = 0.25,
    min_visible_fraction: float = 0.3,
) -> List[Dict]:
    """Generate tiles for all images in a directory."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_img_dir = output_dir / "images"
    output_lbl_dir = output_dir / "labels"

    all_records: List[Dict] = []
    extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    for img_path in sorted(image_dir.iterdir()):
        if not img_path.suffix.lower() in extensions:
            continue
        lbl_path = None
        if label_dir:
            lbl_path = Path(label_dir) / (img_path.stem + ".txt")

        records = generate_tiles(
            img_path, lbl_path, output_img_dir, output_lbl_dir,
            tile_size=tile_size, overlap=overlap,
            min_visible_fraction=min_visible_fraction,
        )
        all_records.extend(records)
        print(f"  {img_path.name} → {len(records)} tiles")

    print(f"\nTotal: {len(all_records)} tiles generated in {output_dir}")
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate overlapping tiles")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/tiles")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min_visible", type=float, default=0.3)
    args = parser.parse_args()

    generate_tiles_batch(
        args.image_dir, args.label_dir, args.output_dir,
        tile_size=args.tile_size, overlap=args.overlap,
        min_visible_fraction=args.min_visible,
    )


if __name__ == "__main__":
    main()
