"""
High-throughput full-image tiled inference engine with batched prediction and cross-tile NMS.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics import YOLO

from src.data.tiling import compute_tile_coords, merge_tile_predictions


@dataclass
class DetectionResult:
    boxes_xyxy: np.ndarray        # (N, 4) float32
    scores: np.ndarray            # (N,) float32
    classes: np.ndarray           # (N,) int32
    count: int
    n_tiles: int
    tile_size: int
    overlap: float
    conf: float
    iou: float
    timing: Dict[str, float]      # timing in seconds: tiling_and_inference, merge, total
    device: str
    half: bool
    model_name: str
    versions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dict format."""
        d = asdict(self)
        d["boxes_xyxy"] = self.boxes_xyxy.tolist()
        d["scores"] = self.scores.tolist()
        d["classes"] = self.classes.tolist()
        return d


def _filter_border_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    tile_coord: Tuple[int, int, int, int],
    tile_size: int,
    image_shape: Tuple[int, int],
    margin: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discard detections touching interior tile edges within `margin` pixels."""
    if len(boxes) == 0:
        return boxes, scores, classes

    tx1, ty1, tx2, ty2 = tile_coord
    img_h, img_w = image_shape

    is_interior_left = (tx1 > 0)
    is_interior_top = (ty1 > 0)
    is_interior_right = (tx2 < img_w)
    is_interior_bottom = (ty2 < img_h)

    keep = np.ones(len(boxes), dtype=bool)

    if is_interior_left:
        keep &= (boxes[:, 0] > margin)
    if is_interior_top:
        keep &= (boxes[:, 1] > margin)
    if is_interior_right:
        keep &= (boxes[:, 2] < (tile_size - margin))
    if is_interior_bottom:
        keep &= (boxes[:, 3] < (tile_size - margin))

    return boxes[keep], scores[keep], classes[keep]


def detect_full_image(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    iou: float = 0.45,
    tile: int = 640,
    overlap: float = 0.25,
    batch: int = 16,
    max_det: int = 300,
    half: bool = False,
    drop_border_boxes: bool = False,
    model_name: str = "E5a",
) -> DetectionResult:
    """Run overlapping tile inference on a full-size image with batched GPU execution.

    Parameters
    ----------
    model : YOLO
        Loaded Ultralytics model instance.
    image : np.ndarray
        Full image (H, W, 3) in RGB uint8 format.
    conf : float
        Confidence threshold.
    iou : float
        Cross-tile NMS IoU threshold (default: 0.45).
    tile : int
        Tile size in pixels (default: 640).
    overlap : float
        Tile overlap fraction (default: 0.25).
    batch : int
        Batch size for model.predict (default: 16).
    max_det : int
        Maximum detections per tile (default: 300).
    half : bool
        Whether FP16 half precision is enabled.
    drop_border_boxes : bool
        If True, discard boxes within 3px of interior tile edges before NMS.
    model_name : str
        Canonical name of the model.

    Returns
    -------
    DetectionResult
        Dataclass containing merged detections, counts, timings, and metadata.
    """
    total_start = time.perf_counter()
    img_h, img_w = image.shape[:2]

    # Grid tile coordinates (same logic as src/data/tiling.py)
    coords = compute_tile_coords(img_h, img_w, tile, overlap)
    n_tiles = len(coords)

    # Prepare all tiles with zero padding if needed
    tiles_list: List[np.ndarray] = []
    for (x1, y1, x2, y2) in coords:
        sub_tile = image[y1:y2, x1:x2]
        th, tw = sub_tile.shape[:2]
        if th < tile or tw < tile:
            padded = np.zeros((tile, tile, 3), dtype=np.uint8)
            padded[:th, :tw] = sub_tile
            tiles_list.append(padded)
        else:
            tiles_list.append(sub_tile)

    # Infer in batches to respect VRAM limits and accelerate throughput
    infer_start = time.perf_counter()
    tile_preds: List[Dict[str, np.ndarray]] = []

    # Detect device
    device_name = str(next(model.model.parameters()).device) if hasattr(model, "model") else "cpu"

    for batch_idx in range(0, n_tiles, batch):
        batch_slice = tiles_list[batch_idx : batch_idx + batch]
        coord_slice = coords[batch_idx : batch_idx + batch]

        # Call predict on batch of images
        predict_kwargs = dict(
            imgsz=tile,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
            device=device_name,
        )
        if half:
            predict_kwargs["half"] = True

        results = model.predict(
            batch_slice,
            **predict_kwargs,
        )

        for res, coord in zip(results, coord_slice):
            b = res.boxes.xyxy.cpu().numpy()
            s = res.boxes.conf.cpu().numpy()
            c = res.boxes.cls.cpu().numpy().astype(np.int32)

            if drop_border_boxes:
                b, s, c = _filter_border_boxes(
                    b, s, c, coord, tile, (img_h, img_w), margin=3
                )

            tile_preds.append({"boxes": b, "scores": s, "classes": c})

    infer_time = time.perf_counter() - infer_start

    # Merge tile predictions using cross-tile NMS
    merge_start = time.perf_counter()
    merged = merge_tile_predictions(
        tile_preds, coords, (img_h, img_w), iou_thresh=iou
    )
    merge_time = time.perf_counter() - merge_start
    total_time = time.perf_counter() - total_start

    if len(merged) > 0:
        boxes_xyxy = merged[:, :4].astype(np.float32)
        scores = merged[:, 4].astype(np.float32)
        classes = merged[:, 5].astype(np.int32)
    else:
        boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int32)

    import ultralytics
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }

    timing = {
        "tiling_and_inference": round(infer_time, 4),
        "merge": round(merge_time, 4),
        "total": round(total_time, 4),
    }

    return DetectionResult(
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        classes=classes,
        count=int(len(boxes_xyxy)),
        n_tiles=n_tiles,
        tile_size=tile,
        overlap=overlap,
        conf=conf,
        iou=iou,
        timing=timing,
        device=device_name,
        half=half,
        model_name=model_name,
        versions=versions,
    )
