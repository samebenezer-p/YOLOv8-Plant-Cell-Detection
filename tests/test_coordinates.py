"""
Tests for coordinate transformation, clipping, and cross-tile NMS merging.
"""

import numpy as np
import pytest

from src.data.tiling import merge_tile_predictions


def test_coordinate_mapping():
    """Verify that a synthetic box inside a tile offsets correctly to full image coordinates."""
    # Suppose tile 1 is at (480, 0, 1120, 640)
    tx1, ty1, tx2, ty2 = 480, 0, 1120, 640
    # Box inside tile: (50, 60, 150, 180)
    tile_box = [50.0, 60.0, 150.0, 180.0]

    tile_preds = [{
        "boxes": np.array([tile_box], dtype=np.float32),
        "scores": np.array([0.95], dtype=np.float32),
        "classes": np.array([0], dtype=np.int32),
    }]
    coords = [(tx1, ty1, tx2, ty2)]
    image_shape = (2160, 3840)

    merged = merge_tile_predictions(tile_preds, coords, image_shape, iou_thresh=0.45)
    assert len(merged) == 1

    x1, y1, x2, y2, score, cls_id = merged[0]
    assert np.isclose(x1, 50.0 + tx1)
    assert np.isclose(y1, 60.0 + ty1)
    assert np.isclose(x2, 150.0 + tx1)
    assert np.isclose(y2, 180.0 + ty1)
    assert np.isclose(score, 0.95)
    assert int(cls_id) == 0


def test_overlap_deduplication():
    """Verify that the same object seen by two overlapping tiles collapses into a single detection."""
    # Tile 0: [0, 0, 640, 640]
    # Tile 1: [480, 0, 1120, 640] (overlap region: x in [480, 640])
    # An object at full-image x in [500, 600], y in [100, 200]
    # In Tile 0 coordinates: x in [500, 600], y in [100, 200]
    # In Tile 1 coordinates: x in [500 - 480, 600 - 480] = [20, 120], y in [100, 200]

    tile_preds = [
        {
            "boxes": np.array([[500.0, 100.0, 600.0, 200.0]], dtype=np.float32),
            "scores": np.array([0.90], dtype=np.float32),
            "classes": np.array([0], dtype=np.int32),
        },
        {
            "boxes": np.array([[20.0, 100.0, 120.0, 200.0]], dtype=np.float32),
            "scores": np.array([0.92], dtype=np.float32),
            "classes": np.array([0], dtype=np.int32),
        },
    ]
    coords = [(0, 0, 640, 640), (480, 0, 1120, 640)]
    image_shape = (2160, 3840)

    merged = merge_tile_predictions(tile_preds, coords, image_shape, iou_thresh=0.45)

    # Cross-tile NMS must collapse the two detections to 1
    assert len(merged) == 1
    # Higher score should be kept
    assert np.isclose(merged[0, 4], 0.92)
    assert np.isclose(merged[0, 0], 500.0)
    assert np.isclose(merged[0, 2], 600.0)


def test_border_clipping():
    """Verify that a box extending beyond the image boundaries is clipped."""
    # Tile at the right edge: [3200, 1520, 3840, 2160] on 3840x2160 image
    # A box whose right edge is at 650 (i.e. full image 3200 + 650 = 3850 > 3840)
    tx1, ty1 = 3200, 1520
    tile_preds = [{
        "boxes": np.array([[600.0, 500.0, 650.0, 660.0]], dtype=np.float32),
        "scores": np.array([0.80], dtype=np.float32),
        "classes": np.array([0], dtype=np.int32),
    }]
    coords = [(tx1, ty1, 3840, 2160)]
    image_shape = (2160, 3840)

    merged = merge_tile_predictions(tile_preds, coords, image_shape, iou_thresh=0.45)
    assert len(merged) == 1
    x1, y1, x2, y2, score, _ = merged[0]
    assert x2 <= 3840.0
    assert y2 <= 2160.0
    assert np.isclose(x2, 3840.0)
    assert np.isclose(y2, 2160.0)
