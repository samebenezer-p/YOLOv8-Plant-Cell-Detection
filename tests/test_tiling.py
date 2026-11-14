"""
Tests for tiling geometry and pixel coverage.
"""

import numpy as np
import pytest

from src.data.tiling import compute_tile_coords


def test_coverage_3840x2160():
    """Verify that every single pixel of a 3840x2160 micrograph is covered by at least one tile."""
    img_h, img_w = 2160, 3840
    tile_size = 640
    overlap = 0.25

    coords = compute_tile_coords(img_h, img_w, tile_size, overlap)

    # Exactly 40 tiles (8 columns x 5 rows)
    assert len(coords) == 40, f"Expected 40 tiles for 3840x2160, got {len(coords)}"

    # Check complete pixel coverage
    mask = np.zeros((img_h, img_w), dtype=bool)
    for x1, y1, x2, y2 in coords:
        assert x2 - x1 == tile_size, f"Tile width {x2 - x1} != {tile_size}"
        assert y2 - y1 == tile_size, f"Tile height {y2 - y1} != {tile_size}"
        assert 0 <= x1 < x2 <= img_w
        assert 0 <= y1 < y2 <= img_h
        mask[y1:y2, x1:x2] = True

    assert np.all(mask), "Not all pixels in 3840x2160 were covered by tiles!"


@pytest.mark.parametrize(
    "img_h, img_w",
    [
        (500, 700),
        (640, 640),
        (100, 100),
        (1000, 1000),
        (2160, 3840),
        (1080, 1920),
    ],
)
def test_coverage_arbitrary_dimensions(img_h, img_w):
    """Verify complete pixel coverage across various image dimensions."""
    tile_size = 640
    overlap = 0.25
    coords = compute_tile_coords(img_h, img_w, tile_size, overlap)

    mask = np.zeros((img_h, img_w), dtype=bool)
    for x1, y1, x2, y2 in coords:
        mask[y1:y2, x1:x2] = True

    assert np.all(mask), f"Failed full coverage for dimensions ({img_w}x{img_h})"
