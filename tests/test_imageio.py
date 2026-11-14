"""
Tests for image reading, type conversion, 16-bit scaling, and channel normalization.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from plantcell_tool.imageio import read_image


def test_imageio_8bit_rgb(tmp_path: Path):
    """Verify 8-bit RGB image loads into (H, W, 3) uint8."""
    img_data = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    p = tmp_path / "test_8bit.jpg"
    # Save BGR for OpenCV
    cv2.imwrite(str(p), cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))

    loaded, meta = read_image(p)
    assert loaded.shape == (128, 128, 3)
    assert loaded.dtype == np.uint8
    assert not meta["scaled_16bit"]


def test_imageio_16bit_gray_tiff(tmp_path: Path):
    """Verify 16-bit grayscale TIFF scales by 1/257 when max > 255."""
    # Create 16-bit grayscale array with values spanning up to 65535
    img_16 = np.linspace(0, 65535, 128 * 128, dtype=np.uint16).reshape((128, 128))
    p = tmp_path / "test_16bit.tif"
    cv2.imwrite(str(p), img_16)

    loaded, meta = read_image(p)
    assert loaded.shape == (128, 128, 3)
    assert loaded.dtype == np.uint8
    assert meta["scaled_16bit"] is True
    # Max value in 16-bit (65535) scaled by 1/257 should be 255
    assert loaded.max() == 255
    assert loaded.min() == 0


def test_imageio_rgba_png(tmp_path: Path):
    """Verify RGBA PNG loads into (H, W, 3) uint8 RGB."""
    img_rgba = (np.random.rand(100, 100, 4) * 255).astype(np.uint8)
    p = tmp_path / "test_rgba.png"
    Image.fromarray(img_rgba, mode="RGBA").save(str(p))

    loaded, meta = read_image(p)
    assert loaded.shape == (100, 100, 3)
    assert loaded.dtype == np.uint8


def test_imageio_reject_undersized(tmp_path: Path):
    """Verify that images smaller than 64 px are rejected with a clear ValueError."""
    tiny = (np.random.rand(40, 50, 3) * 255).astype(np.uint8)
    p = tmp_path / "tiny.png"
    cv2.imwrite(str(p), tiny)

    with pytest.raises(ValueError, match="smaller than minimum allowed 64 px"):
        read_image(p)
