"""
Image I/O utilities for plant cell microscopy images.

Reads 8-bit, 16-bit, grayscale, and RGBA images from JPG, PNG, TIF, and TIFF files
into standard RGB uint8 numpy arrays without resizing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def read_image(
    file_path: Union[str, Path]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an image file into an RGB uint8 array and return image metadata.

    Parameters
    ----------
    file_path : str or Path
        Path to image file (supports .jpg, .jpeg, .png, .tif, .tiff).

    Returns
    -------
    (np.ndarray, dict)
        Tuple of (RGB uint8 array with shape (H, W, 3), metadata dict).

    Raises
    ------
    FileNotFoundError
        If image file does not exist.
    ValueError
        If image dimensions are less than 64 pixels or format is unsupported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    ext = path.suffix.lower()
    raw_img: Optional[np.ndarray] = None
    read_method = "cv2"

    # For TIFF images, attempt reading with tifffile first if available
    if ext in (".tif", ".tiff") and HAS_TIFFFILE:
        try:
            raw_img = tifffile.imread(str(path))
            read_method = "tifffile"
        except Exception:
            raw_img = None

    # Fallback to OpenCV unchanged read
    if raw_img is None:
        raw_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        read_method = "cv2"

    # Fallback to PIL
    if raw_img is None:
        try:
            with Image.open(str(path)) as pil_img:
                raw_img = np.array(pil_img)
            read_method = "PIL"
        except Exception as e:
            raise ValueError(f"Failed to decode image from {path}: {e}")

    if raw_img is None or raw_img.size == 0:
        raise ValueError(f"Could not decode valid image data from {path}")

    orig_dtype = str(raw_img.dtype)
    orig_shape = raw_img.shape
    orig_max = float(raw_img.max()) if raw_img.size > 0 else 0.0
    scaled_16bit = False

    # Check minimum dimensions
    h, w = raw_img.shape[:2]
    if h < 64 or w < 64:
        raise ValueError(
            f"Image dimensions ({w}x{h}) are smaller than minimum allowed 64 px: {path}"
        )

    # 16-bit integer conversion: scale by 1/257 if max > 255
    if np.issubdtype(raw_img.dtype, np.integer) and raw_img.itemsize == 2:
        if orig_max > 255:
            # 65535 / 257 = 255.0
            norm = (raw_img.astype(np.float32) / 257.0).round()
            img_uint8 = np.clip(norm, 0, 255).astype(np.uint8)
            scaled_16bit = True
        else:
            img_uint8 = raw_img.astype(np.uint8)
    elif np.issubdtype(raw_img.dtype, np.floating):
        if orig_max <= 1.0:
            img_uint8 = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = np.clip(raw_img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = raw_img.astype(np.uint8)

    # Channel normalization to RGB
    if img_uint8.ndim == 2:
        # Grayscale -> RGB
        rgb_img = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    elif img_uint8.ndim == 3:
        channels = img_uint8.shape[2]
        if channels == 1:
            rgb_img = cv2.cvtColor(img_uint8[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            # If loaded via cv2, it's BGR; if tifffile/PIL, it's RGB
            if read_method == "cv2":
                rgb_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = img_uint8
        elif channels == 4:
            # RGBA
            if read_method == "cv2":
                rgb_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
            else:
                rgb_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported number of channels ({channels}) in image: {path}")
    else:
        raise ValueError(f"Unsupported image array dimensions ({img_uint8.ndim}): {path}")

    metadata = {
        "path": str(path),
        "filename": path.name,
        "width": int(rgb_img.shape[1]),
        "height": int(rgb_img.shape[0]),
        "orig_shape": orig_shape,
        "orig_dtype": orig_dtype,
        "orig_max": orig_max,
        "scaled_16bit": scaled_16bit,
        "read_method": read_method,
    }

    return rgb_img, metadata
