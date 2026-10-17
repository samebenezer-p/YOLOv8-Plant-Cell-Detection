"""
Boundary heatmap target generation for the auxiliary boundary branch (E4).

Generates single-channel heatmaps where annotated bounding-box edges are
rendered as thin bands. Supports both binary and Gaussian-smoothed modes.

Usage:
    python -m src.data.boundary_targets --label_dir labels/train \
        --output_dir data/boundary_maps --img_size 512 --thickness 3 --mode gaussian
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import cv2
import numpy as np


def generate_boundary_heatmap(
    labels: np.ndarray,
    img_size: int | Tuple[int, int],
    thickness: int = 3,
    mode: Literal["binary", "gaussian"] = "binary",
    sigma: float = 1.5,
) -> np.ndarray:
    """Render a single-channel boundary heatmap from YOLO box annotations.

    Parameters
    ----------
    labels : np.ndarray
        Shape (N, 5) — [class, cx, cy, w, h] in normalized coordinates.
    img_size : int or (height, width)
        Target heatmap size. If int, assumed square.
    thickness : int
        Band thickness in pixels for box edges.
    mode : {"binary", "gaussian"}
        ``"binary"`` draws 1-pixel-thick rectangles then dilates;
        ``"gaussian"`` additionally applies Gaussian blur for smooth falloff.
    sigma : float
        Gaussian blur sigma (only used when ``mode="gaussian"``).

    Returns
    -------
    np.ndarray
        Float32 heatmap in [0, 1], shape (H, W).
    """
    if isinstance(img_size, int):
        h, w = img_size, img_size
    else:
        h, w = img_size

    heatmap = np.zeros((h, w), dtype=np.float32)

    if len(labels) == 0:
        return heatmap

    for row in labels:
        _cls, cx, cy, bw, bh = row
        # Convert normalized → absolute pixel
        cx_abs = cx * w
        cy_abs = cy * h
        bw_abs = bw * w
        bh_abs = bh * h

        x1 = int(round(cx_abs - bw_abs / 2))
        y1 = int(round(cy_abs - bh_abs / 2))
        x2 = int(round(cx_abs + bw_abs / 2))
        y2 = int(round(cy_abs + bh_abs / 2))

        # Clip to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Draw rectangle edges
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), 1.0, thickness=1)

    if mode == "binary":
        # Dilate to desired thickness
        if thickness > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (thickness, thickness)
            )
            heatmap = cv2.dilate(heatmap, kernel)
        heatmap = np.clip(heatmap, 0.0, 1.0)

    elif mode == "gaussian":
        # Dilate first, then smooth
        if thickness > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (thickness, thickness)
            )
            heatmap = cv2.dilate(heatmap, kernel)
        ksize = int(2 * round(3 * sigma) + 1)  # 3-sigma rule
        heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), sigma)
        # Normalize to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

    return heatmap


def generate_boundary_targets_batch(
    label_dir: str | Path,
    output_dir: str | Path,
    img_size: int = 512,
    thickness: int = 3,
    mode: str = "binary",
    sigma: float = 1.5,
) -> int:
    """Generate boundary heatmaps for all label files in a directory.

    Parameters
    ----------
    label_dir : path-like
        Directory containing YOLO label .txt files.
    output_dir : path-like
        Directory to write .npy heatmap files.
    img_size : int
        Square image size.
    thickness : int
        Band thickness.
    mode : str
        ``"binary"`` or ``"gaussian"``.
    sigma : float
        Gaussian sigma.

    Returns
    -------
    int
        Number of heatmaps generated.
    """
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for label_file in sorted(label_dir.glob("*.txt")):
        labels = np.zeros((0, 5), dtype=np.float32)
        raw = np.loadtxt(str(label_file), ndmin=2)
        if raw.size > 0:
            labels = raw.astype(np.float32)

        heatmap = generate_boundary_heatmap(
            labels, img_size, thickness=thickness, mode=mode, sigma=sigma
        )

        out_path = output_dir / (label_file.stem + ".npy")
        np.save(str(out_path), heatmap)
        count += 1

    print(f"Generated {count} boundary heatmaps in {output_dir}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate boundary heatmap proxy targets"
    )
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/boundary_maps")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--mode", type=str, choices=["binary", "gaussian"], default="binary")
    parser.add_argument("--sigma", type=float, default=1.5)
    args = parser.parse_args()

    generate_boundary_targets_batch(
        args.label_dir, args.output_dir, args.img_size,
        args.thickness, args.mode, args.sigma,
    )


if __name__ == "__main__":
    main()
