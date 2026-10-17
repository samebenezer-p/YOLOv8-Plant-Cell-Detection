"""
E6: Parenchyma context refinement classifier.

A lightweight classifier trained on ground-truth parenchyma crops + margin
to refine detector predictions for the parenchyma class. At inference,
detector-predicted parenchyma candidates are cropped with a configurable
margin and passed through this classifier to confirm or reject the
detection.

Key design choices:
- **Training**: Uses GT crops + margin (not detector crops) to avoid
  compounding detector localization errors during classifier training.
- **Inference**: Applied only to parenchyma-candidate detections.
- **Tracking**: Separately tracks refinement-introduced false positives
  vs. detector false positives.
- **Latency**: Measures added latency per parenchyma candidate only.

Implementation approach for E6 ablation:
  Since this is a *post-hoc* classifier that refines the detector output,
  we first train a standard YOLOv8 baseline, then train a small CNN on
  GT parenchyma crops, and finally apply it during evaluation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextRefineClassifier(nn.Module):
    """Lightweight classifier for parenchyma confirmation.

    Takes a crop (with surrounding context margin) of a parenchyma
    candidate detection and predicts whether it is truly parenchyma
    or a false positive.

    Parameters
    ----------
    input_size : int
        Square input crop size after resizing.
    margin_ratio : float
        How much margin (fraction of box size) to add around the crop.
    """

    def __init__(
        self,
        input_size: int = 64,
        margin_ratio: float = 0.25,
    ):
        super().__init__()
        self.input_size = input_size
        self.margin_ratio = margin_ratio

        # Simple lightweight CNN (no heavy backbone needed for binary classification)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """Classify parenchyma crops.

        Parameters
        ----------
        crops : torch.Tensor
            Batch of crops, shape (B, 3, input_size, input_size).

        Returns
        -------
        torch.Tensor
            Sigmoid probabilities, shape (B, 1).
        """
        x = self.features(crops)
        return torch.sigmoid(self.classifier(x))


def extract_crops_from_gt(
    image: np.ndarray,
    labels: np.ndarray,
    target_class: int = 2,
    margin_ratio: float = 0.25,
    crop_size: int = 64,
) -> Tuple[List[np.ndarray], List[int]]:
    """Extract crops with margin from ground-truth boxes for training.

    Parameters
    ----------
    image : np.ndarray
        Full image, shape (H, W, 3).
    labels : np.ndarray
        YOLO labels, shape (N, 5) — [class, cx, cy, w, h] normalized.
    target_class : int
        Class ID to extract (2 = parenchyma).
    margin_ratio : float
        Fraction of box size to add as context margin.
    crop_size : int
        Output crop size after resizing.

    Returns
    -------
    crops : list of np.ndarray
        List of (crop_size, crop_size, 3) crops.
    class_labels : list of int
        1 for target_class, 0 for negative class.
    """
    h, w = image.shape[:2]
    crops = []
    class_labels = []

    for row in labels:
        cls_id = int(row[0])
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]

        # Add margin
        margin_w = bw * margin_ratio
        margin_h = bh * margin_ratio

        x1 = int(max(0, (cx - bw / 2 - margin_w) * w))
        y1 = int(max(0, (cy - bh / 2 - margin_h) * h))
        x2 = int(min(w, (cx + bw / 2 + margin_w) * w))
        y2 = int(min(h, (cy + bh / 2 + margin_h) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (crop_size, crop_size))
        crops.append(crop)
        class_labels.append(1 if cls_id == target_class else 0)

    return crops, class_labels


def extract_crops_from_detections(
    image: np.ndarray,
    detections: List[Dict],
    target_class: int = 2,
    margin_ratio: float = 0.25,
    crop_size: int = 64,
) -> Tuple[List[np.ndarray], List[int]]:
    """Extract crops from detector-predicted parenchyma candidates.

    Parameters
    ----------
    image : np.ndarray
        Full image.
    detections : list of dict
        Detector output with ``"bbox"`` (xyxy), ``"class_id"``, ``"confidence"`` keys.
    target_class : int
        Class ID to filter for refinement.
    margin_ratio : float
        Context margin fraction.
    crop_size : int
        Output crop size.

    Returns
    -------
    crops : list of np.ndarray
        Extracted crops.
    indices : list of int
        Indices into the detections list for tracking.
    """
    h, w = image.shape[:2]
    crops = []
    indices = []

    for i, det in enumerate(detections):
        if det["class_id"] != target_class:
            continue

        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        margin_w = bw * margin_ratio
        margin_h = bh * margin_ratio

        cx1 = int(max(0, x1 - margin_w))
        cy1 = int(max(0, y1 - margin_h))
        cx2 = int(min(w, x2 + margin_w))
        cy2 = int(min(h, y2 + margin_h))

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = image[cy1:cy2, cx1:cx2]
        crop = cv2.resize(crop, (crop_size, crop_size))
        crops.append(crop)
        indices.append(i)

    return crops, indices


def measure_refinement_latency(
    classifier: ContextRefineClassifier,
    num_candidates: int = 100,
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """Measure per-candidate inference latency of the refinement classifier.

    Returns
    -------
    dict
        ``{"mean_ms_per_candidate": float, "std_ms": float}``.
    """
    classifier = classifier.to(device).eval()
    dummy = torch.randn(num_candidates, 3, classifier.input_size, classifier.input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            classifier(dummy)

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            classifier(dummy)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed / num_candidates)

    arr = np.array(times)
    return {"mean_ms_per_candidate": float(arr.mean()), "std_ms": float(arr.std())}
