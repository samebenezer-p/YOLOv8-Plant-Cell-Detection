"""
E5: Swappable bounding-box regression loss functions.

Implements two IoU-based losses as drop-in replacements for the default
CIoU loss in YOLOv8:

- **Shape IoU**: Focuses on shape similarity by comparing width/height
  ratios independent of absolute scale.
- **MPDIoU** (Minimum Point Distance IoU): Uses minimum point distance
  between box corners for more precise localization.

Integration strategy: monkey-patch ``ultralytics.utils.metrics.bbox_iou``
at runtime so that when YOLOv8 requests CIoU, our custom loss is computed
instead. This is non-destructive, scoped to the E5 training runs only.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


def shape_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    scale: float = 0.0,
    eps: float = 1e-7,
    **kwargs,
) -> torch.Tensor:
    """Shape IoU — penalises width/height ratio differences independently of scale.

    Ref: Zhang et al., "Shape-IoU", 2024.

    Accepts the same ``(box1, box2, xywh, ...)`` signature as
    ``ultralytics.utils.metrics.bbox_iou`` so it can be used as a
    monkey-patch replacement.
    """
    # --- Coordinate conversion ---
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # --- Intersection & Union ---
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # --- Enclosing box ---
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    # --- Center distance penalty (same as DIoU) ---
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4

    # --- Shape penalty ---
    # Width ratio penalty
    ww = 2 * (w1 * w2).pow(0.5) / (w1 + w2 + eps)
    # Height ratio penalty
    hh = 2 * (h1 * h2).pow(0.5) / (h1 + h2 + eps)

    # Scale factor (auto if scale == 0)
    if scale == 0.0:
        _scale = 1 - torch.exp(-1.0 * (w2 * h2 / (cw * ch + eps)))
    else:
        _scale = scale

    omiga_w = (1 - ww) * _scale
    omiga_h = (1 - hh) * _scale

    shape_cost = omiga_w + omiga_h

    return iou - (rho2 / c2 + shape_cost)


def mpdiou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    eps: float = 1e-7,
    **kwargs,
) -> torch.Tensor:
    """MPDIoU — Minimum Point Distance IoU.

    Ref: Ma et al., "MPDIoU", 2023.

    Uses the squared L2 distance between the top-left and bottom-right
    corners, normalised by the squared diagonal of the enclosing box.
    """
    # --- Coordinate conversion ---
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # --- Intersection & Union ---
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # --- Enclosing box diagonal squared ---
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    d2 = cw.pow(2) + ch.pow(2) + eps

    # --- Point distances ---
    # Top-left corner distance squared
    d1_sq = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)
    # Bottom-right corner distance squared
    d2_sq = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)

    return iou - (d1_sq + d2_sq) / d2


# ---------------------------------------------------------------------------
# Loss registry for config-driven selection
# ---------------------------------------------------------------------------

LOSS_REGISTRY = {
    "shape_iou": shape_iou,
    "mpdiou": mpdiou,
}


def get_iou_loss(name: str):
    """Look up an IoU loss function by name.

    Parameters
    ----------
    name : str
        One of ``"shape_iou"``, ``"mpdiou"``.

    Returns
    -------
    callable
        The corresponding loss function.
    """
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown IoU loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name]


@contextmanager
def patched_bbox_iou(custom_iou_fn):
    """Context manager that monkey-patches ``ultralytics.utils.metrics.bbox_iou``.

    Inside the context, any call to ``bbox_iou(box1, box2, ..., CIoU=True)``
    will be routed to ``custom_iou_fn(box1, box2, ...)`` instead. All other
    call modes (plain IoU, GIoU, DIoU) fall through to the original.
    """
    import ultralytics.utils.metrics as _metrics
    # Also patch in the loss module where it may be imported directly
    import ultralytics.utils.loss as _loss_module

    original_fn = _metrics.bbox_iou
    # Check if loss module has its own reference
    loss_has_own = hasattr(_loss_module, 'bbox_iou')
    original_loss_fn = getattr(_loss_module, 'bbox_iou', None) if loss_has_own else None

    def patched(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        if CIoU:
            return custom_iou_fn(box1, box2, xywh=xywh, eps=eps)
        return original_fn(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)

    _metrics.bbox_iou = patched
    if loss_has_own:
        _loss_module.bbox_iou = patched

    try:
        yield
    finally:
        _metrics.bbox_iou = original_fn
        if loss_has_own and original_loss_fn is not None:
            _loss_module.bbox_iou = original_loss_fn
