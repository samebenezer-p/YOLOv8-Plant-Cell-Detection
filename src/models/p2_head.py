"""
E2: P2 (stride-4) detection head for improved small-object detection.

Leverages the built-in Ultralytics ``yolov8-p2.yaml`` architecture which adds
a fourth detection path at the P2 feature level (stride 4), enabling
finer-grained detection of small xylem cells (fibers, small parenchyma).

The P2 architecture adds:
  - An additional upsampling path from P3 -> P2 in the neck
  - Concatenation with the P2 backbone features
  - A fourth Detect head: Detect(P2, P3, P4, P5) instead of Detect(P3, P4, P5)

Usage:
    from src.models.p2_head import build_p2_model
    model = build_p2_model(num_classes=3, scale="n")  # yolov8n-p2
"""

from __future__ import annotations

from typing import Optional

from ultralytics import YOLO


def build_p2_model(
    num_classes: int = 3,
    scale: str = "n",
    pretrained_weights: Optional[str] = None,
) -> YOLO:
    """Build a YOLOv8-P2 model with a 4-scale detection head.

    Uses the Ultralytics built-in ``yolov8-p2.yaml`` architecture definition
    which includes detection at P2 (stride 4), P3 (stride 8), P4 (stride 16),
    and P5 (stride 32). The ``scale`` parameter selects the model size
    (n/s/m/l/x).

    Parameters
    ----------
    num_classes : int
        Number of detection classes. Default 3 (vessel/fiber/parenchyma).
    scale : str
        Model scale: ``"n"`` (nano), ``"s"`` (small), etc. For 4GB VRAM,
        use ``"n"`` or ``"s"`` only.
    pretrained_weights : str, optional
        Path to pretrained weights to load for transfer learning. If None,
        uses the corresponding pretrained checkpoint (e.g. ``yolov8n.pt``).

    Returns
    -------
    YOLO
        Ultralytics YOLO model instance with P2 head.
    """
    # Ultralytics resolves "yolov8n-p2.yaml" to the built-in architecture
    yaml_name = f"yolov8{scale}-p2.yaml"

    # Initialize model from the P2 architecture YAML
    model = YOLO(yaml_name)

    # Optionally load pretrained backbone weights for transfer learning
    # Note: This loads compatible layers from the standard model, ignoring
    # the new P2 neck/head layers which will be randomly initialized
    if pretrained_weights:
        print(f"  Loading pretrained weights from: {pretrained_weights}")
        # The YOLO.load() method loads weights with strict=False,
        # allowing mismatched head layers
        model.load(pretrained_weights)
    else:
        # Default: load the standard pretrained weights for transfer learning
        default_weights = f"yolov8{scale}.pt"
        print(f"  Loading default pretrained weights: {default_weights}")
        try:
            model.load(default_weights)
        except Exception as e:
            print(f"  Warning: Could not load pretrained weights ({e}). "
                  f"Training from scratch.")

    return model


def get_p2_model_info() -> dict:
    """Return metadata about the P2 architecture for documentation."""
    return {
        "architecture": "yolov8-p2",
        "detection_scales": ["P2/4", "P3/8", "P4/16", "P5/32"],
        "num_heads": 4,
        "description": (
            "YOLOv8 with an additional P2 (stride-4) detection head for "
            "improved small-object sensitivity. Adds ~20-30% more parameters "
            "and GFLOPs compared to the standard 3-head architecture."
        ),
    }
