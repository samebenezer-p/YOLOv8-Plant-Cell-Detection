"""
Plant Cell Detection Inference Tool
A high-throughput, tiled inference package for plant-cell detection on high-resolution micrographs.
"""

from .models import load_model, model_info, DOCUMENTED_THRESHOLDS
from .imageio import read_image
from .infer import detect_full_image, DetectionResult
from .export import export_overlay, export_csv, export_yolo, export_json

LIMITATIONS_NOTICE = (
    "Trained on partially annotated micrographs (annotations cover ~20% of image area; "
    "56% of training tiles have no labelled cell). Cell counts on full micrographs have "
    "NOT been validated and may under-count in regions that resemble unannotated areas. "
    "Metrics in the paper are tile-level on 2 held-out specimens. Species identity of the "
    "dataset is unverified. Research use only."
)

__version__ = "0.1.0"

__all__ = [
    "load_model",
    "model_info",
    "DOCUMENTED_THRESHOLDS",
    "read_image",
    "detect_full_image",
    "DetectionResult",
    "export_overlay",
    "export_csv",
    "export_yolo",
    "export_json",
    "LIMITATIONS_NOTICE",
]
