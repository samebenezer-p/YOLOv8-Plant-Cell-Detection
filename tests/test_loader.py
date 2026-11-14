"""
Tests for model loading, unpickling patches, E8 forward pass re-binding, and metadata extraction.
"""

from pathlib import Path
import pytest
import torch
from ultralytics import YOLO

from plantcell_tool.models import load_model, model_info, DOCUMENTED_THRESHOLDS


def test_model_info_retrieval():
    """Verify model metadata, parameter counts, and documented F1 thresholds."""
    info_e5a = model_info("E5a")
    assert info_e5a["name"] == "E5a"
    assert info_e5a["best_f1_conf"] == 0.063
    assert info_e5a["tile_map50"] == 0.6324
    assert len(info_e5a["sha256"]) == 64

    info_e3 = model_info("E3")
    assert info_e3["name"] == "E3"
    assert info_e3["best_f1_conf"] == 0.114

    info_e1 = model_info("E1")
    assert info_e1["name"] == "E1"
    assert info_e1["best_f1_conf"] == 0.108

    info_e8 = model_info("E8")
    assert info_e8["best_f1_conf"] is None


def test_e8_loader_rebound_predict_once():
    """Verify that E8 loads with _predict_once re-bound to the BiFPN closure."""
    model, name = load_model("E8", device="cpu")
    assert name == "E8"

    m = model.model
    # Must have fusion_neck
    assert hasattr(m, "fusion_neck")
    # Verify that the rebound function code originates from fusion_neck.py
    co_file = getattr(m._predict_once.__func__.__code__, "co_filename", "")
    assert "fusion_neck" in co_file, f"Expected _predict_once to originate from fusion_neck.py, got {co_file}"


def test_plain_loaded_e8_detected_as_broken():
    """Verify that loading E8 via raw YOLO() bypasses BiFPN feature fusion."""
    from plantcell_tool.models import resolve_checkpoint_path, _patch_main_namespace
    _patch_main_namespace()

    _, ckpt_path = resolve_checkpoint_path("E8")
    raw_yolo = YOLO(str(ckpt_path))

    # Without re-binding, _predict_once points to Ultralytics BaseModel/DetectionModel
    co_file = getattr(raw_yolo.model._predict_once.__func__.__code__, "co_filename", "")
    is_plain = "fusion_neck" not in co_file
    assert is_plain, "Plain loaded model unexpectedly had fusion_neck rebound!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available on this machine")
def test_gpu_loading_and_half_precision():
    """Verify GPU loading and half precision support when CUDA is available."""
    model, _ = load_model("E5a", device="cuda", half=True)
    device = next(model.model.parameters()).device
    assert device.type == "cuda"
