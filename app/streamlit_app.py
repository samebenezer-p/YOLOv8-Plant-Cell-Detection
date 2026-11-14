"""
Streamlit Web Application for High-Resolution Plant Cell Detection.

Provides an interactive GUI for uploading micrographs (.tif, .png, .jpg),
running tiled inference with custom YOLOv8 models, visualizing detections,
inspecting cell size distributions, and downloading results.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plantcell_tool import LIMITATIONS_NOTICE
from plantcell_tool.export import export_csv, export_json, export_overlay
from plantcell_tool.imageio import read_image
from plantcell_tool.infer import DetectionResult, detect_full_image
from plantcell_tool.models import DOCUMENTED_THRESHOLDS, load_model, model_info


@st.cache_resource
def get_model(model_name: str, device: str, half: bool):
    """Cached loader for YOLO models keyed by model name, device, and half precision."""
    return load_model(model_name, device=device, half=half)


def main():
    st.set_page_config(
        page_title="Plant Cell Detection Tool",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("🔬 Plant Cell Detection & Micrograph Analysis")
    st.caption("High-Resolution Tiled Inference with YOLOv8 BiFPN / Shape-IoU Architecture")

    # Persistent Limitations Notice
    st.warning(f"⚠️ **Limitations Notice**: {LIMITATIONS_NOTICE}")

    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")

    model_options = [
        "E5a (Default: Top mAP50 / F1)",
        "E3 (Fastest / Smallest - 2.02M params)",
        "E1 (Baseline YOLOv8n)",
        "E8 (Full Composite: BiFPN+Boundary+MPD)",
        "E2 (Ablation: Shape-IoU)",
        "E4 (Ablation: Boundary Head)",
        "E5b (Ablation: MPDIoU)",
    ]
    model_choice = st.sidebar.selectbox("Select Model", model_options, index=0)
    canon_name = model_choice.split()[0].strip()

    # Determine default confidence threshold
    default_conf = DOCUMENTED_THRESHOLDS.get(canon_name, 0.100)
    is_documented = canon_name in DOCUMENTED_THRESHOLDS

    st.sidebar.markdown(
        f"**Documented Best-F1 Threshold**: `{default_conf:.3f}`"
        if is_documented
        else "*No documented threshold for ablation variant*"
    )

    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=0.90,
        value=float(default_conf),
        step=0.005,
        help="Detection confidence cutoff. Initialized to model's documented max-F1 threshold.",
    )

    iou_thresh = st.sidebar.slider(
        "Cross-Tile NMS IoU",
        min_value=0.10,
        max_value=0.80,
        value=0.45,
        step=0.05,
        help="Intersection-over-Union threshold for merging overlapping tile detections.",
    )

    overlap_frac = st.sidebar.slider(
        "Tile Overlap",
        min_value=0.0,
        max_value=0.50,
        value=0.25,
        step=0.05,
        help="Fractional overlap between neighboring tiles (0.25 matches training).",
    )

    tile_size = st.sidebar.selectbox("Tile Size (px)", [512, 640], index=1)
    batch_size = st.sidebar.select_slider("Batch Size", options=[4, 8, 16, 32], value=16)

    # Compute device & FP16
    cuda_available = torch.cuda.is_available()
    device_options = ["auto", "cuda", "cpu"] if cuda_available else ["cpu"]
    device_choice = st.sidebar.selectbox(
        "Device", device_options, index=0,
        help="Hardware acceleration. Automatically selects CUDA GPU if available."
    )

    fp16_enabled = st.sidebar.checkbox(
        "FP16 Half Precision",
        value=False,
        disabled=not cuda_available,
        help="Enable FP16 for faster GPU inference and lower VRAM usage.",
    )

    drop_borders = st.sidebar.checkbox(
        "Drop Border Boxes",
        value=False,
        help="Discard boxes within 3px of interior tile edges before NMS.",
    )

    # Display Model Details in Sidebar
    with st.sidebar.expander("ℹ️ Model Architecture & Checkpoint"):
        try:
            info = model_info(canon_name)
            st.markdown(f"- **Params**: `{info.get('params', 'N/A'):,}`")
            st.markdown(f"- **GFLOPs**: `{info.get('gflops', 'N/A')}`")
            st.markdown(f"- **Tile mAP50**: `{info.get('tile_map50', 'N/A')}`")
            st.markdown(f"- **File Size**: `{info.get('size_mb', 'N/A')} MB`")
            st.caption(f"SHA256: `{info.get('sha256', '')[:16]}...`")
        except Exception as e:
            st.caption(f"Metadata unavailable: {e}")

    # Main Area: File Upload
    st.subheader("1. Upload Micrograph")
    uploaded_file = st.file_uploader(
        "Upload high-resolution plant cell image (.tif, .tiff, .png, .jpg)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        suffix = Path(uploaded_file.name).suffix.lower()

        # Save to temporary file for robust image reading
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = Path(tmp_file.name)

        try:
            with st.spinner("Decoding image..."):
                rgb_img, img_meta = read_image(tmp_path)

            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
            col_meta1.metric("Dimensions", f"{img_meta['width']} × {img_meta['height']} px")
            col_meta2.metric("Original Dtype", img_meta["orig_dtype"])
            col_meta3.metric("16-Bit Scaled", "Yes (1/257)" if img_meta["scaled_16bit"] else "No")
            col_meta4.metric("Format / Loader", f"{suffix.upper()} ({img_meta['read_method']})")

            # Run Detection Button
            st.subheader("2. Run Inference")
            if st.button("🚀 Detect Cells", type="primary", use_container_width=True):
                with st.spinner(f"Loading {canon_name} and running tiled inference..."):
                    # Load model
                    model, resolved_name = get_model(
                        canon_name,
                        device=device_choice,
                        half=fp16_enabled,
                    )

                    # Run inference
                    t0 = time.perf_counter()
                    result = detect_full_image(
                        model=model,
                        image=rgb_img,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        tile=tile_size,
                        overlap=overlap_frac,
                        batch=batch_size,
                        max_det=300,
                        half=fp16_enabled,
                        drop_border_boxes=drop_borders,
                        model_name=resolved_name,
                    )
                    total_dur = time.perf_counter() - t0
                    result.timing["total"] = round(total_dur, 4)

                # Store result in session state
                st.session_state["result"] = result
                st.session_state["rgb_img"] = rgb_img
                st.session_state["img_meta"] = img_meta
                st.session_state["file_stem"] = Path(uploaded_file.name).stem

        except Exception as e:
            st.error(f"❌ Failed to process image: {str(e)}")
            with st.expander("🔍 View Error Stacktrace"):
                st.code(traceback.format_exc())
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # Results Display
    if "result" in st.session_state:
        result: DetectionResult = st.session_state["result"]
        rgb_img: np.ndarray = st.session_state["rgb_img"]
        img_meta: dict = st.session_state["img_meta"]
        stem: str = st.session_state["file_stem"]

        st.subheader("3. Detection Results")

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cell Count", f"{result.count:,}")
        m2.metric("Total Latency", f"{result.timing.get('total', 0.0):.2f} s")
        m3.metric("Tiles Inferred", f"{result.n_tiles} tiles")
        m4.metric("Device", result.device.upper())
        m5.metric("Precision", "FP16" if result.half else "FP32")

        # Create overlay in memory for display and download
        with st.spinner("Rendering visualization..."):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
                tmp_out_path = Path(tmp_out.name)
            try:
                export_overlay(rgb_img, result, tmp_out_path)
                with open(tmp_out_path, "rb") as f:
                    overlay_png_bytes = f.read()
            finally:
                if tmp_out_path.exists():
                    tmp_out_path.unlink()

        # Display downscaled image for fluid web rendering
        st.markdown("#### Annotated Micrograph Overlay")
        st.image(
            overlay_png_bytes,
            caption=f"Full Micrograph ({img_meta['width']}×{img_meta['height']} px) - {result.count} cells detected",
            use_container_width=True,
        )

        # Size Distribution Histogram
        if result.count > 0:
            st.markdown("#### Cell Size Distribution (Equivalent Diameter $\\sqrt{W \\times H}$)")
            widths = result.boxes_xyxy[:, 2] - result.boxes_xyxy[:, 0]
            heights = result.boxes_xyxy[:, 3] - result.boxes_xyxy[:, 1]
            sqrt_wh = np.sqrt(np.maximum(widths * heights, 0.0))

            median_size = float(np.median(sqrt_wh))
            mean_size = float(np.mean(sqrt_wh))

            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
            ax.hist(sqrt_wh, bins=40, color="#10b981", edgecolor="#065f46", alpha=0.85)
            ax.axvline(median_size, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Median: {median_size:.1f} px")
            ax.axvline(mean_size, color="#3b82f6", linestyle=":", linewidth=1.5, label=f"Mean: {mean_size:.1f} px")
            ax.set_xlabel("Equivalent Diameter $\\sqrt{wh}$ (pixels)")
            ax.set_ylabel("Count")
            ax.set_title("Plant Cell Size Distribution", fontsize=11, fontweight="bold")
            ax.legend(frameon=True)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        # Download Buttons
        st.subheader("4. Export Results")
        dcol1, dcol2, dcol3 = st.columns(3)

        # CSV bytes
        csv_buf = io.StringIO()
        csv_buf.write("x1,y1,x2,y2,score,width,height\n")
        for box, score in zip(result.boxes_xyxy, result.scores):
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            csv_buf.write(f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{score:.4f},{w:.2f},{h:.2f}\n")

        # JSON bytes
        json_meta = result.to_dict()
        json_meta["image"] = img_meta
        json_str = json.dumps(json_meta, indent=2)

        dcol1.download_button(
            label="📥 Download Full-Res Overlay PNG",
            data=overlay_png_bytes,
            file_name=f"{stem}_overlay.png",
            mime="image/png",
            use_container_width=True,
        )

        dcol2.download_button(
            label="📊 Download Detections CSV",
            data=csv_buf.getvalue(),
            file_name=f"{stem}_detections.csv",
            mime="text/csv",
            use_container_width=True,
        )

        dcol3.download_button(
            label="📄 Download Metadata JSON",
            data=json_str,
            file_name=f"{stem}_meta.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
