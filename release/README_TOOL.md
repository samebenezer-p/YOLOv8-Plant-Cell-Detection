# PlantCell Tool: High-Resolution Plant Cell Detection & Inference Pipeline

A lightweight, reliable, high-throughput inference tool for single-class plant cell detection on high-resolution microscopic sequences ($3840 \times 2160$ TIFF / PNG / JPG). Built on top of a frozen YOLOv8 architectural ablation research suite.

---

> [!WARNING]
> ### LIMITATIONS NOTICE
> "Trained on partially annotated micrographs (annotations cover ~20% of image area; 56% of training tiles have no labelled cell). Cell counts on full micrographs have NOT been validated and may under-count in regions that resemble unannotated areas. Metrics in the paper are tile-level on 2 held-out specimens. Species identity of the dataset is unverified. Research use only."

---

## 1. System Requirements & Installation

- **Operating System**: Windows 11 / Windows 10, or Linux (Ubuntu 20.04+)
- **Python**: 3.12 (tested on 3.12.10)
- **Deep Learning Framework**: PyTorch $\ge 2.6.0$, Ultralytics $\ge 8.4.182$
- **Hardware**: NVIDIA GPU with $\ge 4$ GB VRAM (tested on RTX 3050 Laptop GPU, CUDA 12.4), or CPU.

### Installation Steps

1. Activate your Python environment:
   ```powershell
   # Windows PowerShell
   .\.venv_gpu\Scripts\Activate.ps1
   ```
2. Install base frozen dependencies and application dependencies:
   ```powershell
   .\.venv_gpu\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv_gpu\Scripts\python.exe -m pip install -r requirements-app.txt
   ```
   *Note: `requirements-app.txt` strictly pins `streamlit==1.66.1`, `pytest==9.1.1`, and `tifffile==2026.10.15` without upgrading torch, torchvision, or numpy.*

---

## 2. Command Line Interface (CLI)

The package provides a clean command-line interface executable via `python -m plantcell_tool`.

### Basic Prediction Command

Run tiled cell detection on a full micrograph with automatic optimal thresholding and export overlays and CSV:

```powershell
.\.venv_gpu\Scripts\python.exe -m plantcell_tool predict `
    --input data/real_raw/plant_8/img/B6_0808.tif `
    --model E5a `
    --conf auto `
    --save-overlay `
    --save-csv `
    --output-dir outputs_tool/run1
```

### Batch Directory Prediction with High Throughput

Run batch inference across an entire directory of micrographs:

```powershell
.\.venv_gpu\Scripts\python.exe -m plantcell_tool predict `
    --input data/real_raw/plant_16/img `
    --model E3 `
    --conf auto `
    --batch 16 `
    --save-overlay `
    --save-csv `
    --save-yolo `
    --output-dir outputs_tool/plant16_predictions
```

### Inspecting Model Metadata & Thresholds

```powershell
.\.venv_gpu\Scripts\python.exe -m plantcell_tool info E5a
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input` | `str` | *Required* | Path to input image file or directory of images (`.tif`, `.tiff`, `.png`, `.jpg`). |
| `--model` | `str` | `E5a` | Checkpoint name (`E5a`, `E3`, `E1`, `E8`, `E2`, `E4`, `E5b`) or path to `.pt` file. |
| `--conf` | `str` | `auto` | Confidence threshold. `'auto'` selects the model's documented max-F1 threshold. |
| `--output-dir` | `str` | `outputs_tool/run1` | Output destination directory. |
| `--iou` | `float` | `0.45` | IoU threshold for cross-tile NMS merging. |
| `--tile` | `int` | `640` | Sliding window tile dimension in pixels. |
| `--overlap` | `float` | `0.25` | Overlap fraction between neighboring tiles (step size = $640 \times 0.75 = 480$ px). |
| `--batch` | `int` | `16` | Batch size for tile evaluation on GPU. |
| `--device` | `str` | `auto` | Compute device (`'auto'`, `'cuda'`, `'cpu'`). |
| `--half` | `flag` | `False` | Enables FP16 half-precision inference on GPU. |
| `--drop-border-boxes` | `flag` | `False` | Discards boxes within 3px of interior tile edges prior to NMS merging. |
| `--save-overlay` | `flag` | `False` | Exports annotated micrograph PNG with bounding boxes and header badge. |
| `--save-csv` | `flag` | `False` | Exports detections CSV (`x1,y1,x2,y2,score,width,height` in pixels). |
| `--save-yolo` | `flag` | `False` | Exports normalized YOLO-format `.txt` labels (`class_id cx cy w h`). |

---

## 3. Interactive Web Application (Streamlit)

Launch the interactive graphical interface:

```powershell
.\.venv_gpu\Scripts\streamlit.exe run app/streamlit_app.py
```

### Features
- **Sidebar Configuration**: Model selector, real-time confidence slider (pre-set to documented max-F1 threshold), IoU threshold slider, tile overlap, batch size, and FP16 toggle.
- **Fluid Micrograph Viewer**: Automatic downscaling for responsive web display while maintaining full native resolution ($3840 \times 2160$) for exports.
- **Statistical Size Distribution**: Interactive histogram displaying equivalent cell diameter $\sqrt{W \times H}$ in pixels, with mean and median indicators.
- **One-Click Exports**: Dedicated download buttons for full-resolution annotated PNG overlay, pixel coordinate CSV, and metadata JSON.
- **Resource Caching**: Models are cached in memory via `st.cache_resource` for instant inference.

---

## 4. Model Architecture & Selection Guide

| Model | Architecture / Head | Parameters | GFLOPs | Test Tile mAP50 | Best-F1 Conf | Recommended Role |
|---|---|---|---|---|---|---|
| **E5a** | **YOLOv8n + Shape-IoU** | **3.01M** | **8.19** | **0.6324** | **0.063** | **Default Recommended**: Highest test mAP50 and highest max-F1 (0.6244). |
| **E3** | **P2BiFPN / FusionNeck** | **2.02M** | **7.28** | **0.6210** | **0.114** | **Fastest / Smallest**: 33% fewer parameters, 11% fewer GFLOPs, highest throughput. |
| **E1** | YOLOv8n Baseline (CIoU) | 3.01M | 8.19 | 0.4846 | 0.108 | Baseline comparison model. |
| **E2** | YOLOv8n + Shape-IoU (Ablation) | 3.01M | 8.19 | 0.5283 | *None* | Architectural ablation study variant. |
| **E4** | YOLOv8n + Aux BoundaryHead | 3.04M | 8.24 | 0.5055 | *None* | Architectural ablation study variant. |
| **E5b** | YOLOv8n + MPDIoU | 3.01M | 8.19 | 0.5367 | *None* | Architectural ablation study variant. |
| **E8** | P2BiFPN + BoundaryHead + MPD | 3.06M | 9.87 | 0.6043 | *None* | Full composite ablation. Requires custom `_predict_once` rebinding; not recommended as default. |

> [!NOTE]
> **Why E5a is the Recommended Default**:
> In the held-out test split (160 tiles across plant_8 and plant_16), E5a achieved the highest detection accuracy (mAP50 = 0.6324, Max-F1 = 0.6244 at confidence threshold 0.063). For latency-constrained edge deployments, E3 is recommended for its reduced footprint (2.02M params, 7.28 GFLOPs) while maintaining competitive accuracy (mAP50 = 0.6210). E8 is an experimental composite model and is not recommended as default.

---

## 5. Checkpoint Acquisition & Verification

Model weights are binary artifacts and are excluded from version control via `.gitignore`. Checkpoints are stored in `runs/detect/results/real/<MODEL>/weights/best.pt`. Verify checkpoints against the recorded SHA256 checksums:

| Model | Relative Path | Size (Bytes) | Size (MB) | SHA256 Checksum |
|---|---|---|---|---|
| **E1** | `runs/detect/results/real/E1/weights/best.pt` | 6,296,887 | 6.01 MB | `42e874eeac64091e0a0e42cf39957dd59a98c4380990cb9826d2c41254f3b06f` |
| **E2** | `runs/detect/results/real/E2/weights/best.pt` | 6,341,451 | 6.05 MB | `a41e2e3402ef5b0200977246055dafe3fd3dce1301e59ea7163a90609a7e3024` |
| **E3** | `runs/detect/results/real/E3/weights/best.pt` | 4,289,232 | 4.09 MB | `3406db27ceeecdbfcf13df48c2999f5c3110ef143083d1cea7a8a41e7c6f32e0` |
| **E4** | `runs/detect/results/real/E4/weights/best.pt` | 6,338,791 | 6.05 MB | `2b5845d169101e485ce222676bb1acb6688207853f5de03063c9c1744170eef0` |
| **E5a** | `runs/detect/results/real/E5a/weights/best.pt` | 6,296,887 | 6.01 MB | `56aaa35a7192b5bbe80736738a9272817f040f0959e6662fb88bb9c1e8808b39` |
| **E5b** | `runs/detect/results/real/E5b/weights/best.pt` | 6,295,351 | 6.00 MB | `3ef24bec84aeb02a82f3e0cbf95934af3c81735c7fb77e837239a08f2836c12c` |
| **E8** | `runs/detect/results/real/E8/weights/best.pt` | 6,659,261 | 6.35 MB | `d059b2b0e18a7e2ab25c8f82ca534a734f77cfe85be996076d30b29878ed87b8` |

---

## 6. Citation & Research Paper

Paper: *<to be added>*

```bibtex
@inproceedings{ebenezer2026plantcell,
  author    = {Sam Ebenezer .P and M. Denisha},
  title     = {High-Resolution YOLOv8-Based Plant Cell Detection and Architectural Ablation on Microscopic Sequences},
  booktitle = {IEEE Conference on Computer Vision and Bio-Imaging (Under Review)},
  year      = {2026}
}
```
