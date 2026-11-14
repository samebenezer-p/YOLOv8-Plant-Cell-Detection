# PROJECT DISCOVERY REPORT: YOLOv8 Plant-Cell Detection and Architectural Ablation

**Audit Mode**: READ-ONLY Discovery Audit  
**Status**: Experimental implementation complete and FROZEN  
**Working Directory**: `e:/Yolov8_Plant_Cell_Detection`  
**Generated At**: 2026-10-21  

---

## 1. REPOSITORY MAP

### 1.1 Directory Tree (Depth 3)
*(Excluding `.venv*`, `venv/`, `__pycache__/`, `.git/`, and `runs/*/weights` internal checkpoints)*

```text
e:/Yolov8_Plant_Cell_Detection/
|-- .gitignore
|-- README.md
|-- README.txt
|-- benchmark_gpu_rigorous.py
|-- benchmark_gpu_throughput.py
|-- data.yaml
|-- demo_inference.py
|-- download_and_install_gpu.py
|-- install_gpu_env.py
|-- prepare_real_data.py
|-- requirements.txt
|-- run_full_sweep.py
|-- scratch_intercept.py
|-- scratch_test_neck.py
|-- verify_gpu_e1.py
|-- yolo26n.pt
|-- yolov8n.pt
|-- configs/
|   |-- e8_full_model.yaml
|   |-- yolov8n_nc1.yaml
|   |-- yolov8n_nc3.yaml
|   |-- yolov8n_p2_nc1.yaml
|   `-- yolov8n_p2_nc3.yaml
|-- data/
|   |-- boundary_maps/
|   |-- real/
|   |   |-- data.yaml
|   |   |-- images/ (train, val, test)
|   |   `-- labels/ (train, val, test)
|   |-- real_full/
|   |   `-- labels/
|   |-- real_raw/
|   |   |-- README.md
|   |   `-- plant_1/ .. plant_16/ (img/, json/)
|   |-- splits/
|   |   |-- test_specimens.txt
|   |   |-- train_specimens.txt
|   |   `-- val_specimens.txt
|   `-- tiles/
|-- images/
|   |-- test/ (xylem_0201.jpg ..)
|   |-- train/
|   `-- val/
|-- labels/
|   |-- test/ (xylem_0201.txt ..)
|   |-- train/
|   `-- val/
|-- real_dataset/
|   |-- README.md
|   `-- plant_1/ .. plant_16/ (img/, json/)
|-- results/
|   |-- conference_paper/
|   |   |-- FINAL_PROJECT_RECAP_REPORT.md
|   |   |-- conference_paper.md
|   |   |-- conference_paper_outline.md
|   |   |-- final_figures/
|   |   |   |-- make_max_f1.py
|   |   |   |-- make_prcurves.py
|   |   |   |-- make_qualitative.py
|   |   |   |-- max_f1_table.csv
|   |   |   |-- prcurves.pdf / .png
|   |   |   `-- qualitative.pdf / .png
|   |   |-- main.tex
|   |   `-- generate_pdf.py
|   |-- discovery/
|   |   |-- PROJECT_DISCOVERY_REPORT.md
|   |   |-- compute_data_facts.py
|   |   |-- data_facts.json
|   |   `-- extract_benchmarks.py
|   |-- inspection/
|   |   |-- README.md
|   |   `-- real_dataset_manifest.json
|   |-- paper/
|   |   |-- final_results.json
|   |   |-- final_results.md
|   |   `-- reeval_test_q1.json
|   `-- real/
|       |-- E1.json .. E8.json
|-- runs/
|   `-- detect/
|       |-- results/real/ (E1..E8 directories with weights/best.pt)
|       |-- runs/smoke/ (E1..E8 smoke artifacts)
|       `-- val-* / train-* (Ultralytics run directories)
|-- src/
|   |-- __init__.py
|   |-- audit_suppression.py
|   |-- benchmark.py
|   |-- error_analysis.py
|   |-- eval_protocol_audit.py
|   |-- evaluate.py
|   |-- generate_paper_results.py
|   |-- inspect_real_data.py
|   |-- model_factory.py
|   |-- reeval_test_conf.py
|   |-- run_e8_smoke.py
|   |-- run_smoke_audit.py
|   |-- train.py
|   |-- train_e6_refiner.py
|   |-- train_real_suite.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- boundary_targets.py
|   |   |-- prepare_real_dataset.py
|   |   |-- split.py
|   |   `-- tiling.py
|   `-- models/
|       |-- __init__.py
|       |-- boundary_branch.py
|       |-- context_refine.py
|       |-- dcn_optional.py
|       |-- fusion_neck.py
|       |-- losses.py
|       |-- p2_head.py
|       `-- yolov8_baseline.py
`-- weights/
    `-- yolo26n.pt
```

### 1.2 Top-Level Folder and File Purpose Map

| Item | Type | Purpose |
| :--- | :--- | :--- |
| `configs/` | Directory | YAML model definition architectures for stock, P2, nc=1, nc=3, and E8 composite. |
| `data/` | Directory | Data storage: raw micrographs (`real_raw/`), full labels, specimen split manifests, and tiled real dataset (`real/`). |
| `images/` | Directory | Legacy synthetic 3-class xylem image dataset (`vessel`, `fiber`, `parenchyma`). |
| `labels/` | Directory | Legacy synthetic 3-class xylem label text files. |
| `real_dataset/` | Directory | Exact duplicate mirror of `data/real_raw/` containing raw Arabidopsis micrographs and LabelMe JSON contours. |
| `results/` | Directory | Output artifacts: conference paper manuscripts, audit reports, real suite JSONs, paper tables, and discovery files. |
| `runs/` | Directory | Ultralytics detection training and validation run outputs, checkpoint weights, confusion matrices, and training curves. |
| `src/` | Directory | Core source code: dataset tiling, model definitions (P2, BiFPN, BoundaryHead, Shape-IoU), training, and evaluation suites. |
| `weights/` | Directory | Secondary model weights folder containing `yolo26n.pt`. |
| `.gitignore` | File | Git ignore rules excluding Python caches, virtual environments, `.pt` weights, and runs directories. |
| `README.md` | File | Legacy documentation describing the initial 100-epoch synthetic xylem 3-class ablation study. |
| `README.txt` | File | Text copy of the project README. |
| `requirements.txt` | File | Python package requirements for the project. |
| `data.yaml` | File | Root dataset YAML pointing to the legacy 3-class synthetic xylem dataset. |
| `benchmark_gpu_rigorous.py` | File | GPU benchmark suite measuring median latency, FPS, and VRAM across batch sizes 1 & 16. |
| `benchmark_gpu_throughput.py` | File | Throughput stress test measuring maximum inference FPS across batch sizes. |
| `demo_inference.py` | File | Demo script running prediction on a single test image (`images/test/xylem_0212.jpg`). |
| `download_and_install_gpu.py` | File | Resumable wheel downloader and installer for PyTorch CUDA 12.4 in `.venv_gpu`. |
| `install_gpu_env.py` | File | Environment setup script configuring CUDA dependencies in `.venv_gpu`. |
| `prepare_real_data.py` | File | Top-level script to invoke tiled dataset generation. |
| `run_full_sweep.py` | File | Batch runner executing full multi-stage ablation experiments sequentially. |
| `scratch_intercept.py` | File | Prototype verification script for E4 BoundaryHead layer wrapping. |
| `scratch_test_neck.py` | File | Prototype verification script for E3 Bidirectional FusionNeck feature shapes. |
| `verify_gpu_e1.py` | File | Verification script testing PyTorch CUDA availability and baseline E1 forward pass. |
| `yolov8n.pt` | File | Stock Ultralytics YOLOv8n COCO pretrained checkpoint (6.25 MB). |
| `yolo26n.pt` | File | Pretrained checkpoint asset (5.29 MB). |

---

### 1.3 Entry-Point Scripts and CLI Arguments

Below is every executable entry point in the repository that contains an `if __name__ == "__main__":` block or an argument parser:

#### A. Data Pipeline Scripts
1. **`src/data/prepare_real_dataset.py`**
   - **CLI**: None (executes default pipeline).
   - **Action**: Converts LabelMe polygon annotations to single-class YOLO bounding boxes (`nc=1`, `cell`), enforces specimen-level split (11 train, 3 val, 2 test), generates $640 \times 640$ overlapping tiles (overlap=0.25, step=480), and outputs `data/real/data.yaml`.
2. **`prepare_real_data.py`**
   - `parser = argparse.ArgumentParser(description="Prepare real micrograph dataset")`
   - `--tile_size`: `int`, default `640` (Tile side length).
   - `--overlap`: `float`, default `0.25` (Tile overlap fraction).
3. **`src/data/tiling.py`**
   - `parser = argparse.ArgumentParser(description="Generate overlapping tiles")`
   - `--image_dir`: `str`, required (Directory containing full micrographs).
   - `--label_dir`: `str`, default `None` (Directory containing full YOLO labels).
   - `--output_dir`: `str`, default `"data/tiles"` (Output directory for tiles).
   - `--tile_size`: `int`, default `512` (Tile side length).
   - `--overlap`: `float`, default `0.25` (Overlap ratio).
   - `--min_visible`: `float`, default `0.3` (Minimum visible fraction of bbox retained).
4. **`src/data/split.py`**
   - `parser = argparse.ArgumentParser(description="Specimen-aware dataset split")`
   - `--data_dir`: `str`, required (Root image directory).
   - `--output_dir`: `str`, default `"data/splits"` (Destination for split text manifests).
   - `--train`: `float`, default `0.7` (Train ratio).
   - `--val`: `float`, default `0.15` (Validation ratio).
   - `--test`: `float`, default `0.15` (Test ratio).
   - `--seed`: `int`, default `42` (Random seed).
   - `--specimen_pattern`: `str`, default `None` (Regex for specimen grouping).
5. **`src/data/boundary_targets.py`**
   - `parser = argparse.ArgumentParser(description="Generate auxiliary boundary ground truth")`
   - `--label_dir`: `str`, required (Path to YOLO labels).
   - `--output_dir`: `str`, default `"data/boundary_maps"` (Output mask directory).
   - `--img_size`: `int`, default `512` (Grid dimension).
   - `--thickness`: `int`, default `3` (Boundary line thickness in px).
   - `--mode`: `str`, choices `["binary", "gaussian"]`, default `"binary"`.
   - `--sigma`: `float`, default `1.5` (Gaussian blur standard deviation).

#### B. Model Factory & Architecture Verification
6. **`src/model_factory.py`**
   - `parser = argparse.ArgumentParser(description="Build and report ablation complexity table")`
   - `--stages`: `nargs="*"`, default `None` (List of stages to instantiate, e.g., `E1 E2 E3`).
   - `--output`: `str`, default `None` (Path to save summary JSON).

#### C. Training & Ablation Execution
7. **`src/train_real_suite.py`**
   - `parser = argparse.ArgumentParser(description="Real-data training and evaluation suite")`
   - `--stage`: `str`, default `None` (Stage to train: `E1`, `E2`, `E3`, `E4`, `E5a`, `E5b`, `E8`).
   - `--epochs`: `int`, default `50` (Training epochs).
   - `--batch`: `int`, default `16` (Batch size; falls back dynamically if OOM occurs).
   - `--patience`: `int`, default `20` (Early stopping patience).
   - `--seed`: `int`, default `42` (Random seed).
   - `--data`: `str`, default `"data/real/data.yaml"` (Data YAML path).
8. **`src/train.py`** (Legacy sweep runner)
   - `--stage`: `str`, required.
   - `--epochs`: `int`, default `None`.
   - `--batch_size`: `int`, default `None`.
   - `--imgsz`: `int`, default `None`.
   - `--lr0`: `float`, default `None`.
   - `--seeds`: `int`, `nargs="+"`, default `None`.
   - `--device`: `str`, default `None`.
9. **`src/train_e6_refiner.py`**
   - CLI: None (trains post-hoc E6 ContextRefineClassifier).
10. **`run_full_sweep.py`**
    - CLI: None (runs primary experimental sequence).

#### D. Evaluation & Audit Scripts
11. **`src/evaluate.py`**
    - `--model`: `str`, required (Path to checkpoint `.pt`).
    - `--data`: `str`, required (Path to `data.yaml`).
    - `--split`: `str`, default `"test"`, choices `["val", "test"]`.
    - `--conf`: `float`, default `0.25` (Confidence threshold).
    - `--iou`: `float`, default `0.45` (NMS IoU threshold).
    - `--output_dir`: `str`, default `None`.
    - `--imgsz`: `int`, default `640` (Must be 640 per protocol).
    - `--max_det`: `int`, default `300`.
    - `--tiling`: `action="store_true"`.
    - `--tile_size`: `int`, default `512`.
    - `--tile_overlap`: `float`, default `0.25`.
12. **`src/reeval_test_conf.py`**
    - CLI: None. Evaluates E1–E8 on test split at conf=0.001 and conf=0.25.
13. **`src/eval_protocol_audit.py`**
    - CLI: None. Audits fairness consistency across evaluation scripts.
14. **`src/error_analysis.py`**
    - `--model`: `str`, required.
    - `--data`: `str`, required.
    - `--split`: `str`, default `"test"`.
    - `--conf`: `float`, default `0.25`.
    - `--iou`: `float`, default `0.45`.
    - `--output_dir`: `str`, default `None`.
15. **`src/inspect_real_data.py`**
    - CLI: None. Verifies raw data integrity, polygon coordinates, and JSON schemas.
16. **`src/audit_suppression.py`**
    - CLI: None. Audits prediction confidence suppression dynamics for E4.
17. **`src/generate_paper_results.py`**
    - CLI: None. Compiles `results/paper/final_results.json` and `.md`.

#### E. Benchmarking & Hardware Profiling
18. **`src/benchmark.py`**
    - `--model`: `str`, required.
    - `--imgsz`: `int`, default `512`.
    - `--device`: `str`, default `None`.
    - `--warmup`: `int`, default `10`.
    - `--iterations`: `int`, default `100`.
    - `--output`: `str`, default `None`.
19. **`benchmark_gpu_rigorous.py`**
    - CLI: None. Executes formal latency, FPS, and VRAM benchmark protocol for batch sizes 1 & 16.
20. **`benchmark_gpu_throughput.py`**
    - CLI: None. Evaluates sustained throughput across batch sweeps.

#### F. Paper Figures & Discovery Scripts
21. **`results/conference_paper/final_figures/make_prcurves.py`**
    - `--use-cache`: Optional flag to plot from cached `prcurves_data.npz`.
22. **`results/conference_paper/final_figures/make_max_f1.py`**
    - CLI: None. Sweeps confidence thresholds at conf=0.001, calculates max F1, optimal threshold $\tau^*$, and evaluates conf=0.25 metrics.
23. **`results/conference_paper/final_figures/make_qualitative.py`**
    - `--filter-nonzero`: Optional flag to compute median tile exclusively over non-empty tiles. Default runs across all 80 specimen tiles and outputs both standard and companion figures.
24. **`results/conference_paper/generate_pdf.py`**
    - CLI: None. Compiles paper HTML to PDF via headless Chromium/Edge.

---

## 2. ENVIRONMENT

### 2.1 Operating System and Shell
- **OS**: Windows 11 Home (64-bit), Version `10.0.26300` (Build 26300).
- **Platform**: `Windows-11-10.0.26300-SP0`
- **Shell**: Windows PowerShell (execution environment `powershell.exe`).

### 2.2 Python Environment & Activation
- **Python Version**: `3.12.10` (tags/v3.12.10:0cc8128, Apr 8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)].
- **Active Environment Location**: `e:\Yolov8_Plant_Cell_Detection\.venv_gpu`
- **Activation Command**:
  - PowerShell: `.\.venv_gpu\Scripts\Activate.ps1`
  - Command Prompt: `.\.venv_gpu\Scripts\activate.bat`
  - Direct Interpreter: `e:\Yolov8_Plant_Cell_Detection\.venv_gpu\Scripts\python.exe`
- **Secondary Environment**: `e:\Yolov8_Plant_Cell_Detection\venv` (CPU/legacy virtual environment).

### 2.3 Hardware & GPU Subsystem
- **GPU Model**: NVIDIA GeForce RTX 3050 Laptop GPU
- **VRAM Total**: 4,096 MiB (4.00 GB)
- **Compute Capability**: `(8, 6)` (Ampere Architecture)
- **NVIDIA Driver Version**: `617.42`
- **CUDA Driver API Version**: `13.4` (reported by `nvidia-smi`)
- **PyTorch CUDA Runtime Version**: `12.4`
- **cuDNN Version**: `90100` (cuDNN 9.1.0)

### 2.4 Installed Packages (`pip list` targeted query)
```text
Package                Version
---------------------- -------------
matplotlib             3.11.2
numpy                  2.5.4
opencv-python          5.0.0.93
opencv-python-headless 5.0.0.93
pandas                 3.0.6
PyYAML                 6.0.3
torch                  2.6.0+cu124
torchvision            0.21.0+cu124
ultralytics            8.4.182
ultralytics-platform   0.1.90
ultralytics-thop       2.2.2
```

**Web/UI Libraries Query**:
- `streamlit`: **NOT FOUND**
- `flask`: **NOT FOUND**
- `fastapi`: **NOT FOUND**
- `gradio`: **NOT FOUND**

### 2.5 Requirements & Config Files Verbatim

Only one dependency specification file exists in the repository: `requirements.txt`.  
(`pyproject.toml`, `setup.py`, and `environment.yml` were searched across the root and are **NOT FOUND**).

**File Path**: `e:/Yolov8_Plant_Cell_Detection/requirements.txt`
```text
1: # YOLOv8 Plant Cell Detection  Dependencies
2: # Install: pip install -r requirements.txt
3: 
4: # Core
5: ultralytics>=8.1.0
6: torch>=2.0.0
7: torchvision>=0.15.0
8: 
9: # Image processing
10: opencv-python>=4.8.0
11: Pillow>=10.0.0
12: albumentations>=1.3.0
13: 
14: # Metrics & analysis
15: scikit-learn>=1.3.0
16: numpy>=1.24.0
17: pandas>=2.0.0
18: 
19: # Experiment tracking
20: tensorboard>=2.14.0
21: wandb>=0.15.0           # optional; set use_wandb: true in config
22: 
23: # Profiling
24: thop>=0.1.1             # GFLOPs / params counting
25: 
26: # Config
27: PyYAML>=6.0
28: jsonschema>=4.0.0
29: 
30: # Visualization
31: matplotlib>=3.7.0
32: seaborn>=0.12.0
```

---

## 3. MODEL ASSETS AND LOADING

### 3.1 Checkpoints and File Sizes

| Model Key | Description | File Path | File Size (Bytes) | Size (MB) |
| :--- | :--- | :--- | :---: | :---: |
| **E1** | Stock YOLOv8n Baseline | `runs/detect/results/real/E1/weights/best.pt` | 6,296,887 | 6.01 MB |
| **E2** | P2 Stride-4 Head Detector | `runs/detect/results/real/E2/weights/best.pt` | 6,341,451 | 6.05 MB |
| **E3** | Bidirectional FusionNeck (BiFPN) | `runs/detect/results/real/E3/weights/best.pt` | 4,289,232 | 4.09 MB |
| **E4** | Auxiliary BoundaryHead Branch | `runs/detect/results/real/E4/weights/best.pt` | 6,338,791 | 6.05 MB |
| **E5a** | Shape-IoU Loss | `runs/detect/results/real/E5a/weights/best.pt` | 6,296,887 | 6.01 MB |
| **E5b** | MPDIoU Loss | `runs/detect/results/real/E5b/weights/best.pt` | 6,295,351 | 6.00 MB |
| **E8** | Full Combined Composite Model | `runs/detect/results/real/E8/weights/best.pt` | 6,659,261 | 6.35 MB |
| **Stock** | Ultralytics COCO Pretrained | `yolov8n.pt` | 6,549,796 | 6.25 MB |
| **Secondary**| YOLO26 Pretrained Checkpoint | `yolo26n.pt` / `weights/yolo26n.pt` | 5,544,453 | 5.29 MB |

### 3.2 Loading Behavior: Plain `YOLO(path)` vs Extra Steps

1. **E1, E2, E5a, E5b**:
   - **Status**: Load and execute correctly with plain `YOLO(path)`.
   - **Reason**: E1, E5a, and E5b share the stock YOLOv8n computation graph. E2's P2 detection head layers are standard PyTorch modules serialized directly into the sequential architecture.
2. **E3**:
   - **Status**: Loads with plain `YOLO(path)` for standard evaluation. Custom modules `FusionNeck` and `_FastFusion` are serialized within the model layers.
3. **E4**:
   - **Status**: Checkpoint contains `InterceptWrapper` wrapping layer 15. In pure inference mode (`self.training == False`), `InterceptWrapper.forward(x)` directly returns `self.orig_layer(x)`, bypassing `BoundaryHead`. To avoid potential unpickling attribute errors in isolated scripts, `BoundaryHead` and `InterceptWrapper` must be imported and exposed to `__main__`.
4. **E8**:
   - **Status**: **CANNOT be evaluated with plain `YOLO(path)`. REQUIRES EXPLICIT RE-BINDING.**
   - **Failure Mode**: PyTorch / Ultralytics `torch.save` dropped the dynamic instance method binding `_predict_once`. Plain `YOLO(".../E8/.../best.pt")` falls back to stock `DetectionModel._predict_once`, which completely bypasses `fusion_neck`. This drops test mAP50 from **0.6043 down to 0.0165** (confidence scores collapse to zero).
   - **Resolution Protocol**: Must re-bind `_make_e8_predict_once(fusion_neck)` to `model._predict_once` and place `fusion_neck` on CUDA.

### 3.3 Exact Working Code for Model Loading

The following verbatim block demonstrates the exact working loading code used across the repository (from `results/conference_paper/final_figures/make_prcurves.py` and `src/reeval_test_conf.py`):

```python
# ==============================================================================
# Model Loading & Dynamic Forward-Pass Restoration Protocol
# ==============================================================================
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# 1. Custom Module Imports
import src.models.fusion_neck as fn
import src.models.boundary_branch as bb

# 2. Main-Module Attribute Patching (Required for Checkpoint Unpickling)
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "E8FusionNeck", fn.E8FusionNeck)
    setattr(main_mod, "_FastFusion", fn._FastFusion)
    setattr(main_mod, "BoundaryHead", bb.BoundaryHead)
    setattr(main_mod, "InterceptWrapper", bb.InterceptWrapper)
    class E8LossWrapper: pass
    class BoundaryLossWrapper: pass
    setattr(main_mod, "E8LossWrapper", E8LossWrapper)
    setattr(main_mod, "BoundaryLossWrapper", BoundaryLossWrapper)

# 3. Model Loader Function
def load_ablation_model(stage: str, checkpoint_path: str):
    yolo = YOLO(checkpoint_path)
    
    # Place on CUDA before binding so weights match CUDA input tensors
    if torch.cuda.is_available():
        yolo.to("cuda")
        
    # E8-Specific Restoration
    if stage == "E8":
        m = yolo.model
        if hasattr(m, "fusion_neck"):
            # Explicitly re-bind the dynamic forward pass method
            m._predict_once = fn._make_e8_predict_once(m.fusion_neck).__get__(m, type(m))
            
    return yolo
```

### 3.4 Key Code Sections from Repository

#### A. From `src/model_factory.py` (lines 359–385)
```python
359: def _build_e8(nc: int = 1) -> Tuple[Any, StageManifest]:
360:     """E8: P2 head + E8FusionNeck + BoundaryHead + ShapeIoU + ContextRefiner."""
361:     from ultralytics import YOLO
362:     cfg = "configs/yolov8n_p2_nc1.yaml" if nc == 1 else "configs/yolov8n_p2_nc3.yaml"
363:     model = YOLO(cfg)
364: 
365:     # Inject E8FusionNeck for P3/P4/P5
366:     fn = E8FusionNeck(in_ch=(64, 128, 256), inner_ch=128, num_repeats=2)
367:     model.model.fusion_neck = fn
368:     model.model._predict_once = _make_e8_predict_once(fn).__get__(model.model, type(model.model))
369: 
370:     # Wrap layer 18 (P2-level feature in yolov8n-p2) with BoundaryHead
371:     b_head = BoundaryHead(in_channels=32, hidden_channels=32, alpha=0.5)
372:     orig_layer18 = model.model.model[18]
373:     wrapper = InterceptWrapper(orig_layer18, b_head)
374:     wrapper.i = 18
375:     wrapper.f = getattr(orig_layer18, "f", -1)
376:     model.model.model[18] = wrapper
```

#### B. From `src/models/fusion_neck.py` (lines 367–401)
```python
367: def _make_e8_predict_once(fusion_neck: E8FusionNeck):
368:     """
369:     Returns a _predict_once method that injects E8FusionNeck after layer 9.
370:     Overwrites saved backbone feature maps y[4], y[6], y[9] with BiFPN-fused features.
371:     """
372:     def _predict_once(self, x, profile=False, visualize=False, embed=None):
373:         y = []
374:         for m in self.model:
375:             if m.f != -1:
376:                 if isinstance(m.f, int):
377:                     x = y[m.f]
378:                 else:
379:                     x = [x if j == -1 else y[j] for j in m.f]
380: 
381:             x = m(x)
382: 
383:             # After layer 9 (SPPF), apply BiFPN before the neck uses the features
384:             if getattr(m, "i", -999) == 9:
385:                 p3_idx = 4  # y[4] = P3 (C2f after layer 4)
386:                 p4_idx = 6  # y[6] = P4 (C2f after layer 6)
387:                 # P5 = current x (SPPF output)
388: 
389:                 p3_fused, p4_fused, p5_fused = fusion_neck(
390:                     [y[p3_idx], y[p4_idx], x]
391:                 )
392:                 # Overwrite saved feature maps — neck Concat layers read from here
393:                 y[p3_idx] = p3_fused   # 64ch
394:                 y[p4_idx] = p4_fused   # 128ch
395:                 x = p5_fused            # 256ch
396: 
397:             y.append(x if (hasattr(m, "i") and m.i in self.save) else None)
398: 
399:         return x
400: 
401:     return _predict_once
```

---

## 4. INFERENCE PIPELINE THAT ALREADY EXISTS

### 4.1 Single Tile ($640 \times 640$) Inference
- **Status**: **EXISTS**.
- **Existing Implementations**:
  1. `demo_inference.py`: Runs `model.predict(source=test_image, save=True, project="results", name="demo_inference")` on a single synthetic tile.
  2. `src/train_real_suite.py` (`save_representative_predictions` lines 224–260): Runs inference on representative test tiles using `yolo.predict(source=str(img_path), imgsz=imgsz, conf=conf_thresh, iou=iou_thresh, save=False)[0]` and renders predictions via `res.plot()`.
  3. `results/conference_paper/final_figures/make_qualitative.py` (`run_model_inference` lines 88–111): Runs inference on specific test tiles at `conf=0.25, iou=0.45, max_det=300` and extracts `res.boxes.xyxy`.

### 4.2 Full Micrograph ($3840 \times 2160$) Inference
- **Status**: **LIBRARY FUNCTIONS EXIST, BUT STANDALONE END-TO-END SCRIPT IS NOT FOUND**.
- **Details**:
  - The functions for coordinate generation, tile inference, coordinate remapping, and cross-tile NMS are implemented in `src/data/tiling.py` (lines 379–438).
  - However, no CLI script, pipeline runner, or demo script currently imports and calls `predict_tiled` on full micrographs in `data/real_raw/`.

**Verbatim Code from `src/data/tiling.py` (lines 245–314, 379–438)**:
```python
245: def merge_tile_predictions(
246:     tile_preds: List[Dict],
247:     tile_coords: List[Tuple[int, int, int, int]],
248:     image_shape: Tuple[int, int],
249:     iou_thresh: float = 0.45,
250: ) -> np.ndarray:
251:     """Merge detections from overlapping tiles via cross-tile NMS.
...
270:     all_boxes = []
271:     all_scores = []
272:     all_classes = []
273: 
274:     for preds, (tx1, ty1, _tx2, _ty2) in zip(tile_preds, tile_coords):
275:         boxes = np.array(preds["boxes"], dtype=np.float32)
276:         scores = np.array(preds["scores"], dtype=np.float32)
277:         classes = np.array(preds["classes"], dtype=np.int32)
278: 
279:         if len(boxes) == 0:
280:             continue
281: 
282:         # Offset boxes to full-image coordinates
283:         boxes[:, 0] += tx1
284:         boxes[:, 1] += ty1
285:         boxes[:, 2] += tx1
286:         boxes[:, 3] += ty1
287: 
288:         # Clip to image bounds
289:         img_h, img_w = image_shape
290:         boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
291:         boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
292:         boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
293:         boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)
294: 
295:         all_boxes.append(boxes)
296:         all_scores.append(scores)
297:         all_classes.append(classes)
298: 
299:     if not all_boxes:
300:         return np.zeros((0, 6), dtype=np.float32)
301: 
302:     all_boxes = np.concatenate(all_boxes, axis=0)
303:     all_scores = np.concatenate(all_scores, axis=0)
304:     all_classes = np.concatenate(all_classes, axis=0)
305: 
306:     # Per-class NMS
307:     keep_indices = _per_class_nms(all_boxes, all_scores, all_classes, iou_thresh)
308: 
309:     merged = np.column_stack([
310:         all_boxes[keep_indices],
311:         all_scores[keep_indices],
312:         all_classes[keep_indices].astype(np.float32),
313:     ])
314:     return merged
...
379: def predict_tiled(
380:     model: Any,
381:     image: np.ndarray,
382:     tile_size: int = 512,
383:     overlap: float = 0.25,
384:     conf_thresh: float = 0.25,
385:     iou_thresh: float = 0.45,
386:     max_det: int = 300,
387: ) -> np.ndarray:
388:     """Run overlapping tile inference on an image and merge predictions via cross-tile NMS."""
389:     img_h, img_w = image.shape[:2]
390:     coords = compute_tile_coords(img_h, img_w, tile_size, overlap)
391:     tile_preds: List[Dict] = []
392: 
393:     for (x1, y1, x2, y2) in coords:
394:         tile = image[y1:y2, x1:x2]
395:         th, tw = tile.shape[:2]
396:         if th < tile_size or tw < tile_size:
397:             padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
398:             padded[:th, :tw] = tile
399:             tile = padded
400: 
401:         results = model.predict(
402:             tile,
403:             imgsz=tile_size,
404:             conf=conf_thresh,
405:             iou=iou_thresh,
406:             max_det=max_det,
407:             verbose=False,
408:         )[0]
409: 
410:         boxes = results.boxes.xyxy.cpu().numpy()
411:         scores = results.boxes.conf.cpu().numpy()
412:         classes = results.boxes.cls.cpu().numpy()
413:         tile_preds.append({"boxes": boxes, "scores": scores, "classes": classes})
414: 
415:     return merge_tile_predictions(tile_preds, coords, (img_h, img_w), iou_thresh=iou_thresh)
```

### 4.3 Input and Output Formats of Existing Prediction Scripts

1. **`demo_inference.py`**:
   - **Input**: Image file path (e.g. `images/test/xylem_0212.jpg`).
   - **Output**: Terminal stdout (class counts), annotated image saved to `results/demo_inference/*.jpg`.
2. **`src/train_real_suite.py:save_representative_predictions`**:
   - **Input**: Directory of `.jpg`/`.png` test tiles, sample count (4).
   - **Output**: Visualized JPGs with rendered bounding boxes saved to `runs/detect/results/real/<stage>/representative_predictions/*.jpg`.
3. **`results/conference_paper/final_figures/make_qualitative.py`**:
   - **Input**: Deterministically selected test tiles from `data/real/images/test/`.
   - **Output**: Matplotlib figures saved as vector PDF (`qualitative.pdf`) and 300-dpi raster PNG (`qualitative.png`).

---

## 5. DATA FACTS

*(Computed via read-only analysis script `results/discovery/compute_data_facts.py`)*

### 5.1 Dataset YAML and Folder Layout
**File Path**: `e:/Yolov8_Plant_Cell_Detection/data/real/data.yaml`
```yaml
1: path: E:/Yolov8_Plant_Cell_Detection/data/real
2: train: images/train
3: val: images/val
4: test: images/test
5: 
6: nc: 1
7: names:
8:   0: cell
```

- **Folder Layout**:
  - `data/real/images/train/`, `data/real/images/val/`, `data/real/images/test/`
  - `data/real/labels/train/`, `data/real/labels/val/`, `data/real/labels/test/`
- **Tile Image Dimensions**: Exactly $640 \times 640 \times 3$ across all splits.
- **Tile Image Format**: Standard JPEG (`.jpg`).
- **Label Format**: Standard normalized YOLO format (`0 <x_center> <y_center> <width> <height>`).

### 5.2 Split Breakdown: Tiles, Zero-Label Tiles, and Boxes

| Split | Specimens | Total Tiles | Zero-Label Tiles | Tiles with Cells | Total Bounding Boxes |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `plant_1..5, 7, 9..13` (11 specimens) | 880 | 502 (57.0%) | 378 (43.0%) | 5,007 |
| **Val** | `plant_6, 14, 15` (3 specimens) | 240 | 136 (56.7%) | 104 (43.3%) | 1,502 |
| **Test** | `plant_8, 16` (2 specimens) | 160 | 77 (48.1%) | 83 (51.9%) | 1,247 |
| **Total** | **16 specimens** | **1,280** | **715 (55.9%)** | **565 (44.1%)** | **7,756** |

#### Per-Specimen Breakdown

| Specimen | Split | Micrograph Stems | Total Tiles | Zero-Label Tiles | Total Bounding Boxes |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **plant_1** | Train | `A2_0808`, `A2_0809` | 80 | 43 | 381 |
| **plant_2** | Train | `A3_0808`, `A3_0809` | 80 | 43 | 375 |
| **plant_3** | Train | `A4_0808`, `A4_0809` | 80 | 42 | 389 |
| **plant_4** | Train | `B2_0808`, `B2_0809` | 80 | 37 | 542 |
| **plant_5** | Train | `B3_0808`, `B3_0809` | 80 | 40 | 518 |
| **plant_7** | Train | `B5_0808`, `B5_0809` | 80 | 45 | 498 |
| **plant_9** | Train | `250102-1-1_T0`, `250102-1-1_T24` | 80 | 52 | 448 |
| **plant_10** | Train | `250102-1-2_T0`, `250102-1-2_T24` | 80 | 56 | 440 |
| **plant_11** | Train | `250102-1-3_T0`, `250102-1-3_T24` | 80 | 48 | 461 |
| **plant_12** | Train | `250102-2-1_T0`, `250102-2-1_T24` | 80 | 53 | 393 |
| **plant_13** | Train | `250102-2-2_T0`, `250102-2-2_T24` | 80 | 43 | 562 |
| **plant_6** | Val | `B4_0808`, `B4_0809` | 80 | 39 | 501 |
| **plant_14** | Val | `250102-3-1_T0`, `250102-3-1_T24` | 80 | 56 | 377 |
| **plant_15** | Val | `250102-3-2_T0`, `250102-3-2_T24` | 80 | 41 | 624 |
| **plant_8** | Test | `B6_0808`, `B6_0809` | 80 | 28 | 744 |
| **plant_16** | Test | `250102-4-1_T0`, `250102-4-1_T24` | 80 | 49 | 503 |

*(Note: Each specimen consists of 2 full $3840 \times 2160$ micrographs, generating exactly 80 tiles per specimen with 25% overlap).*

---

### 5.3 Ground-Truth Box Size Statistics (Pixels at $640 \times 640$ Tile Scale)

#### Overall Dataset (7,756 bounding boxes)
| Dimension | Min | 5th % | 25th % | 50th % (Median) | 75th % | 95th % | Max | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Width ($w$)** | 29.60 px | 68.85 px | 111.78 px | **138.45 px** | 173.00 px | 216.10 px | 307.70 px | 141.85 $\pm$ 44.48 px |
| **Height ($h$)** | 27.20 px | 66.25 px | 116.70 px | **140.70 px** | 161.20 px | 193.40 px | 271.60 px | 137.18 $\pm$ 36.60 px |
| **$\sqrt{w \cdot h}$** | 54.94 px | 86.78 px | 115.08 px | **135.94 px** | 156.58 px | 187.53 px | 284.54 px | 136.23 $\pm$ 30.28 px |

#### Per-Split Breakdown
- **Train Split (5,007 boxes)**:
  - Width ($w$): Min 29.70 px, Median **140.50 px**, Max 307.70 px (Mean 142.50 px)
  - Height ($h$): Min 27.20 px, Median **139.60 px**, Max 271.60 px (Mean 135.86 px)
  - $\sqrt{w \cdot h}$: Min 54.94 px, Median **135.98 px**, Max 284.54 px (Mean 135.89 px)
- **Validation Split (1,502 boxes)**:
  - Width ($w$): Min 29.60 px, Median **128.50 px**, Max 238.40 px (Mean 133.64 px)
  - Height ($h$): Min 32.60 px, Median **143.60 px**, Max 269.20 px (Mean 140.29 px)
  - $\sqrt{w \cdot h}$: Min 67.36 px, Median **133.29 px**, Max 236.06 px (Mean 133.90 px)
- **Test Split (1,247 boxes)**:
  - Width ($w$): Min 32.80 px, Median **145.60 px**, Max 300.30 px (Mean 149.13 px)
  - Height ($h$): Min 35.00 px, Median **142.60 px**, Max 226.50 px (Mean 138.77 px)
  - $\sqrt{w \cdot h}$: Min 55.09 px, Median **139.84 px**, Max 221.31 px (Mean 140.41 px)

### 5.4 Full Micrograph Area Coverage Fraction (Union of Bounding Boxes)
- **Overall Mean Micrograph Coverage**: **19.84%** ($\pm 4.12\%$).
- **Per-Specimen Mean Coverage**:
  - `plant_1`: 21.35%
  - `plant_2`: 19.63%
  - `plant_3`: 18.57%
  - `plant_4`: 22.84%
  - `plant_5`: 22.12%
  - `plant_6`: 20.31% (Val)
  - `plant_7`: 23.47%
  - `plant_8`: 24.16% (Test)
  - `plant_9`: 17.89%
  - `plant_10`: 16.42%
  - `plant_11`: 17.51%
  - `plant_12`: 15.68%
  - `plant_13`: 20.91%
  - `plant_14`: 16.89% (Val)
  - `plant_15`: 21.90% (Val)
  - `plant_16`: 17.72% (Test)

### 5.5 Dataset License / Redistribution Notes
- **Findings**:
  - In `data/real_raw/README.md` and `real_dataset/README.md`, Section "Usage Notes" states:  
    `"This dataset is published concurrently with the TopoInherit framework. The detailed geometry bounds and temporal mappings within the json/ folder can be directly parsed for executing cell lineage tracking validations or training new geometric cell models."`
  - No formal open-source license file (`LICENSE`, `LICENSE.txt`, or `COPYING`) exists in the repository. Redistribution terms are governed by the parent TopoInherit publication.

---

## 6. TRAINING, EVALUATION AND RESULTS ARTIFACTS

### 6.1 Training Execution
- **Script**: `src/train_real_suite.py`
- **Execution Command Example**:
  ```bash
  .venv_gpu\Scripts\python.exe src/train_real_suite.py --stage E3 --epochs 50 --batch 16 --patience 20 --seed 42 --data data/real/data.yaml
  ```
- **Experimental Protocol**:
  - Standardized across all stages: 50 epochs, early stopping patience 20, seed 42, SGD optimizer, initial learning rate $\text{lr0}=0.01$, image size $640 \times 640$.
  - Dynamic OOM recovery: catches CUDA Out of Memory, clears GPU cache, halves batch size, and resumes.

### 6.2 Results Artifacts Locations
1. **Ultralytics Training & Validation Runs**:
   - Checkpoints: `runs/detect/results/real/{E1,E2,E3,E4,E5a,E5b,E8}/weights/{best.pt, last.pt}`
   - Training curves and args: `runs/detect/results/real/{stage}/args.yaml`, `results.csv`, `BoxPR_curve.png`, `confusion_matrix.png`.
2. **Real Suite JSON Reports**:
   - `results/real/E1.json` through `results/real/E8.json` (contains inference params, GFLOPs, benchmark latency/FPS/VRAM, test metrics, and verification status).
   - `results/real/E6.json` and `E7.json` explicitly marked `SKIPPED`.
3. **Paper Verification Summaries**:
   - `results/paper/final_results.json` and `results/paper/final_results.md`
   - `results/paper/reeval_test_q1.json`
4. **Conference Paper Submission Files**:
   - LaTeX source: `results/conference_paper/main.tex`
   - Master technical recap: `results/conference_paper/FINAL_PROJECT_RECAP_REPORT.md`
   - Figures & vector outputs: `results/conference_paper/final_figures/` (`prcurves.pdf`, `qualitative.pdf`, `max_f1_table.csv`).

### 6.3 Exact Command to Reproduce Test-Set Evaluation of a Model

To reproduce test-set evaluation for **E3**:
```bash
.venv_gpu\Scripts\python.exe -c "
import sys, torch
from ultralytics import YOLO
yolo = YOLO('runs/detect/results/real/E3/weights/best.pt')
if torch.cuda.is_available():
    yolo.to('cuda')
res = yolo.val(
    data='data/real/data.yaml',
    split='test',
    imgsz=640,
    conf=0.001,
    iou=0.45,
    max_det=300,
    device='cuda:0'
)
print('E3 mAP50:', res.box.map50, 'mAP50-95:', res.box.map)
"
```

To reproduce test-set evaluation for **E8** (with forward-pass re-binding):
```bash
.venv_gpu\Scripts\python.exe -c "
import sys, torch
from ultralytics import YOLO
import src.models.fusion_neck as fn
import src.models.boundary_branch as bb

main_mod = sys.modules.get('__main__')
setattr(main_mod, 'E8FusionNeck', fn.E8FusionNeck)
setattr(main_mod, '_FastFusion', fn._FastFusion)

yolo = YOLO('runs/detect/results/real/E8/weights/best.pt')
if torch.cuda.is_available():
    yolo.to('cuda')
yolo.model._predict_once = fn._make_e8_predict_once(yolo.model.fusion_neck).__get__(yolo.model, type(yolo.model))

res = yolo.val(
    data='data/real/data.yaml',
    split='test',
    imgsz=640,
    conf=0.001,
    iou=0.45,
    max_det=300,
    device='cuda:0'
)
print('E8 mAP50:', res.box.map50, 'mAP50-95:', res.box.map)
"
```

---

## 7. CODE HEALTH

### 7.1 Hard-Coded Absolute Paths
The following hard-coded absolute paths were discovered in scripts:
- `data/real/data.yaml` line 1: `path: E:/Yolov8_Plant_Cell_Detection/data/real`
- `download_and_install_gpu.py` lines 15, 18, 20: `WHEELS_DIR = Path("E:/wheels")`, `TMP_DIR = Path("E:/tmp")`, `PIP_CACHE = Path("E:/pip_cache")`
- `install_gpu_env.py` lines 15, 17: `TMP_DIR = Path("E:/tmp")`, `PIP_CACHE = Path("E:/pip_cache")`
- `results/conference_paper/generate_pdf.py` line 771: `out_dir = r"e:\Yolov8_Plant_Cell_Detection\results\conference_paper"`
- `results/conference_paper/generate_pdf.py` lines 781, 783: `C:\Program Files\Google\Chrome\Application\chrome.exe` and `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`
- `results/conference_paper/generate_recap_report_pdf.py`: lines 362, 372, 374 (same paths)
- `results/conference_paper/generate_literature_survey_pdf.py`: lines 317, 327, 329 (same paths)

### 7.2 Windows-Specific Code
- Executable paths referencing `.exe` (`chrome.exe`, `msedge.exe`, `python.exe`).
- Path separators using Windows backslashes (`\`) in string literals.
- Windows process spawning with `ShellExperienceHost.exe` / `msedge.exe`.

### 7.3 Monkey Patches & Global State
1. **`bbox_iou` Monkey-Patching (`src/models/losses.py:patched_bbox_iou`)**:
   - Dynamically overwrites `ultralytics.utils.metrics.bbox_iou` and `ultralytics.utils.loss.bbox_iou` at runtime to intercept `CIoU=True` and redirect to Shape-IoU or MPDIoU.
2. **`DetectionTrainer.loss` Overwrite (`src/train_real_suite.py`)**:
   - Wraps standard `v8DetectionLoss` with `BoundaryLossWrapper` to compute auxiliary BCE heatmap losses.
3. **`__main__` Module Dynamic Injection**:
   - `src/train_real_suite.py`, `src/model_factory.py`, and `src/reeval_test_conf.py` inject class definitions into `sys.modules["__main__"]` (`E8FusionNeck`, `_FastFusion`, `BoundaryHead`, `InterceptWrapper`, `E8LossWrapper`) to allow unpickling checkpoints saved in scripts where `__main__` was the defining module.
4. **Dynamic `_predict_once` Instance Method Overwrite**:
   - Overwrites the PyTorch model's forward execution method dynamically on the instantiated Python object.

### 7.4 Comments & Test Suite
- **`TODO` / `FIXME` / `HACK` comments**: **NONE FOUND** in active codebase.
- **Formal Unit / Integration Tests**: **NOT FOUND**. No `tests/` folder or `test_*.py` files exist in the repository. Smoke testing was performed via standalone scripts (`src/run_smoke_audit.py`, `src/run_e8_smoke.py`).

---

## 8. VERSION CONTROL AND PACKAGING

### 8.1 Git Repository Status
- **Is this a Git Repo?**: **YES** (active `.git` directory present).
- **Current Branch**: `master`
- **Git Remotes**: **NONE CONFIGURED** (`git remote -v` returned empty; purely local git repository).
- **Git Commit Log (Last 5 Commits)**:
  ```text
  d6a2f57 pre-real-data-v3: unified nc=3 model factory, DCN FLOP hook, E8 fusion neck, and verified all 9 smoke stages pass (v2 was premature)
  2f13c5f pre-real-data-v2: unified model factory, reconciled complexity table, and smoke test audit
  d36f64d pre-real-data-v1: fair evaluation protocol, E8 architecture, tiling inference, and refiner audit
  ```
  *(Total commit count on `master`: exactly 3 commits).*

### 8.2 Working Tree Modifications (`git status --short`)
- Tracked modified files:
  - `M src/model_factory.py`
  - `M src/run_smoke_audit.py`
- Untracked files/folders:
  - `.venv_gpu/`, `data/`, `real_dataset/`, `images/`, `labels/`, `runs/`, `results/`, `requirements.txt`, `configs/yolov8n_nc1.yaml`, `configs/yolov8n_p2_nc1.yaml`, `src/data/prepare_real_dataset.py`, `src/train_real_suite.py`, `src/reeval_test_conf.py`, `benchmark_gpu_rigorous.py`, `benchmark_gpu_throughput.py`, `demo_inference.py`, `prepare_real_data.py`.

### 8.3 `.gitignore` Contents
**File Path**: `e:/Yolov8_Plant_Cell_Detection/.gitignore`
```gitignore
1: # Python
2: __pycache__/
3: *.py[cod]
4: *$py.class
5: venv/
6: env/
7: .env
8: 
9: # Ultralytics & Weights
10: *.pt
11: runs/
12: *.cache
13: 
14: # IDE
15: .vscode/
16: .idea/
```
*(Notice: `.venv_gpu/` is NOT explicitly listed in `.gitignore`, which is why it appears as an untracked directory in `git status`).*

### 8.4 Top 10 Largest Items in Repository (Must NOT be committed)

1. `.venv_gpu/` — Directory: **5.15 GB** (5,530,775,172 bytes)
2. `venv/` — Directory: **1.38 GB** (1,486,369,438 bytes)
3. `runs/` — Directory: **977.64 MB** (1,025,125,558 bytes)
4. `data/` — Directory: **592.26 MB** (621,027,297 bytes)
5. `real_dataset/` — Directory: **507.05 MB** (531,684,457 bytes)
6. `images/` — Directory: **13.74 MB** (14,403,616 bytes)
7. `results/` — Directory: **8.37 MB** (8,780,228 bytes)
8. `yolov8n.pt` — File: **6.25 MB** (6,549,796 bytes)
9. `weights/` — Directory: **5.29 MB** (5,544,453 bytes)
10. `yolo26n.pt` — File: **5.29 MB** (5,544,453 bytes)

### 8.5 License File
- **Status**: **NOT FOUND**. No `LICENSE` or `COPYING` file exists in the repository.

---

## 9. PERFORMANCE FACTS ALREADY MEASURED

### 9.1 Storage Location
- Primary structured storage: `results/real/E1.json` through `results/real/E8.json` under the `"benchmark"` object.
- Benchmark source execution script: `benchmark_gpu_rigorous.py` and `src/train_real_suite.py`.

### 9.2 Measured Benchmarks on NVIDIA RTX 3050 Laptop GPU (4GB VRAM)

| Model Key | Description | Inference Params | GFLOPs | Batch 1 FP32 (Latency / FPS / VRAM) | Batch 1 FP16 (Latency / FPS / VRAM) | Batch 16 FP32 (Latency / FPS / VRAM) | Batch 16 FP16 (Latency / FPS / VRAM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Stock Baseline | 3,011,043 | 8.19 | 9.07 ms / 110.2 / 125.2 MB | 11.08 ms / 90.3 / 78.4 MB | 60.76 ms / 263.3 / 460.6 MB | 38.30 ms / 417.8 / 257.2 MB |
| **E2** | P2 Stride-4 Head | 2,926,692 | 12.36 | 10.58 ms / 94.5 / 145.9 MB | 13.80 ms / 72.5 / 89.7 MB | 99.88 ms / 160.2 / 805.1 MB | 62.63 ms / 255.5 / 421.5 MB |
| **E3** | FusionNeck (BiFPN) | **2,019,061** | **7.28** | 9.17 ms / 109.1 / 124.7 MB | **10.92 ms / 91.5** / 78.4 MB | 61.12 ms / 261.8 / 459.6 MB | 38.93 ms / 411.0 / 256.7 MB |
| **E4** | BoundaryHead | 3,011,043 | 8.19 | 9.02 ms / 110.8 / 125.2 MB | 11.07 ms / 90.3 / 78.4 MB | 60.75 ms / 263.4 / 460.6 MB | 38.50 ms / 415.6 / 256.2 MB |
| **E5a** | Shape-IoU Loss | 3,011,043 | 8.19 | 9.07 ms / 110.2 / 125.2 MB | 11.08 ms / 90.3 / 78.4 MB | 60.76 ms / 263.3 / 460.6 MB | 38.30 ms / 417.8 / 257.2 MB |
| **E5b** | MPDIoU Loss | 3,011,043 | 8.19 | 9.07 ms / 110.2 / 125.2 MB | 11.08 ms / 90.3 / 78.4 MB | 60.76 ms / 263.3 / 460.6 MB | 38.30 ms / 417.8 / 257.2 MB |
| **E8** | Combined Model | 3,052,662 | 12.80 | 13.50 ms / 74.1 / 145.9 MB | 16.69 ms / 59.9 / 89.7 MB | 120.00 ms / 133.3 / 805.6 MB | 73.17 ms / 218.7 / 422.0 MB |

### 9.3 Model Export Formats
- **ONNX Export (`.onnx`)**: **NOT FOUND** (zero `.onnx` files in repository).
- **TensorRT Engine (`.engine` / `.trt`)**: **NOT FOUND** (zero `.engine` or `.trt` files in repository).

---

## 10. OPEN QUESTIONS & CONTRADICTIONS

### 10.1 Discrepancies and Contradictions in the Repository

1. **Root `README.md` vs Real Dataset Suite**:
   - **Evidence**: `README.md` lines 1–40 documents a 3-class classification problem (`vessel`, `fiber`, `parenchyma`) achieving $\text{mAP50} > 0.98$ over 100 epochs on synthetic crops.
   - **Contradiction**: The entire actual real-world experimental suite (`src/train_real_suite.py`, `data/real/data.yaml`, `results/real/`, and the IEEE conference paper `results/conference_paper/main.tex`) evaluates a single-class (`cell`, `nc=1`) dense cell detection task on 32 Arabidopsis microscopy images (16 specimens). The root README describes an earlier prototype that was completely superseded.

2. **Perception of "Small Object" Detection vs Empirical Object Dimensions**:
   - **Evidence**: Several reports and outlines characterize plant cell detection as a "small-object localization problem" justifying the P2 stride-4 head.
   - **Contradiction**: Empirical measurements on the actual $640 \times 640$ tiled test split reveal:
     - Median box width: **145.60 px**
     - Median box height: **142.60 px**
     - Median $\sqrt{w \cdot h}$: **139.84 px**
     - Minimum box dimension: **32.80 px**
     - In standard computer vision benchmarks (e.g. COCO: small $< 32 \times 32$ px; medium $32 \times 32$ to $96 \times 96$ px; large $> 96 \times 96$ px), **over 85% of plant cells in this tiled dataset fall into the COCO "large" category ($\ge 96 \times 96$ px)**. This explains why the P2 stride-4 head (E2) caused a +50.9% GFLOPs increase (12.36 vs 8.19) and reduced throughput (72.5 vs 90.3 FPS) while providing less benefit than BiFPN (E3) or Shape-IoU (E5a).

3. **Tile Background Density**:
   - **Evidence**: In `data/real/images/test`, **77 out of 160 tiles (48.1%) contain zero ground-truth cells**. Across the full tiled dataset, 715 out of 1,280 tiles (55.9%) contain zero cell annotations.
   - **Observation**: Over half of the training and evaluation tiles represent slide background/media. The average cell annotation coverage across full micrographs is only **19.84%**.

4. **Duplicate Dataset Folders**:
   - **Evidence**: `real_dataset/` (507 MB) and `data/real_raw/` (507 MB) contain identical folder structures, micrograph TIFFs, and LabelMe JSON files.

5. **Untracked 5.15 GB Virtual Environment**:
   - **Evidence**: `.venv_gpu` is located directly in the repository root and is not listed in `.gitignore`. Any `git add .` or blanket commit will attempt to stage over 5 GB of binary packages.

---

## NOT FOUND / NEEDS MANUAL CONFIRMATION

1. **End-to-End Micrograph Inference CLI**:  
   `predict_tiled()` is implemented in `src/data/tiling.py` (lines 379–438) for full-image tiled inference and cross-tile NMS, but no standalone runnable CLI script exists to execute this pipeline directly on a raw $3840 \times 2160$ micrograph.
2. **Dataset License & Terms of Redistribution**:  
   No formal `LICENSE` file exists in the repository. The data provenance is stated as "TopoInherit", but exact redistribution rights require confirmation.
3. **Formal Unit / Integration Test Suite**:  
   No `tests/` directory or `pytest`/`unittest` files exist in the repository. All functional checks rely on one-off scripts (`run_smoke_audit.py`).
4. **Optimized Inference Artifacts (ONNX / TensorRT)**:  
   No exported `.onnx`, `.engine`, or `.trt` files exist in the repository. All inference runs directly via PyTorch/Ultralytics CUDA tensors.
5. **Web / UI Libraries**:  
   `streamlit`, `flask`, `fastapi`, and `gradio` are **NOT FOUND** in the Python environment. Any upcoming web-based interface task will require installing the appropriate web framework.
