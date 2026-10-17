# YOLOv8 Plant Xylem Cell Detection

Staged ablation study for xylem cell classification (**vessel**, **fiber**, **parenchyma**) from plant stem light micrographs using YOLOv8.

## Project Overview

This pipeline implements a systematic ablation study (E1–E7) to evaluate the contribution of individual architectural modifications to YOLOv8 for dense, multi-class xylem cell detection. Each experiment is independently runnable via a YAML config and logs results to a common schema for direct comparison.

### Ablation Stages

| Stage | Component | Module | Status |
|:---|:---|:---|:---:|
| **E1** | Stock YOLOv8 baseline | `src/models/yolov8_baseline.py` | ✅ Completed |
| **E2** | Tiling + P2 detection head | `src/models/p2_head.py` + `src/data/tiling.py` | ✅ Completed |
| **E3** | Bidirectional fusion neck | `src/models/fusion_neck.py` | ✅ Completed |
| **E4** | Auxiliary boundary head | `src/models/boundary_branch.py` | ✅ Completed |
| **E5a** | Shape IoU loss | `src/models/losses.py` | ✅ Completed |
| **E5b** | MPDIoU loss | `src/models/losses.py` | ✅ Completed |
| **E6** | Parenchyma context refinement | `src/models/context_refine.py` | ✅ Completed |
| **E7** | Deformable convolutions (DCNv2) | `src/models/dcn_optional.py` | ✅ Completed |

---

## Experimental Results (100-Epoch Sweep)

All ablation stages have completed full 100-epoch training and test evaluation under the standardized fairness protocol:

| Stage | Model / Component | mAP@50 | mAP@50-95 | F1 Score |
|:---|:---|:---:|:---:|:---:|
| **E1** | Stock YOLOv8 Baseline | 0.9781 | 0.9070 | 0.9882 |
| **E2** | Tiling + P2 Head (4-head detector) | 0.9883 | 0.9186 | 0.9911 |
| **E3** | Bidirectional Fusion Neck | 0.9883 | 0.9149 | 0.9919 |
| **E4** | Auxiliary Boundary Head | 0.9883 | 0.9296 | 0.9924 |
| **E5a** | Shape IoU Loss | 0.9883 | **0.9399** | 0.9918 |
| **E5b** | MPDIoU Loss | 0.9883 | 0.9392 | 0.9914 |
| **E6** | Context Refinement | **0.9917** | 0.9388 | **0.9937** |
| **E7** | DCNv2 (Optional) | 0.9781 | 0.9070 | 0.9882 |

All per-experiment metrics, confusion matrices, and model weights are archived in `results/{experiment_name}/seed_42/`.

---

## Hardware Optimization (4GB VRAM Safety)

The pipeline is hardened to execute reliably on low-VRAM GPUs (e.g., 4GB RTX 3050):
- **Dynamic OOM Retry**: Automatically catches CUDA Out-of-Memory exceptions, clears cached VRAM, halves the batch size, and resumes from the last checkpoint.
- **Peak VRAM Profiling**: Per-epoch callback tracking `torch.cuda.max_memory_allocated()`.
- **Gradient Checkpointing**: Integrated into memory-heavy custom modules (E3 Fusion Neck, E4 Boundary Head) to balance compute and memory.
- **Safe Defaults**: Automatic mixed precision (AMP fp16), RAM/disk caching disabled, constrained dataloader workers (`workers=2`), and gradient accumulation support.

---

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Train Individual Stage

```bash
# Train E1 Baseline (full 100 epochs)
python src/train.py --config configs/e1_baseline.yaml

# Train any ablation (e.g., E2 P2 Head, E4 Boundary Head, E5a Shape IoU)
python src/train.py --config configs/e2_tiling_p2.yaml
python src/train.py --config configs/e4_boundary.yaml

# Quick 5-epoch smoke test
python src/train.py --config configs/e1_baseline.yaml --epochs 5
```

### Run Full Ablation Sweep

To sequentially train all ablation experiments (E2 through E6):

```bash
python run_full_sweep.py
```

### Evaluation & Analysis

```bash
# Evaluate model on test set
python src/evaluate.py --model results/e1_baseline/seed_42/weights/best.pt --data data.yaml --split test

# Computational profiling (FPS, GFLOPs, latency)
python src/benchmark.py --model results/e1_baseline/seed_42/weights/best.pt --imgsz 512

# Error categorization (Table IX analysis)
python src/error_analysis.py --model results/e1_baseline/seed_42/weights/best.pt --data data.yaml --split test
```

---

## Real Micrograph Data Integration

To transition from the synthetic dataset to genuine plant micrographs:

1. Place raw high-resolution micrographs in `data/raw/` and corresponding YOLO-format annotations in `data/annotations/`.
2. Run the preparation script:
   ```bash
   python prepare_real_data.py --tile_size 640 --overlap 0.25
   ```
3. Train with the generated `real_data.yaml`:
   ```bash
   python src/train.py --config configs/e2_tiling_p2.yaml --data_yaml real_data.yaml
   ```

---

## Fairness Protocol

All ablation configs (E1–E7) share identical baseline parameters to guarantee comparability:
- Dataset split, random seed(s) (`seed=42`)
- Image/tile resolution (`512×512`)
- Optimizer (`AdamW`), LR schedule (`lr0=0.001`, `lrf=0.01`)
- Epoch count (`100`), batch size (`16`)
- Confidence threshold (`0.25`), NMS IoU threshold (`0.45`)
- Augmentation settings

Only the component under test varies between experiments.

---

## Data Modules

### Specimen-Level Split
```bash
python -m src.data.split --data_dir data/raw --output_dir data/splits --seed 42
```
Ensures no single specimen appears across both training and evaluation partitions.

### Tiling
```bash
python -m src.data.tiling --image_dir data/raw --label_dir data/annotations \
    --output_dir data/tiles --tile_size 512 --overlap 0.25
```

### Boundary Targets (E4)
```bash
python -m src.data.boundary_targets --label_dir labels/train \
    --output_dir data/boundary_maps --img_size 512 --thickness 3 --mode gaussian
```

---

## Results Schema

Each experiment writes to `results/{experiment_name}/seed_{N}/`:
```
results/e1_baseline/seed_42/
├── results.json          # Combined training + evaluation + benchmark
├── metrics.json          # P/R/F1/mAP/per-class AP
├── confusion_matrix.png  # Confusion matrix plot
├── benchmark.json        # params/GFLOPs/latency/FPS/size
├── error_analysis.json   # Categorized error analysis
└── weights/
    └── best.pt           # Best checkpoint weights
```

Multi-seed aggregation summary: `results/{experiment_name}/summary.json`

---

## Project Structure

```
├── configs/                    # Per-experiment YAML configs (E1–E7)
├── data/
│   ├── raw/                    # Original micrographs
│   ├── annotations/            # YOLO-format box labels
│   ├── splits/                 # Generated specimen splits
│   └── tiles/                  # Generated overlapping tiles
├── images/                     # Current dataset images (train/val/test)
├── labels/                     # Current dataset labels (train/val/test)
├── results/                    # Populated experiment metrics & artifacts (E1–E7)
├── src/
│   ├── data/                   # Data processing (split, tiling, boundary targets)
│   ├── models/                 # Model definitions (E1–E7)
│   ├── benchmark.py            # Computational profiling
│   ├── error_analysis.py       # Error categorization
│   ├── evaluate.py             # Metrics computation
│   └── train.py                # Unified training entrypoint
├── data.yaml                   # Ultralytics dataset configuration
├── demo_inference.py           # Quick visual inference runner
├── prepare_real_data.py        # Automated ingestion & tiling for real micrographs
├── requirements.txt            # Python dependencies
└── run_full_sweep.py           # Sequential execution script for full ablation sweep
```
