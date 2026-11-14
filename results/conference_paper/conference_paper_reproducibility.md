# Reproducibility Checklist & Environment Specification

This document provides a comprehensive reproducibility specification to ensure full auditability, repeatability, and deterministic verification of all experimental findings.

---

## 1. System & Hardware Specifications

| Component | Specification | Verification Command / Source |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / PowerShell | System Environment |
| **Host Processor** | 12th Gen Intel Core i5-12450H | Device Manager |
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop GPU | `torch.cuda.get_device_name(0)` |
| **GPU Dedicated VRAM**| 4,096 MiB (4.00 GB) | `torch.cuda.get_device_properties(0).total_memory` |
| **CUDA Driver Version**| Driver 560.94 / CUDA 12.6 Driver API | `nvidia-smi` |
| **NVIDIA Compute** | sm_86 (Ampere Architecture) | PyTorch Device Capabilities |

---

## 2. Software & Python Environment

| Package | Exact Version | Installation Source / Index |
| :--- | :--- | :--- |
| **Python** | 3.12.10 (64-bit) | Active venv: `.venv_gpu` |
| **PyTorch (`torch`)** | `2.6.0+cu124` | PyTorch Official Index (`https://download.pytorch.org/whl/cu124`) |
| **TorchVision** | `0.21.0+cu124` | PyTorch Official Index (`https://download.pytorch.org/whl/cu124`) |
| **Ultralytics** | `8.4.182` | PyPI / pip |
| **THOP** | `0.1.1.post2209072238` | PyPI / pip |
| **OpenCV (`cv2`)** | `4.11.0` | PyPI / pip (`opencv-python-headless`) |
| **NumPy** | `2.5.4` | PyPI / pip |
| **Pandas** | `2.2.3` | PyPI / pip |

---

## 3. Training Protocol & Hyperparameters

All models were trained under an identical fairness protocol:

- **Input Resolution (`imgsz`)**: $640 \times 640$ pixels
- **Batch Size**: 16 (verified safe under 4 GB VRAM; peak allocation $< 900$ MB)
- **Epochs**: 50 maximum epochs
- **Early Stopping (`patience`)**: 20 epochs
- **Optimizer**: `auto` (Ultralytics default SGD/AdamW auto-selection)
- **Automatic Mixed Precision (`amp`)**: Enabled (`True`)
- **Random Seed**: `42` (`torch.manual_seed(42)`, `np.random.seed(42)`)
- **Number of DataLoader Workers**: `0` (in-process for deterministic execution on Windows)
- **Pretrained Transfer Source**: `yolov8n.pt` (transferred backbone/pyramid tensors, reinitialized head for $nc=1$)

---

## 4. Dataset Partitioning Specification

- **Root Source**: [data/real_raw/](file:///e:/Yolov8_Plant_Cell_Detection/data/real_raw/) (32 micrographs, 3,359 cell polygons)
- **Tiling Settings**: $640 \times 640$ tile size, 25% overlap (480 px step), area fraction threshold $\ge 0.30$
- **Splits Manifests**:
  - **Train**: [data/splits/train_specimens.txt](file:///e:/Yolov8_Plant_Cell_Detection/data/splits/train_specimens.txt) (11 specimens: `plant_1..5, 7, 9..13`, 880 tiles, 5,007 annotations)
  - **Val**: [data/splits/val_specimens.txt](file:///e:/Yolov8_Plant_Cell_Detection/data/splits/val_specimens.txt) (3 specimens: `plant_6, 14, 15`, 240 tiles, 1,502 annotations)
  - **Test**: [data/splits/test_specimens.txt](file:///e:/Yolov8_Plant_Cell_Detection/data/splits/test_specimens.txt) (2 specimens: `plant_8, 16`, 160 tiles, 1,247 annotations)
- **Data Configuration File**: [data/real/data.yaml](file:///e:/Yolov8_Plant_Cell_Detection/data/real/data.yaml)

---

## 5. Evaluation Protocol

- **Dataset Split**: Strictly evaluated on the held-out `test` split (`data/real/images/test`, 160 tiles, 1,247 instances)
- **Evaluation Settings**:
  - Full PR integration (for mAP50 and mAP50-95): `conf = 0.001`, `iou = 0.45`, `max_det = 300`
  - Operational point metrics (for Precision, Recall, F1): `conf = 0.25`, `iou = 0.45`, `max_det = 300`

---

## 6. Repository File Directory & Checkpoints

### 6.1 Trained Checkpoints
- **E1**: [runs/detect/results/real/E1/weights/best.pt](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E1/weights/best.pt)
- **E2**: [runs/detect/results/real/E2/weights/best.pt](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E2/weights/best.pt)
- **E3**: [runs/detect/results/real/E3/weights/best.pt](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E3/weights/best.pt)
- **E4**: [runs/detect/results/real/E4/weights/best.pt](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E4/weights/best.pt)
- **E5a**: [runs/detect/results/real/E5a/weights/best.pt](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E5a/weights/best.pt)
- **E5b**: [runs/detect/results/real/E5b/weights/best.pt](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E5b/weights/best.pt)
- **E8**: [runs/detect/results/real/E8/weights/best.pt](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E8/weights/best.pt)

### 6.2 Source Code Entrypoints
- **Model Factory**: [src/model_factory.py](file:///e:/Yolov8_Plant_Cell_Detection/src/model_factory.py)
- **Data Tiling & Preparation**: [src/data/prepare_real_dataset.py](file:///e:/Yolov8_Plant_Cell_Detection/src/data/prepare_real_dataset.py)
- **Unified Training Suite**: [src/train_real_suite.py](file:///e:/Yolov8_Plant_Cell_Detection/src/train_real_suite.py)
- **Re-Evaluation & Confidence Profiling**: [src/reeval_test_conf.py](file:///e:/Yolov8_Plant_Cell_Detection/src/reeval_test_conf.py)
- **Rigorous GPU Benchmark**: [benchmark_gpu_rigorous.py](file:///e:/Yolov8_Plant_Cell_Detection/benchmark_gpu_rigorous.py)

### 6.3 Machine-Readable Evidence Files
- **Per-Stage Results**: [results/real/](file:///e:/Yolov8_Plant_Cell_Detection/results/real/) (`E1.json` through `E8.json`)
- **Suite Summary**: [results/paper/experiment_summary.json](file:///e:/Yolov8_Plant_Cell_Detection/results/paper/experiment_summary.json)
- **GPU Benchmark Raw Telemetry**: [results/paper/rigorous_gpu_benchmark_results.json](file:///e:/Yolov8_Plant_Cell_Detection/results/paper/rigorous_gpu_benchmark_results.json)
- **Re-evaluation Data**: [results/paper/reeval_test_q1.json](file:///e:/Yolov8_Plant_Cell_Detection/results/paper/reeval_test_q1.json), [results/paper/reeval_test_q2.json](file:///e:/Yolov8_Plant_Cell_Detection/results/paper/reeval_test_q2.json)
