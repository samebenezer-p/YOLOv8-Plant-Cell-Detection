# Recommended Figures for IEEE Conference Paper

This document outlines the visual evidence package, identifying primary and supplementary figures for the conference paper submission, including figure numbers, captions, analytical purpose, and source file paths.

---

## Master Figure List

### Figure 1: Specimen-Level Tiling & Coordinate Remapping Pipeline
- **Title**: High-Resolution Microscopy Tiling and Specimen-Preserved Coordinate Transformation Pipeline.
- **Purpose**: Illustrates the challenge of detecting microscopic plant cells in massive $3840 \times 2160$ micrographs without resizing degradation, showing the $640 \times 640$ sliding window (step=480 px, 25% overlap) and the strict specimen-level partition into Train, Validation, and Test sets.
- **Demonstrates**: Elimination of spatial aliasing, handling of boundary-intersecting polygons with area-fraction filtering ($\ge 0.30$), and guarantee of zero temporal leakage between 0h and 24h observations.
- **Stage**: Data Pipeline
- **Source Files**:
  - Raw Image: [data/real_raw/plant_8/img/B6_0808.tif](file:///e:/Yolov8_Plant_Cell_Detection/data/real_raw/plant_8/img/B6_0808.tif)
  - Tile Output: [data/real/images/test/B6_0808_tile0000.jpg](file:///e:/Yolov8_Plant_Cell_Detection/data/real/images/test/B6_0808_tile0000.jpg)
  - Pipeline Code: [src/data/prepare_real_dataset.py](file:///e:/Yolov8_Plant_Cell_Detection/src/data/prepare_real_dataset.py)

---

### Figure 2: Architectural Ablation Taxonomy
- **Title**: Unified Lightweight Architectural Modifications to YOLOv8 for Dense Cell Localization.
- **Purpose**: Schematic diagram comparing the baseline YOLOv8n (E1) against the fine-scale P2 stride-4 head (E2), BiFPN Bidirectional FusionNeck (E3), auxiliary BoundaryHead branch (E4), and the combined E8 architecture.
- **Demonstrates**: Structural differences in feature pyramid levels (P2, P3, P4, P5), bidirectional feature routing, and the train-time boundary supervision branch.
- **Stage**: E1, E2, E3, E4, E8
- **Source Files**:
  - Model Factory: [src/model_factory.py](file:///e:/Yolov8_Plant_Cell_Detection/src/model_factory.py)
  - Architectural Modules: [src/models/fusion_neck.py](file:///e:/Yolov8_Plant_Cell_Detection/src/models/fusion_neck.py), [src/models/boundary_branch.py](file:///e:/Yolov8_Plant_Cell_Detection/src/models/boundary_branch.py)

---

### Figure 3: Qualitative Test Detections Across Ablation Stages
- **Title**: Visual Comparison of Cell Localization on Held-Out Test Specimens Across Key Architectures.
- **Purpose**: 4-panel comparison showing predictions on identical held-out test tiles from `plant_8` (Sansha-5) and `plant_16` (WO115-2).
- **Demonstrates**:
  - **E1 Baseline**: Balanced coverage with occasional overlapping duplicate boxes.
  - **E2 (P2 Head)**: Tighter bounding box delineation in high-density cell clusters.
  - **E3 (FusionNeck)**: Clean, high-confidence detections with fewer false positives, matching 33% parameter reduction.
  - **E4 (BoundaryHead)**: Ultra-precise box adherence along cell walls.
- **Stage**: E1, E2, E3, E4, E8
- **Source Files**:
  - E1 Prediction: [runs/detect/results/real/E1/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E1/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg)
  - E2 Prediction: [runs/detect/results/real/E2/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E2/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg)
  - E3 Prediction: [runs/detect/results/real/E3/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E3/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg)
  - E4 Prediction: [runs/detect/results/real/E4/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E4/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg)
  - E8 Prediction: [runs/detect/results/real/E8/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E8/representative_predictions/test_pred_sample_3_B6_0808_tile0000.jpg)

---

### Figure 4: Precision-Recall (PR) Curves & F1-Confidence Dynamics
- **Title**: Precision-Recall and F1 Performance Curves on the Held-Out Test Set.
- **Purpose**: Displays the complete PR curves across all 7 evaluated stages, illustrating the area under curve (mAP50) differences.
- **Demonstrates**: E5a and E3 achieving superior PR frontiers compared to baseline, while highlighting E4's high precision at low confidence thresholds.
- **Stage**: E1, E2, E3, E4, E5a, E5b, E8
- **Source Files**:
  - E1 PR Curve: [runs/detect/results/real/E1/BoxPR_curve.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E1/BoxPR_curve.png)
  - E2 PR Curve: [runs/detect/results/real/E2/BoxPR_curve.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E2/BoxPR_curve.png)
  - E3 PR Curve: [runs/detect/results/real/E3/BoxPR_curve.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E3/BoxPR_curve.png)
  - E4 PR Curve: [runs/detect/results/real/E4/BoxPR_curve.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E4/BoxPR_curve.png)
  - E5a PR Curve: [runs/detect/results/real/E5a/BoxPR_curve.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E5a/BoxPR_curve.png)
  - E8 PR Curve: [runs/detect/results/real/E8/BoxPR_curve.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E8/BoxPR_curve.png)

---

### Figure 5: Accuracy vs. Computational Cost Pareto Frontier
- **Title**: Computational Efficiency Trade-Off: mAP50 vs. Inference Latency (ms) and Model Parameters.
- **Purpose**: Scatter plot mapping mAP50 against inference latency (ms) and parameter count, identifying the Pareto-optimal frontier.
- **Demonstrates**: **E3 (BiFPN FusionNeck)** forms the optimal trade-off point (highest throughput, lowest parameters, high mAP), while E2 and E8 incur substantial latency penalties for marginal accuracy changes.
- **Stage**: All stages
- **Source Files**:
  - Efficiency Data: [results/conference_paper/conference_paper_efficiency.csv](file:///e:/Yolov8_Plant_Cell_Detection/results/conference_paper/conference_paper_efficiency.csv)
  - Benchmark Summary: [results/paper/compute_table.csv](file:///e:/Yolov8_Plant_Cell_Detection/results/paper/compute_table.csv)

---

### Figure 6: Training Convergence & Loss Dynamics
- **Title**: Training and Validation Loss Curves Across 50 Epochs.
- **Purpose**: Compares convergence trajectories of bounding-box loss, classification loss, and distribution focal loss.
- **Demonstrates**: Stable optimization trajectories across E1, E2, E3, and E5a; early stopping trigger in E5b (epoch 38) and E4 (epoch 37).
- **Stage**: E1 through E8
- **Source Files**:
  - E1 Training Curves: [runs/detect/results/real/E1/results.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E1/results.png)
  - E2 Training Curves: [runs/detect/results/real/E2/results.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E2/results.png)
  - E3 Training Curves: [runs/detect/results/real/E3/results.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E3/results.png)
  - E8 Training Curves: [runs/detect/results/real/E8/results.png](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/E8/results.png)
