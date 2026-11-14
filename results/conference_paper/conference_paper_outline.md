# IEEE Conference Paper Writing Outline & Evidence Mapping

This document provides a complete section-by-section writing blueprint for an IEEE-format conference paper, detailing the content, empirical evidence, table/figure citations, safe claims, and prohibited claims for each section.

---

### 1. Title
- **Proposed Title**: *High-Resolution Plant Cell Detection Using Lightweight YOLOv8: Architectural Ablation and Computational Efficiency Analysis on Microscopic Time-Lapse Sequences*
- **Alternative Title**: *A Systematic Evaluation of Lightweight YOLOv8 Architectural Enhancements for Single-Class Plant Cell Localization in High-Resolution Microscopy Images*
- **Guidance**: Avoid claiming novelty of standard components (e.g., do not say "A Novel BiFPN-YOLO"). Emphasize the systematic ablation and efficiency analysis.

---

### 2. Abstract
- **Content**: Problem statement (dense plant cell localization in high-resolution micrographs); challenge of multi-temporal specimen leakage; systematic ablation of 6 architectural variants against baseline YOLOv8n; key numerical findings (E3 reduces parameters by 32.9% while achieving highest F1 of 0.4915 and mAP50 of 0.6210; Shape-IoU delivers 0.6324 mAP50; boundary supervision reaches 0.9038 precision); latency/throughput trade-offs on an RTX 3050 GPU.
- **Evidence**: `conference_paper_results.csv`, `conference_paper_efficiency.csv`.
- **Safe Claims**: Quantified performance gains and parameter reductions under identical training protocol.
- **Prohibited Claims**: Do not claim biological tissue segmentation or invention of constituent modules.

---

### 3. Keywords
- `Plant Cell Detection`, `Microscopy Image Analysis`, `YOLOv8`, `Architectural Ablation`, `BiFPN Feature Fusion`, `Shape-IoU`, `Edge Deployment`, `Specimen-Level Partitioning`.

---

### 4. Introduction
- **Content**: Importance of automated cell localization for high-throughput plant lineage tracking; difficulties in microscopic imaging (high resolution $3840 \times 2160$, shared cell boundaries, variable contrast); limitation of standard anchor-free detectors designed for natural images; research motivation for evaluating fine-scale heads, cross-scale fusion, and geometric losses under a controlled hardware budget.
- **Evidence**: Resolution ($3840 \times 2160$) and density (3,359 cells across 32 micrographs).
- **Figures to Reference**: Figure 1 (Tiling Pipeline), Figure 2 (Architecture Taxonomy).
- **Safe Claims**: Systematic controlled study; leak-free evaluation.
- **Prohibited Claims**: "First automated cell detection paper."

---

### 5. Related Work
- **Content**:
  - *Deep Learning in Plant Microscopy*: U-Net, Mask R-CNN, StarDist, and YOLO adaptations in bio-imaging.
  - *Lightweight Object Detection*: Evolution of YOLO models (YOLOv5 to YOLOv8); trade-offs between depth, width, and resolution.
  - *Architectural Enhancements*: Stride-4 P2 heads for small objects; BiFPN weighted feature fusion; auxiliary multi-task boundary supervision; bounding-box geometric regression losses (CIoU, Shape-IoU, MPDIoU).
- **Safe Claims**: Acknowledging established provenance of all compared techniques.
- **Prohibited Claims**: Attributing P2, BiFPN, or Shape-IoU as our original algorithmic discoveries.

---

### 6. Dataset and Problem Formulation
- **Content**:
  - 32 micrographs, 16 biological specimen groups observed across 0h and 24h.
  - 3,359 annotated cell instances.
  - **Single-Class Formulation**: $nc=1$, class 0 = cell. Explain why cell tracking numbers (1, 2, 34-1) are treated as individual instances rather than semantic classes.
  - **Dataset Provenance**: State the TopoInherit dataset provenance, noting the documentation text ("Arabidopsis plant lines") alongside the strain names ("Sansha-5" and "WO115-2"), stating that external curator verification is required for formal taxonomic assignment.
- **Evidence**: [data/real_raw/README.md](file:///e:/Yolov8_Plant_Cell_Detection/data/real_raw/README.md), [results/inspection/real_dataset_manifest.json](file:///e:/Yolov8_Plant_Cell_Detection/results/inspection/real_dataset_manifest.json).
- **Safe Claims**: Precise physical image count, instance counts, and polygon conversion methodology.
- **Prohibited Claims**: Claiming 3 biological classes (vessel/fiber/parenchyma).

---

### 7. Proposed Experimental Framework & Specimen-Level Splitting
- **Content**:
  - Detailed explanation of temporal autocorrelation between 0h and 24h paired micrographs.
  - Specimen-level split: 11 train specimens (880 tiles), 3 validation specimens (240 tiles), 2 test specimens (160 tiles).
  - Overlapping tiling: $640 \times 640$, 25% overlap, filtering $\ge 0.30$ visible area.
- **Evidence**: [data/splits/](file:///e:/Yolov8_Plant_Cell_Detection/data/splits/), [src/data/prepare_real_dataset.py](file:///e:/Yolov8_Plant_Cell_Detection/src/data/prepare_real_dataset.py).
- **Figures to Reference**: Figure 1.
- **Safe Claims**: Strict disjointness of specimens across splits.

---

### 8. YOLOv8 Baseline (E1)
- **Content**: Baseline YOLOv8n architecture modified for $nc=1$. Pretrained transfer protocol from `yolov8n.pt` (319 transferred tensors, 36 reinitialized head parameters).
- **Evidence**: `E1.json`, parameter verification (3,011,043 params, 8.1942 GFLOPs).
- **Tables to Reference**: Table I (Master Results Table).

---

### 9. Architectural Modifications (E2–E8)
- **Content**:
  - **E2**: P2 stride-4 head (160×160 resolution level).
  - **E3**: Bidirectional FusionNeck (BiFPN-style weighted feature fusion, 128 channels).
  - **E4**: Auxiliary BoundaryHead attached to P3 (BCE + soft Dice loss, train-only).
  - **E5a / E5b**: Swappable IoU loss functions (Shape-IoU vs MPDIoU).
  - **E8**: Full combined model (P2 + FusionNeck + BoundaryHead + Shape-IoU).
- **Evidence**: [src/model_factory.py](file:///e:/Yolov8_Plant_Cell_Detection/src/model_factory.py).
- **Figures to Reference**: Figure 2.

---

### 10. Experimental Methodology
- **Content**: Hardware specifications (RTX 3050 Laptop GPU, 4GB VRAM); hyperparameters (imgsz=640, batch=16, epochs=50, patience=20, seed=42, AMP=True).
- **Evidence**: `args.yaml` and training logs in [runs/detect/results/real/](file:///e:/Yolov8_Plant_Cell_Detection/runs/detect/results/real/).

---

### 11. Evaluation Metrics
- **Content**: Mathematical definitions of Precision, Recall, F1 score, mAP@0.50, and mAP@0.50:0.95. Explain why both $conf=0.25$ (operational filtering) and $conf=0.001$ (complete PR integration) are reported.
- **Evidence**: `reeval_test_q1.json`.

---

### 12. Computational Benchmarking
- **Content**: GPU latency and throughput benchmarking protocol (CUDA events, 100 warm-up, 300 timed passes, 5 interleaved rounds). Reporting median ± standard deviation. Empirical noise baseline between E1 and E4 ($< 0.56\%$).
- **Evidence**: [results/paper/compute_table.csv](file:///e:/Yolov8_Plant_Cell_Detection/results/paper/compute_table.csv), [results/paper/implementation_verification.md](file:///e:/Yolov8_Plant_Cell_Detection/results/paper/implementation_verification.md).
- **Tables to Reference**: Table II (Efficiency Table).

---

### 13. Results
- **Content**: Presentation of test split results across all 7 trained stages.
- **Tables to Reference**: Table I (Master Results), Table II (Computational Efficiency).
- **Figures to Reference**: Figure 3 (Qualitative Detections), Figure 4 (PR Curves).

---

### 14. Ablation Study
- **Content**: Component-by-component delta analysis relative to E1:
  - Impact of P2 resolution (+0.0910 mAP50, +0.2010 Precision, +4.16 GFLOPs).
  - Impact of BiFPN fusion (+0.1364 mAP50, -32.9% parameters, +1.2 FPS).
  - Impact of BoundaryHead (+0.4907 Precision, extreme regularization).
  - Shape-IoU vs MPDIoU (+0.1478 mAP50 vs -0.0410 mAP50).
  - Combined model dynamics (E8: 0.6043 mAP50, 0.7317 Precision).
- **Evidence**: `conference_paper_ablation.csv`.
- **Figures to Reference**: Figure 5 (Pareto Frontier).

---

### 15. Discussion
- **Content**:
  - Why BiFPN feature fusion is Pareto-optimal for dense biological cells.
  - The trade-off between boundary supervision and confidence score distribution.
  - The failure mode of MPDIoU on rectangular microscopy tiles.
  - Why combining all modifications does not produce a simple additive sum of individual gains.
- **Safe Claims**: Explanations grounded directly in measured statistics.

---

### 16. Limitations
- **Content**:
  - Single-class cell localization without histological sub-typing.
  - Small biological sample size (16 specimens, 32 micrographs).
  - Evaluated on a single camera/microscope acquisition setup.
  - Absence of multi-seed statistical dispersion across full 50-epoch training.
  - Botanical identity documentation conflict requiring external confirmation.
- **Safe Claims**: Transparent reporting of scope and limitations.

---

### 17. Conclusion
- **Content**: Summarize that lightweight YOLOv8 modifications provide significant gains for automated plant cell detection; highlight E3 as the most efficient model and Shape-IoU as the superior regression objective; conclude that specimen-level partitioning is essential for credible biological computer vision.

---

### 18. Future Work
- **Content**: Multi-seed variance quantification; self-supervised pretraining on unlabeled micrographs; adaptive confidence threshold calibration for boundary-regularized models.

---

### 19. References
- Cite YOLOv8 (Jocher et al.), BiFPN (Tan et al.), Shape-IoU (Zhang et al.), MPDIoU (Ma et al.), DCNv2 (Zhu et al.), and key plant cell tracking literature.
