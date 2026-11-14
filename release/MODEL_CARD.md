# Model Card: YOLOv8 Plant Cell Detection Suite

## 1. Model Details

- **Model Name**: YOLOv8 Plant Cell Detection Suite (Ablation Models E1 through E8)
- **Primary / Recommended Model**: **E5a** (YOLOv8n + Shape-IoU Loss)
- **Lightweight / High-Throughput Variant**: **E3** (P2BiFPN / FusionNeck)
- **Architecture**: Single-class bounding box detector (`class: cell`) based on Ultralytics YOLOv8n with architectural modifications (bidirectional feature fusion neck, auxiliary boundary prediction branch, and bounding box regression loss formulations).
- **Primary Authors**: Sam Ebenezer .P (Karunya Institute of Technology and Sciences)
- **Supervision**: Mrs. M. Denisha, M.E.
- **License**: Research Use Only. Dataset and checkpoint redistribution subject to TopoInherit terms.

---

## 2. Intended Use

- **Intended Biological Task**: Automated localization and counting of plant epidermal/parenchymal cells in high-resolution brightfield optical micrographs.
- **Input Modality**: High-resolution brightfield micrographs ($3840 \times 2160$ px) in 8-bit RGB, 16-bit grayscale, or RGBA (`.tif`, `.tiff`, `.png`, `.jpg`).
- **Inference Pipeline**: Tiled sliding-window inference ($640 \times 640$ window, 25% overlap, step 480 px) with cross-tile Non-Maximum Suppression (NMS).
- **Out-of-Scope Uses**: Clinical diagnostics, unvalidated phenotypic commercial deployment, or species-level taxonomic classification without re-annotation.

---

## 3. Training & Validation Data

- **Dataset**: TopoInherit Micrograph Dataset.
- **Specimens & Micrographs**:
  - Total: 32 full micrographs ($3840 \times 2160$ px) captured across 16 biological plant specimens (2 temporal micrographs per specimen: T0 and T24).
- **Specimen-Disjoint Split**:
  - **Train**: 11 specimens (22 micrographs, 880 tiles of size $640 \times 640$)
  - **Validation**: 3 specimens (6 micrographs, 240 tiles of size $640 \times 640$)
  - **Test (Held-Out)**: 2 specimens (`plant_8` [B6_0808, B6_0809] and `plant_16` [250102-4-1_T0, 250102-4-1_T24], totaling 4 micrographs, 160 tiles of size $640 \times 640$).
- **Tile Properties**:
  - Tile size: $640 \times 640$ pixels.
  - Overlap: 25% fractional overlap (step = 480 pixels).
  - Tiling format: JPEG tiles, normalized bounding box coordinates (`class cx cy w h`).

---

## 4. Evaluation Metrics (Frozen Paper Benchmark)

The following metrics are transcribed directly from `results/conference_paper/final_figures/max_f1_table.csv` evaluated on the 160 held-out test tiles (1,247 ground-truth cell instances, IoU threshold 0.50, confidence 0.001 for curves):

| Model | Architecture Description | Parameters | GFLOPs | Test Tile mAP50 | Max F1 | Optimal Conf | Precision @ Max F1 | Recall @ Max F1 |
|---|---|---|---|---|---|---|---|---|
| **E1** | YOLOv8n Baseline | 3,005,843 | 8.19 | 0.4845 | 0.5230 | 0.1081 | 0.3752 | 0.8629 |
| **E2** | P2 Stride-4 Head (Ablation) | 3,005,843 | 8.19 | 0.5755 | 0.5740 | 0.0611 | 0.4594 | 0.7647 |
| **E3** | FusionNeck (P2BiFPN) | **2,015,781** | **7.28** | **0.6210** | **0.5964** | **0.1141** | **0.5263** | **0.6881** |
| **E4** | Aux BoundaryHead (Ablation) | 3,036,723 | 8.24 | 0.5422 | 0.5538 | 0.0440 | 0.4663 | 0.6816 |
| **E5a** | **Shape-IoU Loss (Recommended)** | **3,005,843** | **8.19** | **0.6325** | **0.6244** | **0.0631** | **0.5109** | **0.8027** |
| **E5b** | MPDIoU Loss (Ablation) | 3,005,843 | 8.19 | 0.4436 | 0.4967 | 0.0771 | 0.3767 | 0.7289 |
| **E8** | Full Composite Architecture | 3,056,455 | 9.87 | 0.6043 | 0.5813 | 0.0891 | 0.5179 | 0.6624 |

*Note: E6 and E7 were exploratory training configurations that diverged during initial sweeps and were excluded from the final ablation series.*

---

## 5. Known Limitations & Technical Gotchas

1. **Partial Ground-Truth Annotation**:
   - The reference TopoInherit labels cover only ~19.8% of the micrograph surface area on average.
   - Approximately 55.9% of training tiles contain zero annotated cells, despite visual inspection confirming the presence of morphological cells in these regions.
   - **Operational Consequence**: Cell counts on full micrographs reflect dense detection across unannotated regions and will exceed manual partial GT counts. Raw cell counts must not be interpreted as absolute biological counts without manual validation.
2. **Confidence Threshold Sensitivity**:
   - Standard YOLO default threshold of `conf=0.25` is excessively high for these models and causes severe under-detection (e.g. recall drops to 0.30–0.40).
   - Inference MUST use the documented per-model thresholds (`E5a: 0.063`, `E3: 0.114`, `E1: 0.108`) or `auto` mode.
3. **Evaluation Population**:
   - The test split comprises only 2 held-out specimens (`plant_8` and `plant_16`). Generalization across diverse plant families or alternative staining protocols is unverified.
4. **Custom Loss Implementation Details**:
   - The bounding box losses designated "Shape-IoU" and "MPDIoU" in this codebase are simplified PyTorch variants implemented directly in `src/model_factory.py`. They capture scale-dependent aspect ratio and diagonal penalty terms but may differ in exact hyperparameter formulation from original literature implementations.
5. **E8 Checkpoint Forward Pass Re-binding**:
   - Checkpoint serialization dropped `_predict_once` dynamic method bindings.
   - Plain loading via `YOLO("E8/best.pt")` silently routes features through the default Ultralytics backbone, dropping test mAP50 from 0.6043 to 0.0165. The `plantcell_tool.models.load_model` routine asserts and dynamically re-binds `_predict_once`.

---

## 6. Ethical Considerations

- This model suite is developed strictly for computational biology and academic plant morphology research.
- Micrograph images do not contain human subjects or identifiable personal data.
