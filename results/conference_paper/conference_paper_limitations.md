# Limitations & Threats to Validity

In accordance with rigorous peer-review and scientific ethics standards, this document comprehensively details the methodological constraints, architectural trade-offs, and threats to empirical validity in this study.

---

## 1. Dataset & Botanical Scope Limitations

1. **Dataset Provenance & Taxonomic Uncertainty**:
   - The repository documentation explicitly exhibits a naming conflict: the dataset text refers to "two major Arabidopsis plant lines", while naming lines "Sansha-5" and "WO115-2" (which are phycological cultivars of the marine macroalga *Pyropia haitanensis*). Without external sequencing validation or author consultation, the exact botanical species cannot be conclusively verified from internal repository files alone.
   - *Impact on Paper*: The paper must transparently report this ambiguity rather than asserting an unverified taxonomic identity.

2. **Sample Size Constraints**:
   - The study evaluates 16 biological specimens across 32 micrographs (3,359 annotated cell instances). While high resolution ($3840 \times 2160$) yields 1,280 tiles ($640 \times 640$), the number of distinct biological plants (16) remains modest.
   - *Threat to Validity*: Biological variation between specimens (e.g. wall thickness, cell packing density) can cause performance variance between the training split (11 specimens) and test split (2 specimens).

3. **Absence of Biological Histological Labels**:
   - The dataset provides geometric cell boundaries and tracking lineage IDs, but **no** cell-type classes (such as vessel, fiber, parenchyma, or epidermal layers).
   - *Impact on Scope*: The model solves binary cell localization (`cell` vs `background`), not multi-class biological tissue segmentation.

4. **Single-Sensor Acquisition**:
   - All micrographs were acquired using an identical brightfield/optical microscopy configuration. Cross-domain generalization to other microscope modalities (fluorescence, confocal, electron microscopy) or varied staining protocols was not evaluated.

---

## 2. Experimental & Statistical Limitations

1. **Single-Seed Full-Epoch Training**:
   - Full 50-epoch training was conducted with a fixed random seed (`seed = 42`). While inference benchmarking utilized 5 interleaved rounds with standard deviations reported, the complete training loop was not repeated across multiple seeds due to computational constraints on the RTX 3050 Laptop GPU.
   - *Threat to Validity*: Metric deltas reflecting small differences (e.g. F1 score deltas $< 0.01$) must be interpreted cautiously without full multi-seed error bars.

2. **Exclusion of Stages E6 and E7**:
   - Stages E6 (ContextRefineClassifier) and E7 (DCNv2 deformable convolutions) were verified via smoke testing, but were intentionally excluded from the primary real-data training order per the established project protocol.
   - *Impact on Scope*: Deformable convolution and post-hoc classification refinement remain un-evaluated on the real dataset and are documented as skipped.

---

## 3. Architectural & Component-Specific Limitations

1. **E4 (BoundaryHead) Confidence Suppression**:
   - While E4 achieved near-perfect precision (0.9038) at $conf=0.25$, its recall dropped drastically (0.0377). The auxiliary boundary loss strongly regularizes the network's confidence distribution, requiring non-standard confidence thresholds ($\sim 0.10$) for balanced detection.

2. **E5b (MPDIoU) Geometric Instability**:
   - MPDIoU performed poorly on dense microscopic cells ($mAP50 = 0.4436$, vs $0.4846$ baseline and $0.6324$ Shape-IoU), suffering premature early stopping at epoch 38. The point-distance formulation across diagonally normalized bounding boxes exhibits numerical instability when bounding boxes have near-square aspect ratios and dense overlaps.

3. **E8 Compound Complexity Overhead**:
   - Although E8 achieves high mAP50 ($0.6043$ at $conf=0.001$), it introduces the highest latency (16.69 ms, 59.9 FPS) and highest memory footprint (422.0 MB VRAM) without outperforming the simpler, lighter E3 (0.6210 mAP50, 91.5 FPS) or E5a (0.6324 mAP50, 90.3 FPS). Compound architectural stacking does not deliver additive improvements.

4. **Framework Method Serialization Fragility**:
   - As revealed during E8 verification, PyTorch's `torch.save` fails to serialize dynamically bound instance methods (`_predict_once`), requiring explicit re-binding upon loading. This fragility highlights the need for standalone subclass architectures rather than monkey-patched instance wrappers in production deployment.
