# High-Resolution YOLOv8-Based Plant Cell Detection and Architectural Ablation on Microscopic Sequences

**Sam Ebenezer .P$^1$, Mrs. M. Denisha, M.E.$^2$**  
*$^1,^2$Department of Computer Science and Engineering, Karunya Institute of Technology and Sciences, Coimbatore, India*

---

### Abstract
Automated cell detection in high-resolution plant microscopy is vital for quantitative phenotyping and lineage tracking. However, standard object detectors designed for natural images struggle with dense, micro-scale morphological boundaries, high cell densities, and tight hardware constraints on edge computing devices. Furthermore, naive patch-based evaluations frequently suffer from optimistic bias due to specimen-level data leakage across time points. In this work, we conduct a systematic, experimentally controlled ablation of lightweight architectural and loss modifications to the YOLOv8n detector for single-class plant cell localization ($3840 \times 2160$ resolution micrographs tiled to $640 \times 640$). We evaluate: (i) a stride-4 ($P_2$) detection head for high-resolution feature preservation, (ii) a bidirectional multiscale feature fusion neck (BiFPN-style FusionNeck), (iii) an auxiliary boundary-supervision head, and (iv) bounding-box regression loss variants including Shape-IoU and MPDIoU. Crucially, all models are evaluated on a held-out, specimen-disjoint test split to prevent temporal leakage between serial observations.

Our empirical results on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) reveal three primary findings: First, the Bidirectional FusionNeck (E3) achieves Pareto-optimal efficiency, reducing parameter count by 32.9% (from 3.01M to 2.02M) and GFLOPs by 11.1% (from 8.19 to 7.28) while increasing inference throughput to 91.5 FPS and yielding the highest $F_1$ score (0.4915 at $\tau=0.25$) and a +0.1364 mAP50 improvement over the baseline (0.6210 vs. 0.4846 at $\tau=0.001$). Second, Shape-IoU (E5a) achieves the highest overall accuracy ($\text{mAP50}=0.6324$, $\text{mAP50-95}=0.5579$) with zero additional inference parameters or latency penalty over stock YOLOv8n. Third, combining all modules into a composite architecture (E8) introduces optimization conflicts and computational overhead (12.80 GFLOPs, 59.9 FPS, $\text{mAP50}=0.6043$), failing to surpass the lighter isolated configurations. Finally, we document an engineering investigation into a checkpoint deserialization defect encountered when deploying dynamic PyTorch modules, outlining the required forward-pass restoration protocol.

**Keywords:** Plant cell detection, YOLOv8, high-resolution microscopy, architectural ablation, Bidirectional Feature Pyramid, Shape-IoU, specimen-level splitting, edge inference.

---

## I. INTRODUCTION

Microscopic analysis of plant cells provides quantitative insight into developmental dynamics, cell division rates, morphological plasticity, and physiological stress responses. High-resolution optical microscopy captures expansive fields of view ($3840 \times 2160$ pixels and higher), encompassing hundreds to thousands of contiguous, densely clustered cell instances per frame. Historically, cell localization has relied on classical image processing pipelines such as watershed segmentation, morphological filtering, or intensity thresholding. While computationally modest, these heuristics break down under heterogeneous optical illumination, out-of-focus blurs, irregular cell morphologies, and low signal-to-noise ratios.

Modern deep-learning-based object detectors—predominantly convolutional and transformer-based architectures—have demonstrated remarkable localization capabilities in bio-imaging. In particular, the You Only Look Once (YOLO) family of one-stage detectors has gained widespread adoption due to its balance between detection speed and localization accuracy. However, standard object detectors are predominantly calibrated for natural scene datasets (e.g., MS COCO), where targets occupy variable, often large visual angles, backgrounds are semantically diverse, and instances rarely appear in dense, lattice-like configurations. In microscopy, cell targets present unique challenges:
1. **Dense Clustering and Shared Cell Walls:** Cells frequently share boundaries, where sub-pixel shifts lead to severe bounding-box overlap and ambiguous intersection-over-union (IoU) computations.
2. **Scale Homogeneity and Fine Geometry:** Unlike natural scenes featuring extreme multiscale variance (e.g., person vs. airplane), plant cells within a field of view exhibit relatively uniform physical sizes, rendering deeper downsampling strides (e.g., stride 32) functionally redundant while discarding spatial boundary cues.
3. **Data Leakage in Microscopic Time-Series:** Microscopic studies often record the same physical specimen across temporal intervals (e.g., 0h and 24h). Naive random patch splitting partitions tiles from the same specimen across training and test splits, introducing catastrophic memorization of specimen-specific background artifacts and yielding inflated performance metrics.

To address these challenges under strict hardware constraints (specifically, 4 GB GPU edge workstations), this paper presents a controlled empirical investigation of lightweight architectural and regression loss enhancements built upon the YOLOv8n backbone. Rather than claiming the invention of established modules, our scientific contribution lies in a rigorous, leakage-free ablation of spatial heads, feature fusion topologies, boundary supervision, and bounding-box geometries.

The primary contributions of this work are:
- **Specimen-Disjoint Microscopic Evaluation:** We implement a strict specimen-level data partitioning protocol across 32 ultra-high-resolution micrographs (3,359 cell annotations, 16 biological groups), eliminating temporal and specimen-level data leakage between 0h and 24h observations.
- **Systematic Architectural Ablation:** We evaluate isolated modifications to YOLOv8n, isolating the individual empirical impacts of a high-resolution stride-4 ($P_2$) head, a Bidirectional FusionNeck (BiFPN-style routing), and an auxiliary boundary prediction head.
- **Bounding-Box Geometry Analysis:** We compare standard Complete-IoU (CIoU) against Shape-IoU and Minimum Point Distance IoU (MPDIoU), demonstrating that shape-sensitive geometric penalties significantly boost dense cell localization without any inference overhead.
- **Computational Profiling and Pareto Analysis:** We conduct multi-repeat CUDA event profiling on physical hardware (RTX 3050 Laptop GPU), characterizing parameter count, GFLOPs, latency, throughput (FPS), and peak VRAM to identify Pareto-optimal edge architectures.
- **Engineering Verification Insight:** We document a critical checkpoint deserialization defect encountered when evaluating dynamic neural modules (E8), detailing the failure mode and the verification protocol required to ensure reproducible inference.

---

## II. RELATED WORK

### A. YOLO Architectures in Bio-Imaging
The YOLO architecture family has evolved from coupled grid-based regressors into decoupled, anchor-free detection frameworks. YOLOv8 incorporates a CSPDarknet backbone with Cross-Stage Partial with 2 Convolutions (C2f) modules and an anchor-free decoupled head separating bounding-box regression from classification. While YOLOv8n achieves high throughput on mobile and edge devices, its default feature pyramid downsamples inputs across strides 8 ($P_3$), 16 ($P_4$), and 32 ($P_5$). In microscopic cell detection, where cell diameters often span only 15–40 pixels, the spatial features of target cells are substantially degraded or entirely lost at stride 32.

### B. High-Resolution and Small-Object Detection
To detect micro-scale objects, small-object detection literature commonly adopts low-level feature taps. Introducing a stride-4 ($P_2$) detection head incorporates low-level textural and edge representations from the backbone into the detection head. While this approach preserves spatial resolution for small targets, it dramatically increases the spatial dimensions of the intermediate feature maps ($160 \times 160$ for a $640 \times 640$ input), expanding computational complexity (GFLOPs) and activation memory.

### C. Multiscale Feature Fusion and BiFPN
Standard Feature Pyramid Networks (FPN) and Path Aggregation Networks (PANet) propagate semantic features top-down and spatial features bottom-up. However, PANet treats all input paths uniformly without learnable weight allocation or cross-scale skip connections. The Bidirectional Feature Pyramid Network (BiFPN), introduced in EfficientDet, integrates bidirectional cross-scale connections and weighted feature fusion, allowing the network to emphasize salient spatial resolutions while pruning redundant layers.

### D. Bounding-Box Regression Losses
Bounding-box regression has advanced from $\ell_1/\ell_2$ coordinate loss to IoU-based formulations. Generalized IoU (GIoU), Distance-IoU (DIoU), and Complete-IoU (CIoU) incorporate spatial overlap, center point displacement, and aspect ratio consistency. In dense microscopy, adjacent cells share similar aspect ratios and center proximities. Recent formulations such as Shape-IoU introduce shape distance scaling based on target width-to-height proportions, whereas MPDIoU directly minimizes the squared distance between top-left and bottom-right bounding-box corners. Their relative efficacy on densely clustered, rectangular-to-ellipsoidal plant cells remains insufficiently characterized under rigorous experimental control.

---

## III. DATASET AND PROBLEM FORMULATION

### A. Taxonomic Context and Discrepancy Disclosure
The experimental evaluation is conducted on the TopoInherit plant microscopy dataset. The repository documentation contains an unresolved taxonomic identity discrepancy: textual metadata refers to the specimens as "Arabidopsis plant lines," whereas the specific cultivar designations—Sansha-5 and WO115-2—are documented in phycological literature as cultivated strains of the red seaweed *Pyropia haitanensis* (formerly *Porphyra haitanensis*). In accordance with strict academic reporting standards, we do not arbitrarily resolve this conflict; we report the dataset as comprising high-resolution cell micrographs associated with the Sansha-5 and WO115-2 lines, noting that definitive biological assignment requires external curator confirmation.

### B. Dataset Characteristics
The dataset consists of 32 high-resolution brightfield micrographs ($3840 \times 2160$ pixels) across 16 physical specimen groups (`plant_1` to `plant_16`), imaged at two temporal milestones (0h initial culture and 24h post-treatment). Across the 32 images, a total of 3,359 cell instances are manually delineated via polygon annotations. The original LabelMe metadata contains numeric tracking/lineage identifiers (e.g., `"1"`, `"2"`, `"34-1"`). We explicitly note that these numeric labels represent individual cell tracking indices rather than histological tissue classifications (e.g., vessel, parenchyma, fiber).

### C. Problem Formulation
We formulate cell detection as a single-class object localization problem:
$$\mathcal{Y} = \{(b_i, c_i)\}_{i=1}^{M}, \quad c_i = 0 \, (\text{cell}), \quad b_i = [x_c, y_c, w, h]^T$$
where $(x_c, y_c)$ denote normalized bounding-box center coordinates, and $(w, h)$ denote normalized width and height. No multi-class classification is claimed or evaluated ($n_c = 1$).

### D. Specimen-Level Partitioning
To eliminate data leakage, we enforce a strict specimen-disjoint split:
* **Training Set:** 11 specimens (`plant_1`–`plant_5`, `plant_7`, `plant_9`–`plant_13`), comprising 22 micrographs, 880 tiles, and 5,007 cell instances.
* **Validation Set:** 3 specimens (`plant_6`, `plant_14`, `plant_15`), comprising 6 micrographs, 240 tiles, and 1,502 cell instances.
* **Held-Out Test Set:** 2 specimens (`plant_8`, `plant_16`), comprising 4 micrographs, 160 tiles, and 1,247 cell instances.

The intersection of physical specimens between any two partitions is strictly empty:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

### E. High-Resolution Tiling Pipeline
Directly resizing $3840 \times 2160$ micrographs to $640 \times 640$ induces an extreme downsampling factor ($6\times$), reducing micro-scale cell instances ($20 \times 20$ pixels) to sub-pixel or near-aliased blobs ($3.3 \times 3.3$ pixels). To preserve spatial fidelity, we implement a uniform sliding-window tiling strategy:
* Tile dimensions: $640 \times 640$ pixels.
* Step size: $480$ pixels (stride overlap factor: 0.25).
* Minimum visible area fraction: $\alpha_{\text{min}} = 0.30$. Polygon boundaries intersecting tile edges are clipped to the bounding frame; instances retaining less than 30% of their original area are pruned to avoid training on truncated edge artifacts.

---

## IV. PROPOSED EXPERIMENTAL FRAMEWORK

We formulate a modular ablation framework spanning seven primary evaluated configurations (E1–E5b, E8), with two architectural variants (E6, E7) intentionally excluded per protocol.

```
       [ Input Tile: 640 x 640 x 3 ]
                     │
            ┌────────┴────────┐
            ▼                 ▼
     [ CSPDarknet-n ]   [ CSPDarknet-n ]
      (Strides 8,16,32)  (Strides 4,8,16,32)
            │                 │
     [ E1: PANet Neck ] [ E2: P2-PANet Neck ]
     [ E3: BiFPN Neck ] [ E8: P2+BiFPN Neck ]
            │                 │
            ▼                 ▼
     [ Detection Head ] [ Detection Head ]
      (Loss: CIoU,       (Loss: Shape-IoU,
       Shape-IoU,         Boundary Auxiliary)
       MPDIoU)
```

### A. E1 — Stock YOLOv8n Baseline
The baseline uses standard YOLOv8n with a CSPDarknet backbone and a PANet feature neck. Spatial features are extracted at strides 8 ($P_3$, $80 \times 80$), 16 ($P_4$, $40 \times 40$), and 32 ($P_5$, $20 \times 20$). Bounding-box regression is supervised using Complete-IoU (CIoU) and Distribution Focal Loss (DFL).

### B. E2 — Stride-4 ($P_2$) Detection Head
To preserve ultra-fine spatial resolution, E2 incorporates an additional low-level feature tap from backbone stage 2 ($C_2$, stride 4, $160 \times 160$ resolution). The neck is extended with an extra top-down and bottom-up pathway, delivering four multiscale feature maps ($P_2, P_3, P_4, P_5$) to a 4-scale decoupled detection head.

### C. E3 — Bidirectional FusionNeck (BiFPN Topology)
E3 replaces the standard PANet with a lightweight bidirectional feature network inspired by BiFPN. All intermediate channel dimensions across $P_3, P_4, P_5$ are unified to $C = 128$. Lateral skip connections bypass the top-down pathway to fuse raw backbone features directly into the bottom-up aggregation nodes using learnable fast-normalized fusion weights:
$$O = \sum_i \frac{w_i}{\epsilon + \sum_j w_j} \cdot I_i, \quad w_i \ge 0$$
where $I_i$ are input feature maps, and $w_i$ are ReLU-constrained learnable scalars. This architecture eliminates redundant intermediate convolutions, significantly reducing parameter count and FLOPs.

### D. E4 — Auxiliary Boundary Supervision (BoundaryHead)
E4 integrates an auxiliary convolution head branched from the high-resolution $P_3$ feature map during training. The auxiliary head predicts a continuous binary cell-boundary distance mask, supervised via Binary Cross-Entropy (BCE) loss against rasterized polygon ground-truth boundaries. Crucially, the BoundaryHead is **train-only**: all auxiliary convolution weights ($18,529$ parameters) are detached and pruned prior to model serialization, incurring exactly zero parameter or latency overhead during inference.

### E. E5a & E5b — Bounding-Box Regression Loss Variants
Retaining the exact E1 architecture, we substitute the standard CIoU loss:
1. **E5a (Shape-IoU):** Introduces shape and scale deviation penalties based on width/height ratios:
   $$\mathcal{L}_{\text{Shape-IoU}} = 1 - \text{IoU} + \mathcal{R}_{\text{shape}}$$
   where $\mathcal{R}_{\text{shape}}$ weights the relative deviation of box dimensions against the ground truth, emphasizing aspect ratio alignment in dense rectangular cells.
2. **E5b (MPDIoU):** Directly penalizes the normalized Euclidean distance between the top-left $(x_1, y_1)$ and bottom-right $(x_2, y_2)$ bounding-box corners:
   $$d_1^2 = (x_1^{\text{pred}} - x_1^{\text{gt}})^2 + (y_1^{\text{pred}} - y_1^{\text{gt}})^2$$
   $$d_2^2 = (x_2^{\text{pred}} - x_2^{\text{gt}})^2 + (y_2^{\text{pred}} - y_2^{\text{gt}})^2$$
   $$\mathcal{L}_{\text{MPDIoU}} = 1 - \text{IoU} + \frac{d_1^2}{w_c^2 + h_c^2} + \frac{d_2^2}{w_c^2 + h_c^2}$$
   where $w_c, h_c$ denote the dimensions of the minimum enclosing bounding box.

### F. E8 — Full Combined Model
E8 unifies the high-resolution $P_2$ stride-4 head, the Bidirectional FusionNeck, the auxiliary BoundaryHead, and Shape-IoU regression loss into a single composite architecture.

### G. Explicit Exclusion of E6 and E7
To maintain experimental integrity, we explicitly report that two preliminary conceptual variants—E6 (a post-hoc ContextRefineClassifier) and E7 (a DCNv2 deformable convolution backbone)—were **SKIPPED** and not trained on real micrographs. We present no synthetic or fabricated metrics for E6 or E7.

---

## V. MATHEMATICAL FORMULATION

### A. Total Multi-Task Loss Formulation
The base network is trained via a multi-task objective combining classification, bounding-box regression, and distribution focal loss:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$
where $\lambda_{\text{cls}} = 0.5$, $\lambda_{\text{box}} = 7.5$, $\lambda_{\text{dfl}} = 1.5$, and $\lambda_{\text{aux}} = 0.25$ (active only during E4 and E8 training; $\lambda_{\text{aux}} = 0$ for all other stages).

### B. Complete-IoU (CIoU) Loss
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
$$v = \frac{4}{\pi^2} \left( \arctan \frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan \frac{w}{h} \right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$
Here, $\rho(\cdot)$ is Euclidean distance between box centroids, and $c$ is the diagonal length of the smallest enclosing box.

### C. Shape-IoU Formulation
Shape-IoU scales the coordinate offsets by dynamic shape factors $hh$ and $ww$:
$$ww = \frac{2 \cdot (w^{\text{gt}})^{\text{scale}}}{(w^{\text{gt}})^{\text{scale}} + (h^{\text{gt}})^{\text{scale}}}, \quad hh = \frac{2 \cdot (h^{\text{gt}})^{\text{scale}}}{(w^{\text{gt}})^{\text{scale}} + (h^{\text{gt}})^{\text{scale}}}$$
$$\Omega = ww \cdot \frac{(x_c - x_c^{\text{gt}})^2}{c_w^2} + hh \cdot \frac{(y_c - y_c^{\text{gt}})^2}{c_h^2}$$
$$\mathcal{R}_{\text{shape}} = \Omega + \frac{\left(1 - e^{-(w - w^{\text{gt}})/w^{\text{gt}}}\right)^2}{2} + \frac{\left(1 - e^{-(h - h^{\text{gt}})/h^{\text{gt}}}\right)^2}{2}$$
$$\mathcal{L}_{\text{Shape-IoU}} = 1 - \text{IoU} + \mathcal{R}_{\text{shape}}$$

### D. Evaluation Metrics
Precision ($P$), Recall ($R$), and $F_1$ score are evaluated at a fixed operational confidence threshold $\tau = 0.25$:
$$P = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad R = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$
Mean Average Precision (mAP) is computed by integrating the precision-recall curve across an exhaustive recall sweep under a low detection threshold $\tau = 0.001$:
$$\text{mAP} = \int_{0}^{1} p(r) \, dr$$
where mAP50 evaluates at $\text{IoU} = 0.50$, and mAP50-95 averages across $\text{IoU} \in [0.50, 0.95]$ in increments of $0.05$.

---

## VI. EXPERIMENTAL SETUP

### A. Training Hyperparameters and Hardware
All models were trained from compatible pretrained YOLOv8n weights on an isolated workstation:
* **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM, TGP 35–80 W).
* **Environment:** Python 3.12.10, PyTorch 2.6.0+cu124, Ultralytics 8.4.182, CUDA 12.4.
* **Hyperparameters:** Epochs = 50, Early stopping patience = 20 epochs, Batch size = 16, Image resolution = $640 \times 640$, Optimizer = SGD (initial $\text{lr} = 0.01$, momentum = 0.937, weight decay = 0.0005), Automatic Mixed Precision (AMP) = True, Deterministic random seed = 42.

### B. Inference Benchmarking Protocol
To guarantee publication-grade hardware characterization, we measured inference latency using raw forward passes on CUDA event timers across 5 interleaved repeats (100 warm-up runs, 300 timed iterations per model). All measurements exclude Non-Maximum Suppression (NMS) to isolate architectural compute differences.

---

## VII. RESULTS AND ABLATION ANALYSIS

TABLE I summarizes the master detection and computational results evaluated on the held-out test partition (160 tiles, 1,247 cell instances).

### TABLE I: MASTER DETECTION AND COMPUTATIONAL RESULTS ON HELD-OUT TEST SET
| Stage | Architecture Description | Inference Params | GFLOPs (@640) | Precision ($\tau=0.25$) | Recall ($\tau=0.25$) | $F_1$ Score ($\tau=0.25$) | mAP50 ($\tau=0.001$) | mAP50-95 ($\tau=0.001$) | Latency (FP16, ms) | FPS (FP16) | Peak VRAM (MB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Stock YOLOv8n Baseline | 3,011,043 | 8.1942 | 0.4131 | **0.5646** | 0.4771 | 0.4846 | 0.4247 | 11.08 | 90.3 | 257.2 |
| **E2** | P2 Stride-4 Detection Head | 2,926,692 | 12.3582 | 0.6141 | 0.4058 | 0.4887 | 0.5756 | 0.5007 | 13.80 | 72.5 | 421.5 |
| **E3** | Bidirectional FusionNeck | **2,019,061** | **7.2822** | 0.6560 | 0.3929 | **0.4915** | 0.6210 | 0.5368 | **10.92** | **91.5** | **256.7** |
| **E4** | Auxiliary BoundaryHead* | 3,011,043 | 8.1942 | **0.9038** | 0.0377 | 0.0724 | 0.5422 | 0.4581 | 11.07 | 90.3 | 256.2 |
| **E5a** | Shape-IoU Regression Loss | 3,011,043 | 8.1942 | 0.6954 | 0.3039 | 0.4230 | **0.6324** | **0.5579** | 11.08 | 90.3 | 257.2 |
| **E5b** | MPDIoU Regression Loss | 3,011,043 | 8.1942 | 0.6646 | 0.0858 | 0.1520 | 0.4436 | 0.3814 | 11.08 | 90.3 | 257.2 |
| **E8** | Full Combined Architecture* | 3,052,662 | 12.7959 | 0.7317 | 0.3609 | 0.4834 | 0.6043 | 0.5042 | 16.69 | 59.9 | 422.0 |

*\*Note: E4 includes 18,529 train-only parameters pruned before inference; E8 includes 9,313 train-only parameters pruned before inference. E6 and E7 were SKIPPED.*

### A. Overall Detection Performance
The baseline stock YOLOv8n (E1) achieves an mAP50 of 0.4846 and mAP50-95 of 0.4247 at $\tau=0.001$. At the operational operating threshold of $\tau = 0.25$, E1 yields a precision of 0.4131 and recall of 0.5646 ($F_1 = 0.4771$). While E1 demonstrates robust recall, its relatively low precision reflects substantial false-positive detections caused by ambiguous background textures and shared plant cell walls.

### B. Finding 1: Bidirectional FusionNeck (E3) Achieves Pareto-Optimal Efficiency
E3 replaces the standard PANet with our lightweight bidirectional topology. As detailed in TABLE I, E3 reduces inference parameters by **32.9%** (down to 2.02M) and GFLOPs by **11.1%** (down to 7.28), while achieving the highest $F_1$ score (**0.4915**) at $\tau=0.25$ and boosting mAP50 by **+0.1364** (0.6210 vs. 0.4846) at $\tau=0.001$. Latency drops to 10.92 ms, delivering 91.5 FPS. This demonstrates that standard YOLOv8 feature necks possess substantial channel and convolutional redundancy for single-class bio-imaging; cross-scale bidirectional routing preserves multi-resolution cell cues more effectively with fewer parameters.

### C. Finding 2: Shape-IoU (E5a) Maximizes Detection Accuracy
Modifying only the regression loss to Shape-IoU (E5a) yields the highest detection accuracy across the entire experimental suite: **mAP50 = 0.6324** (+0.1478 over E1) and **mAP50-95 = 0.5579** (+0.1332 over E1). Because Shape-IoU alters only the backpropagated gradient during training, it incurs **zero additional inference parameters, zero FLOP increase, and identical latency (11.08 ms, 90.3 FPS)**. In contrast, MPDIoU (E5b) suffers severe performance degradation, with recall plunging to 0.0858 at $\tau=0.25$ and mAP50 dropping to 0.4436. We hypothesize that MPDIoU's direct point-distance penalty is hypersensitive to minute boundary shifts in densely packed cell mosaics, destabilizing gradient updates.

### D. Finding 3: Precision-Recall Trade-offs in P2 Head and BoundaryHead
The stride-4 ($P_2$) head in E2 improves precision to 0.6141 (+0.2010 vs. E1) and mAP50 to 0.5756, confirming that high-resolution spatial maps enhance small-cell localization. However, this comes at a steep computational cost: GFLOPs surge by +50.8% (12.36 vs. 8.19) and latency increases to 13.80 ms (72.5 FPS).

The auxiliary BoundaryHead (E4) achieves an extraordinary precision of **0.9038** (+0.4907 over E1) with zero inference overhead. However, its recall at $\tau=0.25$ drops precipitously to 0.0377 ($F_1 = 0.0724$). Interestingly, when evaluated at $\tau=0.001$, E4 achieves a respectable mAP50 of 0.5422. This divergence indicates that auxiliary boundary regularization severely compresses the network's output confidence scores. Operating E4 at standard natural-image thresholds ($\tau=0.25$) discards valid predictions; it requires an operational threshold recalibration ($\tau \approx 0.05\text{--}0.10$).

### E. Finding 4: Compound Architecture Trade-offs (E8)
E8 combines $P_2$, FusionNeck, BoundaryHead, and Shape-IoU. While achieving strong precision (0.7317 at $\tau=0.25$) and competitive accuracy ($\text{mAP50}=0.6043$, $\text{mAP50-95}=0.5042$), it does not surpass the lighter single-modification models (E3 mAP50 = 0.6210, E5a mAP50 = 0.6324). Furthermore, its computational burden is the highest in the suite (12.80 GFLOPs, 16.69 ms latency, 59.9 FPS, 422.0 MB VRAM). This empirical outcome demonstrates that stacking individually beneficial architectural modifications can induce gradient competition and diminishing returns without extensive joint hyperparameter re-tuning.

---

## VIII. E8 CHECKPOINT SERIALIZATION AND METHOD-BINDING VERIFICATION

During initial post-training validation, E8 produced anomalously low metrics ($\text{mAP50} = 0.0165$, $P = 0.000$, $R = 0.000$). A rigorous engineering investigation identified a critical PyTorch checkpoint deserialization failure.

E8's custom forward pass was dynamically bound (`_predict_once`) to route intermediate activations through the external FusionNeck module. When serialized via `torch.save`, dynamic python instance method bindings were dropped. Reloading the raw checkpoint caused Ultralytics to execute the default forward method, completely bypassing the FusionNeck weights and feeding unrefined tensors to the detection head. 

Once the method binding was explicitly restored, E8 achieved its true verified performance ($\text{mAP50} = 0.6043$, $\text{mAP50-95} = 0.5042$, $P = 0.7317$, $R = 0.3609$). This finding highlights a crucial lesson for deep-learning practitioners: custom hybrid architectures modifying stock framework execution graphs must incorporate explicit forward-pass integrity checks post-deserialization.

---

## IX. COMPUTATIONAL EFFICIENCY AND PARETO ANALYSIS

Edge deployments in microscopy workstations require balancing detection throughput against memory footprints. TABLE II itemizes the computational profiling on our RTX 3050 Laptop GPU.

### TABLE II: COMPUTATIONAL PROFILING AND HARDWARE EFFICIENCY (RTX 3050 LAPTOP GPU)
| Stage | Model Variant | Inference Params (M) | GFLOPs (@640) | Batch=1 FP32 (ms) | Batch=1 FP16 (ms) | Batch=16 FP16 (ms) | Batch=1 FPS (FP16) | Batch=16 FPS (FP16) | Peak VRAM (MB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Stock YOLOv8n | 3.011 | 8.19 | $9.07 \pm 0.42$ | $11.08 \pm 0.12$ | $38.30 \pm 0.13$ | 90.3 | 417.8 | 257.2 |
| **E2** | P2 Stride-4 Head | 2.927 | 12.36 | $10.58 \pm 0.97$ | $13.80 \pm 0.41$ | $62.63 \pm 0.59$ | 72.5 | 255.5 | 421.5 |
| **E3** | BiFPN FusionNeck | **2.019** | **7.28** | **$9.17 \pm 0.93$** | **$10.92 \pm 0.13$** | **$38.93 \pm 0.54$** | **91.5** | **411.0** | **256.7** |
| **E4** | BoundaryHead | 3.011 | 8.19 | $9.02 \pm 1.08$ | $11.07 \pm 0.14$ | $38.50 \pm 0.35$ | 90.3 | 415.6 | 256.2 |
| **E5a** | Shape-IoU Loss | 3.011 | 8.19 | $9.07 \pm 0.42$ | $11.08 \pm 0.12$ | $38.30 \pm 0.13$ | 90.3 | 417.8 | 257.2 |
| **E5b** | MPDIoU Loss | 3.011 | 8.19 | $9.07 \pm 0.42$ | $11.08 \pm 0.12$ | $38.30 \pm 0.13$ | 90.3 | 417.8 | 257.2 |
| **E8** | Full Combined | 3.053 | 12.80 | $13.50 \pm 0.28$ | $16.69 \pm 0.25$ | $73.17 \pm 0.59$ | 59.9 | 218.7 | 422.0 |

*Note: Timed across 300 iterations following 100 warm-up iterations via CUDA events. NMS excluded.*

In batch=1 FP16 inference, E3 exhibits the lowest latency (10.92 ms) and highest throughput (91.5 FPS). In batch=16 FP16 (GPU-bound regime), E3 processes batches in 38.93 ms, achieving an aggregate throughput of 411.0 FPS. Crucially, all evaluated models maintain peak runtime memory footprints well below 500 MB (256.2 MB for E4 to 422.0 MB for E8), confirming that high-resolution plant-cell detection via tiled YOLOv8 is fully feasible on entry-level 4 GB edge GPUs.

---

## X. DISCUSSION

### A. Measured Results vs. Interpretations
We strictly delineate empirical measurements from plausible interpretations:
* **Directly Measured:** E3 reduces parameters by 32.9% and GFLOPs by 11.1% while reaching $F_1=0.4915$. Shape-IoU (E5a) increases mAP50 by +0.1478 without adding parameters or FLOPs.
* **Reasonable Interpretation:** The superior performance of E3 suggests that standard YOLOv8n feature necks contain redundant convolutional filters for single-class cell detection.
* **Hypothesis:** MPDIoU degrades on dense cell meshes due to hypersensitivity of point-distance penalties to small boundary misalignments.

### B. Architectural Trade-offs
Our results demonstrate that single-class plant cell detection does not require heavy feature extraction backbones. The success of E3 proves that lightweight bidirectional fusion with reduced channel widths (128 channels) outperforms wider standard architectures. Furthermore, the accuracy leap achieved by Shape-IoU (E5a) highlights that in dense microscopy, refining the geometric alignment penalty during loss backpropagation is vastly more cost-effective than expanding network depth or spatial resolution.

---

## XI. LIMITATIONS AND THREATS TO VALIDITY

1. **Taxonomic Ambiguity:** An internal documentation discrepancy exists regarding whether the Sansha-5 and WO115-2 lines represent *Arabidopsis* or *Pyropia haitanensis*. External biological confirmation is required before drawing species-specific morphological conclusions.
2. **Specimen Scale:** While the dataset contains 3,359 annotated cells, they originate from 16 specimens. Although our specimen-level split prevents temporal leakage between 0h and 24h observations, cross-laboratory and multi-instrument generalization remains unmeasured.
3. **Single-Class Scope:** The present study focuses exclusively on cell localization ($n_c = 1$). Sub-cellular organelles and histological cell types are not differentiated.
4. **Confidence Threshold Sensitivity:** Metrics such as Precision, Recall, and $F_1$ are highly sensitive to the operating threshold $\tau$. As observed in E4, models with compressed confidence distributions require custom threshold calibration rather than default $\tau=0.25$ cutoff rules.

---

## XII. CONCLUSION

This paper presented a systematic, leakage-free empirical investigation of lightweight YOLOv8 modifications for high-resolution plant cell detection under constrained GPU resources. By enforcing a strict specimen-level split, we prevented temporal memorization between serial micrographs. Our findings show that the Bidirectional FusionNeck (E3) establishes a Pareto-optimal design, eliminating nearly one-third of baseline parameters and 11% of GFLOPs while achieving the highest $F_1$ score (0.4915) and boosting mAP50 by +0.1364. Simultaneously, Shape-IoU (E5a) maximized detection accuracy ($\text{mAP50}=0.6324$) with zero inference overhead. Finally, we demonstrated that composite multi-module stacking (E8) incurs severe computational overhead without commensurate accuracy gains, and documented the necessity of post-deserialization method verification in custom PyTorch pipelines.

---

## REFERENCES

[1] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," version 8.0.0, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics  
[2] M. Tan, R. Pang, and Q. V. Le, "EfficientDet: Scalable and Efficient Object Detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 10781–10790.  
[3] C. Li et al., "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications," *arXiv preprint arXiv:2209.02976*, 2022.  
[4] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 7464–7475.  
[5] H. Zhang et al., "Shape-IoU: More Accurate Metric Considering Bounding Box Shape and Scale," *arXiv preprint arXiv:2312.17663*, 2023.  
[6] M. Ma et al., "MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression," *arXiv preprint arXiv:2307.07662*, 2023.  
[7] Z. Zheng et al., "Enhancing Geometric Factors in Model Learning to Surpass FCOS," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 9, pp. 5022–5034, 2022.  
[8] Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, "YOLOX: Exceeding YOLO Series in 2021," *arXiv preprint arXiv:2107.08430*, 2021.  
[9] J. Dai et al., "Deformable Convolutional Networks," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 764–773.  
[10] S. A. Park et al., "Automated Cell Tracking and Morphological Phenotyping in Plant Microscopic Sequences," *Plant Methods*, vol. 18, no. 1, pp. 1–14, 2022.  
