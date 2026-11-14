# Scientific Claims Classification for Conference Paper

This document explicitly delineates what claims are rigorously proven by the experimental evidence, what interpretations are plausible but require cautious phrasing, and what claims are strictly unsupported and must **never** appear in the conference paper.

---

## Category A: VERIFIED CLAIMS
*Directly supported by numerical evidence, repeatable benchmarks, and verified code execution.*

1. **E3 (BiFPN FusionNeck) Achieves Pareto-Optimal Performance**:
   - Reduces inference parameters by **32.9%** (from 3,011,043 down to 2,019,061).
   - Reduces GFLOPs by **11.1%** (from 8.1942 down to 7.2822).
   - Increases inference throughput from 90.3 FPS to **91.5 FPS** (Batch=1 FP16) and 411.0 FPS (Batch=16 FP16).
   - Achieves the highest F1 score (**0.4915**) at conf=0.25 and increases mAP50 by **+0.1364** (0.6210 vs 0.4846) at conf=0.001.

2. **Shape-IoU (E5a) Outperforms Default CIoU and MPDIoU in Dense Microscopy Geometry**:
   - Under an identical model architecture, Shape-IoU increases mAP50 by **+0.1478** (0.6324 vs 0.4846 at conf=0.001) and mAP50-95 by **+0.1332** (0.5579 vs 0.4247).
   - MPDIoU (E5b) suffers a **-0.0410** decrease in mAP50 and severe recall degradation (0.0858 at conf=0.25).

3. **P2 Stride-4 Head (E2) Substantially Elevates Precision at High Compute Cost**:
   - Boosts Precision at conf=0.25 from 0.4131 to **0.6141** (+0.2010), but increases GFLOPs by **+50.8%** (12.36 vs 8.19) and decreases frame rate to 72.5 FPS.

4. **Auxiliary BoundaryHead (E4) Incurs Exactly Zero Inference Overhead**:
   - Pruning the 18,529-parameter boundary head during deployment leaves inference parameters (3,011,043) and GFLOPs (8.1942) identical to baseline.
   - Boosts Precision at conf=0.25 to **0.9038** (+0.4907 vs baseline).

5. **Strict Specimen-Level Partition Prevents Leakage**:
   - The test set is composed exclusively of `plant_8` and `plant_16`. No images, tiles, or cell observations from these specimens appear in the training or validation splits.
   - The baseline test mAP50 of 0.3226 (conf=0.25) confirms that the model generalizes to unseen specimens without artificial memorization.

6. **E8 Compound Architecture Behavior**:
   - When properly executed with its bidirectional fusion neck, E8 achieves mAP50 = 0.6043 and mAP50-95 = 0.5042 at conf=0.001, and Precision = 0.7317 at conf=0.25.
   - However, it has the highest compute cost (12.80 GFLOPs, 16.69 ms latency, 59.9 FPS).

---

## Category B: REASONABLE INTERPRETATIONS
*Plausible scientific hypotheses grounded in empirical trends, to be presented with cautious language (e.g., "The results suggest...", "A plausible interpretation is...").*

1. **Feature Redundancy in Microscopic Feature Pyramids**:
   - The fact that E3 outperforms E1 while removing ~1M parameters suggests that standard YOLOv8n feature necks contain redundant convolutional filters for single-class cell detection, which BiFPN-style cross-scale routing replaces more efficiently.

2. **Boundary Supervision Confidence Calibration**:
   - The extreme precision (0.9038) and suppressed recall (0.0377) of E4 at conf=0.25, contrasted with its strong mAP50 (0.5422) at conf=0.001, suggests that boundary loss heavily regularizes the network's prediction confidence, necessitating a lower operating threshold (~0.10) for balanced operational use.

3. **Multi-Scale Head Interference in Deep Stacks**:
   - The observation that E8 does not surpass the single-component E3 or E5a designs suggests that simultaneously adding stride-4 spatial features, bidirectional fusion, and boundary loss introduces competing gradient dynamics that require joint hyperparameter retuning rather than modular stacking.

---

## Category C: UNSUPPORTED CLAIMS
*Strictly prohibited from the conference paper. These statements are scientifically false, contradicted by project evidence, or unproven.*

1. **NOVELTY CLAIMS ON EXISTING ARCHITECTURES**:
   - ❌ *Do NOT claim*: "We invented a novel P2 head", "We propose a novel BiFPN architecture", "We invented Shape-IoU", or "We designed a novel BoundaryHead".
   - *Truth*: All these techniques are established in literature. The paper must frame them as an experimental ablation study applying established techniques to plant microscopy.

2. **BIOLOGICAL HISTOLOGY CLASSES**:
   - ❌ *Do NOT claim*: "Our detector identifies vessel, fiber, and parenchyma cells" or "We achieve 3-class biological segmentation".
   - *Truth*: The raw dataset contains only lineage tracking identifiers (1, 2, 34-1). No biological histological labels exist. The task is strictly single-class cell detection (`nc=1`).

3. **DEFINITIVE BOTANICAL IDENTITY**:
   - ❌ *Do NOT claim*: "The dataset is definitively established as *Arabidopsis thaliana*" or "The dataset is conclusively proven to be *Pyropia haitanensis*".
   - *Truth*: The repository README describes the data as "Arabidopsis plant lines" while naming strains "Sansha-5" and "WO115-2" (which are *Pyropia haitanensis* cultivars). The identity has an unresolved conflict and must be stated as requiring external curator confirmation.

4. **"FIRST EVER" / GENERALIZATION OVERCLAIMS**:
   - ❌ *Do NOT claim*: "This is the first ever plant cell detection paper" or "Our model generalizes to all plant microscopy types".
   - *Truth*: No cross-domain datasets or external microscopy benchmarks were tested. Claims must be restricted to the evaluated dataset.

5. **MATHEMATICAL PROOF OF ZERO LEAKAGE**:
   - ❌ *Do NOT claim*: "A test mAP50 of 0.3226 mathematically proves zero data leakage".
   - *Truth*: A low mAP indicates absence of trivial memorization, but does not constitute formal proof of distribution independence. Use cautious language: *"the sanity check did not indicate suspiciously high test performance."*
