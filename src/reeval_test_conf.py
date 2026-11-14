"""
Re-evaluation script for test split:
1. Re-evaluate E1, E2, E3, E4, E5a, E5b, E8 on test split at conf=0.001 for mAP50 & mAP50-95.
   Report P, R, F1 at conf=0.25 separately.
2. For E4, E5b, E8: compute max detection confidence per test tile, its distribution,
   and total number of detections at conf=0.001.
3. Print first 20 lines of results/inspection/README.md verbatim.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Backwards compatibility imports & aliases
import src.models.fusion_neck as fn
import src.models.boundary_branch as bb

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

CHECKPOINTS = {
    "E1": "runs/detect/results/real/E1/weights/best.pt",
    "E2": "runs/detect/results/real/E2/weights/best.pt",
    "E3": "runs/detect/results/real/E3/weights/best.pt",
    "E4": "runs/detect/results/real/E4/weights/best.pt",
    "E5a": "runs/detect/results/real/E5a/weights/best.pt",
    "E5b": "runs/detect/results/real/E5b/weights/best.pt",
    "E8": "runs/detect/results/real/E8/weights/best.pt",
}

DATA_YAML = "data/real/data.yaml"

def run_eval():
    print("================================================================================")
    print("1. TEST EVALUATION (conf=0.001 for mAP50/mAP50-95; conf=0.25 for P, R, F1)")
    print("================================================================================")
    results_q1 = {}

    for stage, ckpt in CHECKPOINTS.items():
        print(f"\nEvaluating {stage} ({ckpt})...")
        yolo = YOLO(ckpt)

        # 1a. Evaluate at conf=0.001
        res_001 = yolo.val(
            data=DATA_YAML,
            split="test",
            imgsz=640,
            conf=0.001,
            iou=0.45,
            max_det=300,
            verbose=False,
            plots=False,
            save=False,
        )
        box_001 = res_001.box
        map50_001 = float(box_001.map50)
        map_001 = float(box_001.map)

        # 1b. Evaluate at conf=0.25
        res_025 = yolo.val(
            data=DATA_YAML,
            split="test",
            imgsz=640,
            conf=0.25,
            iou=0.45,
            max_det=300,
            verbose=False,
            plots=False,
            save=False,
        )
        box_025 = res_025.box
        p_025 = float(box_025.mp)
        r_025 = float(box_025.mr)
        f1_025 = (2 * p_025 * r_025 / (p_025 + r_025)) if (p_025 + r_025) > 0 else 0.0
        map50_025 = float(box_025.map50)
        map_025 = float(box_025.map)

        results_q1[stage] = {
            "mAP50_conf0.001": map50_001,
            "mAP50-95_conf0.001": map_001,
            "P_conf0.25": p_025,
            "R_conf0.25": r_025,
            "F1_conf0.25": f1_025,
            "mAP50_conf0.25": map50_025,
            "mAP50-95_conf0.25": map_025,
        }

    print("\n--------------------------------------------------------------------------------")
    print(f"{'Stage':<6} | {'mAP50 (0.001)':<14} | {'mAP50-95 (0.001)':<18} | {'P (0.25)':<10} | {'R (0.25)':<10} | {'F1 (0.25)':<10}")
    print("--------------------------------------------------------------------------------")
    for s, d in results_q1.items():
        print(f"{s:<6} | {d['mAP50_conf0.001']:<14.4f} | {d['mAP50-95_conf0.001']:<18.4f} | {d['P_conf0.25']:<10.4f} | {d['R_conf0.25']:<10.4f} | {d['F1_conf0.25']:<10.4f}")
    print("--------------------------------------------------------------------------------")

    # Save to file
    with open("results/paper/reeval_test_q1.json", "w") as f:
        json.dump(results_q1, f, indent=2)

    # 2. Confidence distributions for E4, E5b, E8
    print("\n================================================================================")
    print("2. CONFIDENCE DISTRIBUTION AND DETECTIONS AT conf=0.001 (E4, E5b, E8)")
    print("================================================================================")
    test_img_dir = Path("data/real/images/test")
    test_images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    print(f"Total test tiles: {len(test_images)}")

    results_q2 = {}

    for stage in ["E4", "E5b", "E8"]:
        ckpt = CHECKPOINTS[stage]
        yolo = YOLO(ckpt)

        max_confs = []
        total_detections = 0
        all_confs = []

        for img_path in test_images:
            res = yolo.predict(
                source=str(img_path),
                imgsz=640,
                conf=0.001,
                iou=0.45,
                max_det=300,
                verbose=False,
            )[0]
            confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.array([])
            num_det = len(confs)
            total_detections += num_det
            if num_det > 0:
                max_confs.append(float(np.max(confs)))
                all_confs.extend(confs.tolist())
            else:
                max_confs.append(0.0)

        max_confs = np.array(max_confs)
        all_confs = np.array(all_confs)

        stats = {
            "total_test_tiles": len(test_images),
            "tiles_with_any_detection_at_0.001": int(np.sum(max_confs > 0.001)),
            "tiles_with_detection_at_0.25": int(np.sum(max_confs >= 0.25)),
            "total_detections_at_0.001": int(total_detections),
            "max_conf_min": float(np.min(max_confs)),
            "max_conf_p25": float(np.percentile(max_confs, 25)),
            "max_conf_median": float(np.median(max_confs)),
            "max_conf_mean": float(np.mean(max_confs)),
            "max_conf_p75": float(np.percentile(max_confs, 75)),
            "max_conf_max": float(np.max(max_confs)),
            "max_conf_std": float(np.std(max_confs)),
            "bins_max_conf": {
                "< 0.001 (no det)": int(np.sum(max_confs < 0.001)),
                "0.001 - 0.05": int(np.sum((max_confs >= 0.001) & (max_confs < 0.05))),
                "0.05 - 0.10": int(np.sum((max_confs >= 0.05) & (max_confs < 0.10))),
                "0.10 - 0.25": int(np.sum((max_confs >= 0.10) & (max_confs < 0.25))),
                ">= 0.25": int(np.sum(max_confs >= 0.25)),
            }
        }
        results_q2[stage] = stats

        print(f"\n--- {stage} ---")
        print(f"Total detections at conf=0.001: {stats['total_detections_at_0.001']}")
        print(f"Tiles with detections (at conf>=0.001): {stats['tiles_with_any_detection_at_0.001']} / {len(test_images)}")
        print(f"Tiles with detections (at conf>=0.25):  {stats['tiles_with_detection_at_0.25']} / {len(test_images)}")
        print(f"Max Conf Distribution per tile:")
        print(f"  Min:    {stats['max_conf_min']:.4f}")
        print(f"  25th%:  {stats['max_conf_p25']:.4f}")
        print(f"  Median: {stats['max_conf_median']:.4f}")
        print(f"  Mean:   {stats['max_conf_mean']:.4f}")
        print(f"  75th%:  {stats['max_conf_p75']:.4f}")
        print(f"  Max:    {stats['max_conf_max']:.4f}")
        print(f"  Std:    {stats['max_conf_std']:.4f}")
        print(f"Tile Max Conf Bins:")
        for b, count in stats['bins_max_conf'].items():
            print(f"  {b}: {count} tiles ({count/len(test_images)*100:.1f}%)")

    with open("results/paper/reeval_test_q2.json", "w") as f:
        json.dump(results_q2, f, indent=2)

    # 3. First 20 lines of results/inspection/README.md
    print("\n================================================================================")
    print("3. FIRST 20 LINES OF results/inspection/README.md (EXACT QUOTE)")
    print("================================================================================")
    readme_path = Path("results/inspection/README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = [next(f) for _ in range(20)]
    for line in lines:
        sys.stdout.write(line)
    print("================================================================================\n")

if __name__ == "__main__":
    run_eval()
