"""
make_max_f1.py
==============
Computes Figure C data table (threshold-free F1 dynamics vs operating threshold)
for all models on the held-out test split (160 tiles).
Outputs: results/conference_paper/final_figures/max_f1_table.csv

Metrics computed:
1. At conf=0.001:
   - Max F1 across the full confidence spectrum
   - Optimal confidence threshold (tau*) at which Max F1 occurs
   - Precision at tau*
   - Recall at tau*
   - mAP50 (conf=0.001)
   - mAP50-95 (conf=0.001)
2. At conf=0.25 (standard operating threshold):
   - Precision (conf=0.25)
   - Recall (conf=0.25)
   - F1 score (conf=0.25)
   - Verified against paper table within +/- 0.003:
     E1 (0.4131 / 0.5646 / 0.4771),
     E3 (0.6560 / 0.3929 / 0.4915),
     E5a (0.6954 / 0.3039 / 0.4230),
     E8 (0.7317 / 0.3609 / 0.4834).
"""

import sys
import os
import csv
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# Project root setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Custom module definitions for unpickling & dynamic forward pass
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

MODELS = {
    "E1":  {"ckpt": "runs/detect/results/real/E1/weights/best.pt",  "desc": "YOLOv8n Baseline",      "exp_p025": 0.4131, "exp_r025": 0.5646, "exp_f1025": 0.4771},
    "E2":  {"ckpt": "runs/detect/results/real/E2/weights/best.pt",  "desc": "P2 Stride-4 Head",       "exp_p025": 0.6141, "exp_r025": 0.4058, "exp_f1025": 0.4887},
    "E3":  {"ckpt": "runs/detect/results/real/E3/weights/best.pt",  "desc": "FusionNeck (BiFPN)",     "exp_p025": 0.6560, "exp_r025": 0.3929, "exp_f1025": 0.4915},
    "E4":  {"ckpt": "runs/detect/results/real/E4/weights/best.pt",  "desc": "Aux BoundaryHead",      "exp_p025": 0.9038, "exp_r025": 0.0377, "exp_f1025": 0.0724},
    "E5a": {"ckpt": "runs/detect/results/real/E5a/weights/best.pt", "desc": "Shape-IoU Loss",         "exp_p025": 0.6954, "exp_r025": 0.3039, "exp_f1025": 0.4230},
    "E5b": {"ckpt": "runs/detect/results/real/E5b/weights/best.pt", "desc": "MPDIoU Loss",           "exp_p025": 0.6646, "exp_r025": 0.0858, "exp_f1025": 0.1520},
    "E8":  {"ckpt": "runs/detect/results/real/E8/weights/best.pt",  "desc": "Full Combined Architecture", "exp_p025": 0.7317, "exp_r025": 0.3609, "exp_f1025": 0.4834},
}

DATA_YAML = str(PROJECT_ROOT / "data" / "real" / "data.yaml")


def load_model(stage: str, ckpt_rel_path: str):
    ckpt_path = str(PROJECT_ROOT / ckpt_rel_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    yolo = YOLO(ckpt_path)
    if torch.cuda.is_available():
        yolo.to("cuda")
    if stage == "E8":
        m = yolo.model
        if hasattr(m, "fusion_neck"):
            m._predict_once = fn._make_e8_predict_once(m.fusion_neck).__get__(m, type(m))
    return yolo


def evaluate_model(stage: str, info: dict, device: str):
    """Evaluates a model at conf=0.001 and conf=0.25 to calculate threshold-free and fixed-threshold metrics."""
    yolo = load_model(stage, info["ckpt"])

    # 1. Evaluate at conf=0.001 (sweeps full confidence spectrum)
    res_001 = yolo.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.45,
        max_det=300,
        device=device,
        verbose=False,
        plots=False,
        save=False,
    )
    b001 = res_001.box
    map50_001 = float(b001.map50)
    map_001 = float(b001.map)

    conf_x = b001.curves_results[1][0]
    f1_y = b001.curves_results[1][1][0]
    max_idx = int(np.argmax(f1_y))
    max_f1 = float(f1_y[max_idx])
    best_conf = float(conf_x[max_idx])
    p_at_max = float(b001.curves_results[2][1][0][max_idx])
    r_at_max = float(b001.curves_results[3][1][0][max_idx])

    # 2. Evaluate at conf=0.25 (paper operating threshold)
    res_025 = yolo.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.45,
        max_det=300,
        device=device,
        verbose=False,
        plots=False,
        save=False,
    )
    b025 = res_025.box
    p_025 = float(b025.mp)
    r_025 = float(b025.mr)
    f1_025 = (2 * p_025 * r_025 / (p_025 + r_025)) if (p_025 + r_025) > 0 else 0.0

    # Verification against expected paper values
    delta_p = abs(p_025 - info["exp_p025"])
    delta_r = abs(r_025 - info["exp_r025"])
    delta_f1 = abs(f1_025 - info["exp_f1025"])

    passed = (delta_p <= 0.003) and (delta_r <= 0.003) and (delta_f1 <= 0.003)
    if not passed:
        print(f"WARNING: Discrepancy detected for {stage} at conf=0.25:")
        print(f"  P: got {p_025:.4f} vs exp {info['exp_p025']:.4f} (delta={delta_p:.4f})")
        print(f"  R: got {r_025:.4f} vs exp {info['exp_r025']:.4f} (delta={delta_r:.4f})")
        print(f"  F1: got {f1_025:.4f} vs exp {info['exp_f1025']:.4f} (delta={delta_f1:.4f})")

    return {
        "Model": stage,
        "Description": info["desc"],
        "Max_F1": max_f1,
        "Conf_at_Max_F1": best_conf,
        "P_at_Max_F1": p_at_max,
        "R_at_Max_F1": r_at_max,
        "P_conf_0.25": p_025,
        "R_conf_0.25": r_025,
        "F1_conf_0.25": f1_025,
        "mAP50_conf_0.001": map50_001,
        "mAP50_95_conf_0.001": map_001,
        "Verified_conf025": passed,
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Executing make_max_f1.py on device: {device} ...")

    table_rows = []
    print("\n" + "=" * 105)
    print(f"{'Model':<6} | {'Max F1':<8} | {'tau*':<7} | {'P(tau*)':<8} | {'R(tau*)':<8} | {'P(0.25)':<8} | {'R(0.25)':<8} | {'F1(0.25)':<8} | {'mAP50':<8} | {'Status'}")
    print("=" * 105)

    for stage, info in MODELS.items():
        row = evaluate_model(stage, info, device)
        table_rows.append(row)
        st = "VERIFIED" if row["Verified_conf025"] else "MISMATCH"
        print(
            f"{row['Model']:<6} | "
            f"{row['Max_F1']:<8.4f} | "
            f"{row['Conf_at_Max_F1']:<7.4f} | "
            f"{row['P_at_Max_F1']:<8.4f} | "
            f"{row['R_at_Max_F1']:<8.4f} | "
            f"{row['P_conf_0.25']:<8.4f} | "
            f"{row['R_conf_0.25']:<8.4f} | "
            f"{row['F1_conf_0.25']:<8.4f} | "
            f"{row['mAP50_conf_0.001']:<8.4f} | "
            f"{st}"
        )
    print("=" * 105 + "\n")

    # Save to CSV
    out_csv = SCRIPT_DIR / "max_f1_table.csv"
    fieldnames = [
        "Model",
        "Description",
        "Max_F1",
        "Conf_at_Max_F1",
        "P_at_Max_F1",
        "R_at_Max_F1",
        "P_conf_0.25",
        "R_conf_0.25",
        "F1_conf_0.25",
        "mAP50_conf_0.001",
        "mAP50_95_conf_0.001",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in table_rows:
            filtered_r = {k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fieldnames}
            writer.writerow(filtered_r)

    print(f"Successfully saved max F1 table to: {out_csv}")


if __name__ == "__main__":
    main()
