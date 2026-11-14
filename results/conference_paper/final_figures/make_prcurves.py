"""
make_prcurves.py
================
Generates Figure A for the IEEE conference paper: Precision-Recall curves on held-out test split.
- Models evaluated: E1, E2, E3, E4, E5a, E5b, E8 (nc=1, class "cell", IoU=0.50).
- Test split: 160 test tiles from data/real/data.yaml (plant_8 and plant_16).
- Evaluation settings: imgsz=640, conf=0.001, iou=0.45, max_det=300.
- Restores custom module bindings (FusionNeck, BoundaryHead, E8 dynamic forward pass).
- Verifies mAP50 against frozen benchmark values (+/- 0.003).
- Outputs vector PDF (pdf.fonttype=42) and 300-dpi PNG:
    results/conference_paper/final_figures/prcurves.pdf
    results/conference_paper/final_figures/prcurves.png
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Frozen target values for Step 0 verification gate
MODELS = {
    "E1":  {"ckpt": "runs/detect/results/real/E1/weights/best.pt",  "exp_mAP50": 0.4846, "name": "E1 Baseline"},
    "E2":  {"ckpt": "runs/detect/results/real/E2/weights/best.pt",  "exp_mAP50": 0.5756, "name": "E2 (+P2)"},
    "E3":  {"ckpt": "runs/detect/results/real/E3/weights/best.pt",  "exp_mAP50": 0.6210, "name": "E3 (+FusionNeck)"},
    "E4":  {"ckpt": "runs/detect/results/real/E4/weights/best.pt",  "exp_mAP50": 0.5422, "name": "E4 (+BoundaryHead)"},
    "E5a": {"ckpt": "runs/detect/results/real/E5a/weights/best.pt", "exp_mAP50": 0.6324, "name": "E5a (+Shape-IoU)"},
    "E5b": {"ckpt": "runs/detect/results/real/E5b/weights/best.pt", "exp_mAP50": 0.4436, "name": "E5b (+MPDIoU)"},
    "E8":  {"ckpt": "runs/detect/results/real/E8/weights/best.pt",  "exp_mAP50": 0.6043, "name": "E8 (Combined)"},
}

DATA_YAML = str(PROJECT_ROOT / "data" / "real" / "data.yaml")

# Colorblind-safe palette (Wong / Okabe-Ito inspired) with distinct linestyles & markers
STYLE_CFG = {
    "E1":  {"color": "#333333", "linestyle": "-",  "marker": "o", "markevery": 100},
    "E2":  {"color": "#0072B2", "linestyle": "--", "marker": "s", "markevery": 115},
    "E3":  {"color": "#009E73", "linestyle": "-",  "marker": "^", "markevery": 130},
    "E4":  {"color": "#E69F00", "linestyle": "-.", "marker": "v", "markevery": 145},
    "E5a": {"color": "#D55E00", "linestyle": "-",  "marker": "D", "markevery": 160},
    "E5b": {"color": "#CC79A7", "linestyle": ":",  "marker": "P", "markevery": 175},
    "E8":  {"color": "#332288", "linestyle": (0, (3, 1, 1, 1)), "marker": "X", "markevery": 190},
}


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


def evaluate_and_extract_curves():
    """Runs validation at conf=0.001 and extracts PR curve data for all models."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device} ...")

    curves_data = {}
    print("\n" + "=" * 78)
    print(f"{'Model':<6} | {'mAP50 (Exp)':<12} | {'mAP50 (Obtained)':<16} | {'Delta':<8} | {'AUC (trapz)':<12} | {'Gate'}")
    print("=" * 78)

    for stage, info in MODELS.items():
        yolo = load_model(stage, info["ckpt"])
        res = yolo.val(
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
        box = res.box
        map50 = float(box.map50)
        exp_50 = info["exp_mAP50"]
        delta = abs(map50 - exp_50)

        # PR curve data: x is Recall (0 to 1, 1000 pts), y is Precision (1, 1000)
        rec = box.curves_results[0][0]
        prec = box.curves_results[0][1][0]
        auc_trapz = float(np.trapezoid(prec, rec))

        gate_status = "PASS" if delta <= 0.003 else "FAIL"
        print(f"{stage:<6} | {exp_50:<12.4f} | {map50:<16.4f} | {delta:<8.4f} | {auc_trapz:<12.4f} | {gate_status}")

        if delta > 0.003:
            raise RuntimeError(f"Step 0 Verification Gate Failed for {stage}: obtained {map50:.4f} vs expected {exp_50:.4f}")

        curves_data[stage] = {
            "recall": rec,
            "precision": prec,
            "map50": map50,
            "auc": auc_trapz,
        }

    print("=" * 78 + "\n")
    return curves_data


def plot_pr_curves(curves_data: dict, out_pdf: Path, out_png: Path):
    """Plots IEEE single-column Precision-Recall curves."""
    # IEEE single-column settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.autolayout": True,
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=300)

    # Plot order: baseline first, ablated intermediates, top performers last so they stand out
    plot_order = ["E1", "E5b", "E4", "E2", "E8", "E3", "E5a"]

    for stage in plot_order:
        data = curves_data[stage]
        style = STYLE_CFG[stage]
        rec = data["recall"]
        prec = data["precision"]
        map50 = data["map50"]

        label = f"{stage} ({map50:.3f})"
        ax.plot(
            rec,
            prec,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.15,
            marker=style["marker"],
            markevery=style["markevery"],
            markersize=3.8,
            alpha=0.92,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.30, color="#888888")

    # Legend inside plot in lower-left (unoccupied area for PR curves)
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.90,
        edgecolor="#cccccc",
        fancybox=False,
        handlelength=2.2,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.25,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_pdf), format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved vector figure: {out_pdf}")
    print(f"Saved raster preview: {out_png}")


def main():
    out_pdf = SCRIPT_DIR / "prcurves.pdf"
    out_png = SCRIPT_DIR / "prcurves.png"
    cache_path = SCRIPT_DIR / "prcurves_data.npz"

    if "--use-cache" in sys.argv and cache_path.exists():
        print(f"Loading cached PR curve data from {cache_path}...")
        npz = np.load(str(cache_path))
        curves_data = {}
        for s in MODELS.keys():
            curves_data[s] = {
                "recall": npz[f"{s}_recall"],
                "precision": npz[f"{s}_precision"],
                "map50": float(npz[f"{s}_map50"]),
                "auc": float(npz[f"{s}_auc"]),
            }
    else:
        curves_data = evaluate_and_extract_curves()
        # Save cache for instant re-plotting
        save_dict = {}
        for s, d in curves_data.items():
            save_dict[f"{s}_recall"] = d["recall"]
            save_dict[f"{s}_precision"] = d["precision"]
            save_dict[f"{s}_map50"] = np.array(d["map50"])
            save_dict[f"{s}_auc"] = np.array(d["auc"])
        np.savez(str(cache_path), **save_dict)
        print(f"Cached curve data to {cache_path}")

    plot_pr_curves(curves_data, out_pdf, out_png)
    print("make_prcurves.py complete.")


if __name__ == "__main__":
    main()
