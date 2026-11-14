"""
make_qualitative.py
===================
Generates Figure B for the IEEE conference paper: Qualitative detection comparison.
- Columns: Ground Truth | E1 (Baseline) | E3 (FusionNeck) | E5a (Shape-IoU)
- Rows: Two test tiles, row 1 from plant_8 and row 2 from plant_16.
- Deterministic tile selection rule:
    For each of plant_8 and plant_16, select the test tile whose ground-truth box count
    is the MEDIAN among that specimen's test tiles (lower median if even count).
    Tie-breaking: alphabetical by tile stem.
- Double-column IEEE size: 7.16 in wide x 3.8 in tall.
- Only the 640x640 micrographs are rasterized; bounding boxes and typography are pure vector.
- Thin outline boxes (~0.6 pt, no fill, no labels).
- Operating confidence threshold: conf >= 0.25, NMS iou=0.45, max_det=300.
- Outputs:
    results/conference_paper/final_figures/qualitative.pdf
    results/conference_paper/final_figures/qualitative.png
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

SPECIMEN_PREFIXES = {
    "plant_8":  ("B6_0808", "B6_0809"),
    "plant_16": ("250102-4-1_T0", "250102-4-1_T24"),
}

MODELS = {
    "E1":  {"ckpt": "runs/detect/results/real/E1/weights/best.pt",  "title": "E1 (Baseline)"},
    "E3":  {"ckpt": "runs/detect/results/real/E3/weights/best.pt",  "title": "E3 (FusionNeck)"},
    "E5a": {"ckpt": "runs/detect/results/real/E5a/weights/best.pt", "title": "E5a (Shape-IoU)"},
}

COLOR_GT = "#00E5FF"     # High-contrast vivid cyan for Ground Truth
COLOR_PRED = "#FF3300"   # High-contrast vibrant red-orange for Predictions


def get_specimen_tiles(specimen_name: str, filter_nonzero: bool = False):
    """Finds all test tiles for a specimen and returns list of (stem, gt_count)."""
    prefixes = SPECIMEN_PREFIXES[specimen_name]
    lbl_dir = PROJECT_ROOT / "data" / "real" / "labels" / "test"

    records = []
    for f in sorted(lbl_dir.glob("*.txt")):
        if any(f.name.startswith(p) for p in prefixes):
            with open(f, "r", encoding="utf-8") as fp:
                lines = [l.strip() for l in fp if l.strip()]
            records.append((f.stem, len(lines)))

    if filter_nonzero:
        records = [r for r in records if r[1] > 0]

    # Sort deterministically: ascending by GT count, tie-break by stem name
    records_sorted = sorted(records, key=lambda x: (x[1], x[0]))
    return records_sorted


def select_median_tile(specimen_name: str, filter_nonzero: bool = False):
    """Selects median tile (lower median if even count)."""
    records = get_specimen_tiles(specimen_name, filter_nonzero=filter_nonzero)
    n = len(records)
    if n == 0:
        raise ValueError(f"No test tiles found for {specimen_name}")
    median_idx = (n - 1) // 2
    chosen_stem, gt_count = records[median_idx]
    return chosen_stem, gt_count, median_idx, n


def load_gt_boxes(stem: str):
    """Reads YOLO format boxes: class x_center y_center w h (normalized)."""
    lbl_path = PROJECT_ROOT / "data" / "real" / "labels" / "test" / f"{stem}.txt"
    boxes = []
    if lbl_path.exists():
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    boxes.append((xc, yc, w, h))
    return boxes


def run_model_inference(model_key: str, img_path: Path, device: str):
    """Runs inference at conf=0.25, iou=0.45, max_det=300."""
    ckpt_rel = MODELS[model_key]["ckpt"]
    ckpt_path = str(PROJECT_ROOT / ckpt_rel)
    yolo = YOLO(ckpt_path)
    if device.startswith("cuda"):
        yolo.to(device)

    res = yolo.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.25,
        iou=0.45,
        max_det=300,
        device=device,
        verbose=False,
    )[0]

    boxes_xyxy = []
    if res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy().tolist()
    return boxes_xyxy


def draw_detection_panel(
    ax,
    img_rgb: np.ndarray,
    boxes: list,
    color: str,
    caption: str,
    is_yolo_norm: bool = False,
    row_label: str = None,
    col_label: str = None,
):
    """Renders a single 640x640 panel with vector bounding boxes."""
    # Micrograph is the only rasterized element
    ax.imshow(img_rgb, rasterized=True, interpolation="nearest")
    ax.set_xlim(0, 640)
    ax.set_ylim(640, 0)  # Inverted y for image coordinates
    ax.set_xticks([])
    ax.set_yticks([])

    for b in boxes:
        if is_yolo_norm:
            xc, yc, w, h = b
            x1 = (xc - w / 2.0) * 640.0
            y1 = (yc - h / 2.0) * 640.0
            bw = w * 640.0
            bh = h * 640.0
        else:
            x1, y1, x2, y2 = b
            bw = x2 - x1
            bh = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            bw,
            bh,
            linewidth=0.60,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

    # Captions below panel: e.g. "GT n=..." or "<model> n=..."
    ax.text(
        0.50,
        -0.08,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        fontfamily="serif",
    )

    if row_label:
        ax.set_ylabel(row_label, fontsize=7.5, fontfamily="serif", labelpad=3)

    if col_label:
        ax.set_title(col_label, fontsize=7.5, fontfamily="serif", pad=4)


def generate_figure(chosen_tiles: dict, out_pdf: Path, out_png: Path, device: str):
    """Generates the 2x4 IEEE double-column qualitative grid."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.8), dpi=300)

    rows = ["plant_8", "plant_16"]
    col_names = ["Ground truth", "E1", "E3", "E5a"]

    for r_idx, specimen in enumerate(rows):
        stem = chosen_tiles[specimen]["stem"]
        gt_count = chosen_tiles[specimen]["gt_count"]
        img_path = PROJECT_ROOT / "data" / "real" / "images" / "test" / f"{stem}.jpg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Ground Truth (Col 0)
        gt_boxes = load_gt_boxes(stem)
        gt_caption = f"GT n={len(gt_boxes)}"
        draw_detection_panel(
            axes[r_idx, 0],
            img_rgb,
            gt_boxes,
            color=COLOR_GT,
            caption=gt_caption,
            is_yolo_norm=True,
            row_label=specimen if r_idx in [0, 1] else None,
            col_label=col_names[0] if r_idx == 0 else None,
        )

        # 2. E1, E3, E5a (Cols 1, 2, 3)
        for c_idx, model_key in enumerate(["E1", "E3", "E5a"], start=1):
            preds = run_model_inference(model_key, img_path, device)
            pred_caption = f"{model_key} n={len(preds)}"
            draw_detection_panel(
                axes[r_idx, c_idx],
                img_rgb,
                preds,
                color=COLOR_PRED,
                caption=pred_caption,
                is_yolo_norm=False,
                row_label=None,
                col_label=col_names[c_idx] if r_idx == 0 else None,
            )

    plt.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.92,
        bottom=0.08,
        wspace=0.05,
        hspace=0.22,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_pdf), format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved qualitative vector figure: {out_pdf}")
    print(f"Saved qualitative raster preview: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure B: Qualitative detection grid.")
    parser.add_argument("--filter-nonzero", action="store_true", help="Compute median only over tiles with GT > 0.")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("FIGURE B: DETERMINISTIC TILE SELECTION & QUALITATIVE RENDERING")
    print("=" * 80)

    # 1. Deterministic selection (Strict rule across all specimen test tiles)
    p8_stem, p8_gt, p8_idx, p8_total = select_median_tile("plant_8", filter_nonzero=args.filter_nonzero)
    p16_stem, p16_gt, p16_idx, p16_total = select_median_tile("plant_16", filter_nonzero=args.filter_nonzero)

    mode_str = "NON-ZERO TILES" if args.filter_nonzero else "ALL SPECIMEN TEST TILES"
    print(f"Selection Domain: {mode_str}")
    print(f"  plant_8:  Chosen tile: {p8_stem}.jpg  (GT count: {p8_gt}, index: {p8_idx}/{p8_total})")
    print(f"  plant_16: Chosen tile: {p16_stem}.jpg (GT count: {p16_gt}, index: {p16_idx}/{p16_total})")

    chosen = {
        "plant_8":  {"stem": p8_stem,  "gt_count": p8_gt},
        "plant_16": {"stem": p16_stem, "gt_count": p16_gt},
    }

    if args.filter_nonzero:
        out_pdf = SCRIPT_DIR / "qualitative_nonzero.pdf"
        out_png = SCRIPT_DIR / "qualitative_nonzero.png"
    else:
        out_pdf = SCRIPT_DIR / "qualitative.pdf"
        out_png = SCRIPT_DIR / "qualitative.png"

    generate_figure(chosen, out_pdf, out_png, device)

    # If running standard without arguments, ALSO generate the nonzero variant as companion
    if not args.filter_nonzero:
        print("\nAlso generating companion figure with non-zero median tiles (qualitative_nonzero.pdf)...")
        p8_nz_stem, p8_nz_gt, _, _ = select_median_tile("plant_8", filter_nonzero=True)
        p16_nz_stem, p16_nz_gt, _, _ = select_median_tile("plant_16", filter_nonzero=True)
        print(f"  plant_8 (non-zero median):  {p8_nz_stem}.jpg (GT count: {p8_nz_gt})")
        print(f"  plant_16 (non-zero median): {p16_nz_stem}.jpg (GT count: {p16_nz_gt})")
        chosen_nz = {
            "plant_8":  {"stem": p8_nz_stem,  "gt_count": p8_nz_gt},
            "plant_16": {"stem": p16_nz_stem, "gt_count": p16_nz_gt},
        }
        generate_figure(chosen_nz, SCRIPT_DIR / "qualitative_nonzero.pdf", SCRIPT_DIR / "qualitative_nonzero.png", device)

    print("=" * 80)
    print("Qualitative figure generation completed successfully.")


if __name__ == "__main__":
    main()
