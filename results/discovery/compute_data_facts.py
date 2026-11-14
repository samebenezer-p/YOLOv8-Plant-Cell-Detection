"""
compute_data_facts.py
=====================
Read-only script to compute data statistics for discovery report:
- Per split and specimen tile counts, empty tile counts, box counts.
- Box dimensions at 640x640 scale (min, 5th, 25th, 50th, 75th, 95th, max for w, h, sqrt(w*h)).
- Full micrograph bounding box area coverage (union of boxes / image area).
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = PROJECT_ROOT / "data" / "real"
DATA_YAML = REAL_DATA_DIR / "data.yaml"

def analyze_dataset():
    splits = ["train", "val", "test"]
    stats_per_split = {}
    all_boxes = []

    # Map prefixes to specimens
    # Let's inspect data/real_raw to get exact mapping of micrograph stems to specimen names
    raw_dir = PROJECT_ROOT / "data" / "real_raw"
    stem_to_specimen = {}
    if raw_dir.exists():
        for spec_dir in raw_dir.iterdir():
            if spec_dir.is_dir() and spec_dir.name.startswith("plant_"):
                img_dir = spec_dir / "img"
                if img_dir.exists():
                    for f in img_dir.glob("*.*"):
                        stem_to_specimen[f.stem] = spec_dir.name

    box_stats = {
        "overall": {"w": [], "h": [], "sqrt_wh": []},
        "train": {"w": [], "h": [], "sqrt_wh": []},
        "val": {"w": [], "h": [], "sqrt_wh": []},
        "test": {"w": [], "h": [], "sqrt_wh": []},
    }

    tile_sizes_found = set()
    formats_found = set()

    for sp in splits:
        img_dir = REAL_DATA_DIR / "images" / sp
        lbl_dir = REAL_DATA_DIR / "labels" / sp

        img_files = sorted(list(img_dir.glob("*.*")))
        lbl_files = sorted(list(lbl_dir.glob("*.txt")))

        for img_path in img_files:
            formats_found.add(img_path.suffix.lower())

        # Check a sample of images for dimensions
        if img_files:
            sample_img = cv2.imread(str(img_files[0]))
            tile_sizes_found.add(sample_img.shape[:2])

        split_boxes = 0
        zero_label_tiles = 0
        specimen_counts = {}

        for img_path in img_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            # Identify specimen
            stem = img_path.stem
            # Stem format: <micrograph_stem>_tile<idx>
            # Find matching micrograph stem
            matched_spec = "unknown"
            for m_stem, spec_name in stem_to_specimen.items():
                if stem.startswith(m_stem):
                    matched_spec = spec_name
                    break

            if matched_spec not in specimen_counts:
                specimen_counts[matched_spec] = {"tiles": 0, "zero_tiles": 0, "boxes": 0}
            specimen_counts[matched_spec]["tiles"] += 1

            lines = []
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    lines = [l.strip() for l in lf if l.strip()]

            n_boxes = len(lines)
            if n_boxes == 0:
                zero_label_tiles += 1
                specimen_counts[matched_spec]["zero_tiles"] += 1
            else:
                split_boxes += n_boxes
                specimen_counts[matched_spec]["boxes"] += n_boxes
                for l in lines:
                    parts = l.split()
                    if len(parts) >= 5:
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                        w_px = w_norm * 640.0
                        h_px = h_norm * 640.0
                        sqrt_wh_px = np.sqrt(w_px * h_px)
                        box_stats["overall"]["w"].append(w_px)
                        box_stats["overall"]["h"].append(h_px)
                        box_stats["overall"]["sqrt_wh"].append(sqrt_wh_px)
                        box_stats[sp]["w"].append(w_px)
                        box_stats[sp]["h"].append(h_px)
                        box_stats[sp]["sqrt_wh"].append(sqrt_wh_px)

        stats_per_split[sp] = {
            "total_tiles": len(img_files),
            "zero_label_tiles": zero_label_tiles,
            "total_boxes": split_boxes,
            "specimens": specimen_counts,
        }

    # Compute percentiles: min, 5, 25, 50, 75, 95, max
    percentiles = [0, 5, 25, 50, 75, 95, 100]
    box_summary = {}
    for key, data in box_stats.items():
        box_summary[key] = {}
        for dim in ["w", "h", "sqrt_wh"]:
            arr = np.array(data[dim])
            if len(arr) > 0:
                vals = np.percentile(arr, percentiles)
                box_summary[key][dim] = {
                    "count": len(arr),
                    "min": float(vals[0]),
                    "p5": float(vals[1]),
                    "p25": float(vals[2]),
                    "p50_median": float(vals[3]),
                    "p75": float(vals[4]),
                    "p95": float(vals[5]),
                    "max": float(vals[6]),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }

    # 4. Full micrograph coverage fraction
    # In data/real_raw or data/real_full
    full_labels_dir = PROJECT_ROOT / "data" / "real_full" / "labels"
    micrograph_coverage = {}
    W, H = 3840, 2160
    total_area = W * H

    if raw_dir.exists():
        for spec_dir in sorted(raw_dir.iterdir(), key=lambda x: int(x.name.split('_')[1]) if '_' in x.name else 0):
            if not (spec_dir.is_dir() and spec_dir.name.startswith("plant_")):
                continue
            spec_name = spec_dir.name
            img_dir = spec_dir / "img"
            json_dir = spec_dir / "json"

            spec_coverages = []
            for json_file in sorted(json_dir.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as jf:
                    jdata = json.load(jf)
                shapes = jdata.get("shapes", [])

                # Create binary mask to compute union of box area
                mask = np.zeros((H, W), dtype=np.uint8)
                for sh in shapes:
                    pts = np.array(sh["points"], dtype=np.float32)
                    x1 = int(np.clip(np.min(pts[:, 0]), 0, W))
                    x2 = int(np.clip(np.max(pts[:, 0]), 0, W))
                    y1 = int(np.clip(np.min(pts[:, 1]), 0, H))
                    y2 = int(np.clip(np.max(pts[:, 1]), 0, H))
                    mask[y1:y2, x1:x2] = 1

                union_area = int(np.sum(mask))
                frac = union_area / total_area
                spec_coverages.append({
                    "micrograph": json_file.stem,
                    "num_cells": len(shapes),
                    "union_pixel_area": union_area,
                    "coverage_fraction": frac,
                })

            mean_frac = float(np.mean([x["coverage_fraction"] for x in spec_coverages])) if spec_coverages else 0.0
            micrograph_coverage[spec_name] = {
                "micrographs": spec_coverages,
                "mean_coverage_fraction": mean_frac,
            }

    overall_mean_coverage = float(np.mean([
        m["coverage_fraction"]
        for spec in micrograph_coverage.values()
        for m in spec["micrographs"]
    ])) if micrograph_coverage else 0.0

    output = {
        "data_yaml": str(DATA_YAML),
        "tile_sizes_found": [list(x) for x in tile_sizes_found],
        "formats_found": list(formats_found),
        "stats_per_split": stats_per_split,
        "box_size_percentiles_640scale": box_summary,
        "micrograph_coverage": {
            "overall_mean_coverage_fraction": overall_mean_coverage,
            "by_specimen": micrograph_coverage,
        }
    }

    out_file = PROJECT_ROOT / "results" / "discovery" / "data_facts.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Computed data facts successfully saved to: {out_file}")

if __name__ == "__main__":
    analyze_dataset()
