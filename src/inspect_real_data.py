"""
Phase 1: Comprehensive Inspection of data/real_raw/
This script reads and analyzes the real dataset without modifying any raw files.
"""

import os
import sys
import json
import glob
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_RAW_DIR = PROJECT_ROOT / "data" / "real_raw"
INSPECTION_OUT_DIR = PROJECT_ROOT / "results" / "inspection"
INSPECTION_OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    import torch
    print(f"torch.__version__: {torch.__version__} | torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA unavailable. Training must not start on CPU.")

    print("\n--- 1. Top-Level Structure of data/real_raw/ ---")
    top_entries = sorted(os.listdir(REAL_RAW_DIR))
    print(f"Top-level entries ({len(top_entries)}): {top_entries}")

    plant_dirs = [d for d in top_entries if (REAL_RAW_DIR / d).is_dir() and d.startswith("plant_")]
    plant_dirs.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)
    print(f"Subfolders found: {len(plant_dirs)} plant folders")

    folder_stats = {}
    all_images = {}
    all_annotations = {}
    label_counter = Counter()
    shape_types = Counter()
    objects_per_img = []
    fields_per_object = set()
    top_level_json_keys = set()
    image_dimensions = Counter()
    has_flags = {
        "class_category_names": False,
        "tracking_ids": False,
        "lineage": False,
        "parent_child_links": False,
        "time_points": False
    }

    # Tracking inspection details
    sample_objects = []

    for p_dir in plant_dirs:
        p_path = REAL_RAW_DIR / p_dir
        img_dir = p_path / "img"
        json_dir = p_path / "json"

        img_files = sorted(list(img_dir.glob("*.*"))) if img_dir.exists() else []
        json_files = sorted(list(json_dir.glob("*.json"))) if json_dir.exists() else []

        img_exts = sorted(list({f.suffix.lower() for f in img_files}))
        json_exts = sorted(list({f.suffix.lower() for f in json_files}))

        folder_stats[p_dir] = {
            "img_count": len(img_files),
            "json_count": len(json_files),
            "img_exts": img_exts,
            "json_exts": json_exts,
            "example_imgs": [f.name for f in img_files[:2]],
            "example_jsons": [f.name for f in json_files[:2]]
        }

        for img_f in img_files:
            all_images[f"{p_dir}/{img_f.stem}"] = img_f

        for json_f in json_files:
            stem_key = f"{p_dir}/{json_f.stem}"
            all_annotations[stem_key] = json_f

            # Parse JSON
            try:
                with open(json_f, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"ERROR reading {json_f}: {e}")
                continue

            top_level_json_keys.update(data.keys())
            h = data.get("imageHeight")
            w = data.get("imageWidth")
            if h is not None and w is not None:
                image_dimensions[(w, h)] += 1

            shapes = data.get("shapes", [])
            objects_per_img.append(len(shapes))

            for sh in shapes:
                stype = sh.get("shape_type", "unknown")
                shape_types[stype] += 1
                fields_per_object.update(sh.keys())

                lbl = sh.get("label", "")
                label_counter[lbl] += 1

                # Check flags
                if lbl:
                    has_flags["class_category_names"] = True
                for k in ["tracking_id", "track_id", "id", "cell_id"]:
                    if k in sh or k in data:
                        has_flags["tracking_ids"] = True
                for k in ["parent", "child", "children", "lineage", "division"]:
                    if k in sh or k in data:
                        has_flags["parent_child_links"] = True
                        has_flags["lineage"] = True

                if len(sample_objects) < 5:
                    sample_objects.append(sh)

    print("\n--- Folder Details ---")
    for p_dir, s in folder_stats.items():
        print(f"{p_dir:10s} | Imgs: {s['img_count']:2d} ({','.join(s['img_exts'])}) | "
              f"JSONs: {s['json_count']:2d} ({','.join(s['json_exts'])}) | "
              f"Examples: {s['example_imgs']}")

    print("\n--- 2. README Analysis ---")
    readme_path = REAL_RAW_DIR / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "not stated"
    (INSPECTION_OUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")
    print("README saved to results/inspection/README.md")

    print("\n--- 3. Annotation Structure ---")
    print(f"Top-level JSON keys: {sorted(list(top_level_json_keys))}")
    print(f"Shape types: {dict(shape_types)}")
    print(f"Fields per shape object: {sorted(list(fields_per_object))}")
    print(f"Objects per image: min={min(objects_per_img) if objects_per_img else 0}, "
          f"max={max(objects_per_img) if objects_per_img else 0}, "
          f"mean={np.mean(objects_per_img) if objects_per_img else 0:.1f}, "
          f"total={sum(objects_per_img)}")
    print(f"Image dimensions (W x H): {dict(image_dimensions)}")
    print("Sample shape object:")
    if sample_objects:
        sample_copy = {k: v for k, v in sample_objects[0].items() if k != "points"}
        sample_copy["points_head"] = sample_objects[0].get("points", [])[:3]
        print(json.dumps(sample_copy, indent=2))

    print("\n--- 4. Class Check (Critical) ---")
    print(f"Total distinct labels found: {len(label_counter)}")
    print("Distinct labels with counts:")
    labels_output = {}
    for lbl, count in sorted(label_counter.items(), key=lambda x: -x[1]):
        print(f"  '{lbl}': {count}")
        labels_output[lbl] = count

    with open(INSPECTION_OUT_DIR / "labels_found.json", "w", encoding="utf-8") as f:
        json.dump(labels_output, f, indent=2)
    print("Labels saved to results/inspection/labels_found.json")

    target_classes = ["vessel", "fiber", "parenchyma"]
    all_targets_present = all(c in label_counter for c in target_classes)
    if not all_targets_present:
        print("\nTarget 3-class mapping is NOT supported by the annotations.")
    else:
        print("\nTarget 3-class mapping is supported.")

    print("\n--- 5. Matching & Integrity ---")
    img_keys = set(all_images.keys())
    json_keys = set(all_annotations.keys())

    unmatched_imgs = img_keys - json_keys
    unmatched_jsons = json_keys - img_keys
    print(f"Total images: {len(img_keys)}")
    print(f"Total JSON annotations: {len(json_keys)}")
    print(f"Unmatched images: {len(unmatched_imgs)} ({sorted(list(unmatched_imgs))})")
    print(f"Unmatched JSONs: {len(unmatched_jsons)} ({sorted(list(unmatched_jsons))})")

    # Check for empty annotations
    zero_anno_imgs = []
    for k, json_f in all_annotations.items():
        with open(json_f, "r", encoding="utf-8") as f:
            d = json.load(f)
            if len(d.get("shapes", [])) == 0:
                zero_anno_imgs.append(k)
    print(f"Images with zero annotations: {len(zero_anno_imgs)} ({zero_anno_imgs})")

    # Check duplicate or augmented suffixes (.rf.<hash>)
    rf_copies = [k for k in img_keys if ".rf." in k]
    print(f"Duplicate / augmented copies (.rf.<hash>): {len(rf_copies)}")

    print("\n--- 6. Specimen Unit & Grouping ---")
    # Determine specimen grouping
    # In each plant_x, there are pairs of images (e.g. 0h and 24h of the same plant specimen)
    specimen_groups = defaultdict(list)
    for p_dir in plant_dirs:
        p_path = REAL_RAW_DIR / p_dir / "img"
        for img_f in sorted(p_path.glob("*.*")):
            specimen_groups[p_dir].append(img_f.name)

    print("Specimen groups:")
    for spec_id, flist in sorted(specimen_groups.items(), key=lambda x: int(x[0].split('_')[1])):
        print(f"  {spec_id:10s} | image_count: {len(flist)} | example files: {flist}")

    print("\n--- 7. Generating Sample Inspection Overlays ---")
    # Generate 6 sample images with annotations overlaid from 3 different specimens
    selected_specimens = ["plant_1", "plant_9", "plant_16"]
    sample_count = 0
    for spec_id in selected_specimens:
        p_path = REAL_RAW_DIR / spec_id
        img_dir = p_path / "img"
        json_dir = p_path / "json"
        img_files = sorted(list(img_dir.glob("*.*")))
        for img_f in img_files[:2]:
            json_f = json_dir / (img_f.stem + ".json")
            if not json_f.exists():
                continue

            # Load image
            img = cv2.imread(str(img_f))
            if img is None:
                print(f"Warning: could not read {img_f}")
                continue

            with open(json_f, "r", encoding="utf-8") as f:
                anno_data = json.load(f)

            overlay = img.copy()
            for sh in anno_data.get("shapes", []):
                pts = np.array(sh.get("points", []), dtype=np.int32)
                if len(pts) >= 3:
                    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    # Label text at first point
                    lbl_text = str(sh.get("label", ""))[:8]
                    cv2.putText(overlay, lbl_text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Resize for convenient inspection review (e.g. max 1280 wide)
            h, w = overlay.shape[:2]
            scale = 1280 / max(h, w)
            preview = cv2.resize(overlay, (int(w * scale), int(h * scale)))

            out_name = f"{spec_id}_{img_f.stem}_overlay.jpg"
            cv2.imwrite(str(INSPECTION_OUT_DIR / out_name), preview)
            sample_count += 1
            print(f"  Saved overlay {sample_count}/6: {out_name}")

    print(f"Inspection overlays saved to {INSPECTION_OUT_DIR}")

if __name__ == "__main__":
    main()
