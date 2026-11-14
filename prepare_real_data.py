"""
Real data integration script for YOLOv8.

This script automates the process of converting raw high-resolution plant micrographs
and their corresponding annotations into the YOLOv8 format expected by our pipeline.

It performs the following steps:
1. Slices the high-res images and labels into overlapping tiles using `src/data/tiling.py`.
2. Splits the resulting tiles into train, validation, and test sets.
3. Generates the `real_data.yaml` file for training.

Usage:
    Place your raw high-res images in `data/raw/` and your YOLO annotations in `data/annotations/`.
    Then run:
        python prepare_real_data.py --tile_size 640 --overlap 0.25
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare real micrograph dataset")
    parser.add_argument("--tile_size", type=int, default=640, help="Tile size")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap fraction")
    args = parser.parse_args()

    project_root = Path(__file__).parent.resolve()
    raw_dir = project_root / "data" / "raw"
    anno_dir = project_root / "data" / "annotations"
    tiles_dir = project_root / "data" / "tiles"
    
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print("ERROR: No raw images found in data/raw/. Please add your real plant micrographs.")
        return
        
    print("Step 1: Tiling high-resolution micrographs...")
    # Clean previous tiles
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True)
    
    # Run the tiling module
    try:
        subprocess.run([
            "python", "-m", "src.data.tiling",
            "--image_dir", str(raw_dir),
            "--label_dir", str(anno_dir) if anno_dir.exists() and any(anno_dir.iterdir()) else "none",
            "--output_dir", str(tiles_dir),
            "--tile_size", str(args.tile_size),
            "--overlap", str(args.overlap)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR during tiling: {e}")
        return

    print("Step 2: Splitting tiles into train/val/test...")
    images_dir = project_root / "images" / "real"
    labels_dir = project_root / "labels" / "real"
    
    # Run the split module
    try:
        subprocess.run([
            "python", "-m", "src.data.split",
            "--input_dir", str(tiles_dir),
            "--output_images", str(images_dir),
            "--output_labels", str(labels_dir),
            "--train_ratio", "0.7",
            "--val_ratio", "0.15"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR during splitting: {e}")
        return

    print("Step 3: Generating real_data.yaml...")
    yaml_content = f"""# YOLOv8 Real Plant Micrographs Dataset
path: {project_root}
train: images/real/train
val: images/real/val
test: images/real/test

# Classes
names:
  0: vessel
  1: fiber
  2: parenchyma
"""
    yaml_path = project_root / "real_data.yaml"
    yaml_path.write_text(yaml_content)
    
    print(f"\nSUCCESS! Real data integration complete.")
    print(f"You can now train using: --config configs/e2_tiling_p2.yaml (and changing data_yaml to real_data.yaml)")

if __name__ == "__main__":
    main()
