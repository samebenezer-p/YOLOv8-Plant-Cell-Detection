"""
Specimen-level train / val / test splitter.

Ensures no data leakage: all images/tiles from a single biological specimen
stay in the same split partition. Uses stratified grouping when class-level
metadata is available.

Usage:
    python -m src.data.split --data_dir data/raw --output_dir data/splits \
        --train 0.7 --val 0.15 --test 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Specimen ID parsing
# ---------------------------------------------------------------------------

# Default pattern: extract prefix before the last numeric suffix.
# "specimen_A_0001.jpg" -> "specimen_A"
# "xylem_0042.jpg"      -> "xylem" (all synthetic images share one specimen)
_SPECIMEN_RE = re.compile(r"^(.+?)_?\d+\.\w+$")


def extract_specimen_id(filename: str, pattern: Optional[str] = None) -> str:
    """Parse the specimen ID from an image filename.

    Parameters
    ----------
    filename : str
        Image filename (basename, not full path).
    pattern : str, optional
        Custom regex with one capture group for specimen ID.  Falls back to
        the default pattern which strips the trailing ``_NNNN.ext`` suffix.

    Returns
    -------
    str
        Specimen identifier string.

    Raises
    ------
    ValueError
        If no specimen ID can be parsed from *filename*.
    """
    if pattern:
        m = re.match(pattern, filename)
    else:
        m = _SPECIMEN_RE.match(filename)
    if not m:
        raise ValueError(
            f"Cannot extract specimen ID from '{filename}'. "
            "Provide a custom --specimen_pattern regex."
        )
    return m.group(1)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_by_specimen(
    data_dir: str | Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    specimen_pattern: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Group image filenames by specimen ID.

    Parameters
    ----------
    data_dir : path-like
        Directory containing image files (searched recursively).
    extensions : tuple of str
        Allowed image file extensions (case-insensitive).
    specimen_pattern : str, optional
        Custom regex forwarded to :func:`extract_specimen_id`.

    Returns
    -------
    dict
        ``{specimen_id: [relative_path_1, relative_path_2, ...]}``
    """
    data_dir = Path(data_dir)
    groups: Dict[str, List[str]] = defaultdict(list)

    for root, _dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(extensions):
                continue
            rel = os.path.relpath(os.path.join(root, fname), data_dir)
            sid = extract_specimen_id(fname, pattern=specimen_pattern)
            groups[sid].append(rel)

    return dict(groups)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def specimen_split(
    data_dir: str | Path,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    specimen_pattern: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, List[str]]:
    """Split images into train / val / test by specimen ID.

    All images from the same specimen are assigned to exactly one split.

    Parameters
    ----------
    data_dir : path-like
        Root directory of images.
    ratios : (float, float, float)
        Train / val / test ratios (must sum to 1.0 ± 0.01).
    seed : int
        Random seed for reproducibility.
    specimen_pattern : str, optional
        Custom regex for specimen ID extraction.
    output_dir : path-like, optional
        If given, write ``train.txt``, ``val.txt``, ``test.txt`` with one
        relative image path per line.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}`` of relative image
        paths.

    Raises
    ------
    ValueError
        If ratios don't sum to ~1.0 or if there are fewer than 3 specimens.
    """
    r_train, r_val, r_test = ratios
    if abs(r_train + r_val + r_test - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.3f}")

    groups = group_by_specimen(data_dir, specimen_pattern=specimen_pattern)
    specimen_ids = sorted(groups.keys())

    if len(specimen_ids) < 3:
        raise ValueError(
            f"Need ≥3 specimens for a 3-way split, found {len(specimen_ids)}. "
            "Check data_dir or specimen_pattern."
        )

    rng = random.Random(seed)
    rng.shuffle(specimen_ids)

    n = len(specimen_ids)
    n_train = max(1, round(n * r_train))
    n_val = max(1, round(n * r_val))
    # test gets the rest
    train_ids = specimen_ids[:n_train]
    val_ids = specimen_ids[n_train : n_train + n_val]
    test_ids = specimen_ids[n_train + n_val :]

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for sid in train_ids:
        splits["train"].extend(groups[sid])
    for sid in val_ids:
        splits["val"].extend(groups[sid])
    for sid in test_ids:
        splits["test"].extend(groups[sid])

    # Sort for determinism
    for k in splits:
        splits[k].sort()

    # Validate no leakage
    _validate_no_leakage(splits)

    # Write to disk
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, paths in splits.items():
            out_file = output_dir / f"{split_name}.txt"
            out_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
            print(f"  Wrote {len(paths):>4d} images → {out_file}")

    # Summary
    print(f"\nSpecimen-level split (seed={seed}):")
    print(f"  Specimens: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
    print(
        f"  Images:    {len(splits['train'])} train / "
        f"{len(splits['val'])} val / {len(splits['test'])} test"
    )

    return splits


def _validate_no_leakage(splits: Dict[str, List[str]]) -> None:
    """Assert that no image appears in more than one split."""
    all_images: List[str] = []
    for paths in splits.values():
        all_images.extend(paths)
    if len(all_images) != len(set(all_images)):
        dupes = [x for x in all_images if all_images.count(x) > 1]
        raise RuntimeError(f"Data leakage detected – duplicated images: {dupes[:10]}")


# ---------------------------------------------------------------------------
# Existing split loader (for the synthetic dataset)
# ---------------------------------------------------------------------------

def load_existing_split(
    images_dir: str | Path,
) -> Dict[str, List[str]]:
    """Load the pre-existing train/val/test directory split.

    Works with the standard YOLO layout::

        images/
        ├── train/
        ├── val/
        └── test/

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}`` of absolute image
        paths.
    """
    images_dir = Path(images_dir)
    splits: Dict[str, List[str]] = {}
    for split_name in ("train", "val", "test"):
        split_dir = images_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected split directory: {split_dir}")
        paths = sorted(
            str(p) for p in split_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        )
        splits[split_name] = paths
    return splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Specimen-level train/val/test splitter"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Root image directory")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--specimen_pattern", type=str, default=None,
                        help="Custom regex with one capture group for specimen ID")
    args = parser.parse_args()

    specimen_split(
        data_dir=args.data_dir,
        ratios=(args.train, args.val, args.test),
        seed=args.seed,
        specimen_pattern=args.specimen_pattern,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
