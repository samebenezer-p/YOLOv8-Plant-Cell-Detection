"""
CLI end-to-end smoke tests on test tile and full test micrograph.
"""

import csv
import json
from pathlib import Path
import pytest

from plantcell_tool.cli import main


def test_cli_smoke_single_tile(tmp_path: Path):
    """Run CLI predict on a single 640x640 tile and verify output consistency."""
    tile_path = Path("data/real/images/test/250102-4-1_T0_tile0000.jpg")
    assert tile_path.is_file(), f"Test tile not found at {tile_path}"

    out_dir = tmp_path / "out_tile"
    argv = [
        "predict",
        "--input", str(tile_path),
        "--model", "E5a",
        "--conf", "auto",
        "--output-dir", str(out_dir),
        "--save-overlay",
        "--save-csv",
        "--save-yolo",
    ]

    ret = main(argv)
    assert ret == 0, "CLI returned non-zero exit code"

    stem = tile_path.stem
    csv_file = out_dir / f"{stem}_detections.csv"
    json_file = out_dir / f"{stem}_meta.json"
    overlay_file = out_dir / f"{stem}_overlay.png"
    yolo_file = out_dir / f"{stem}_yolo.txt"

    assert csv_file.is_file(), "CSV output not generated"
    assert json_file.is_file(), "JSON output not generated"
    assert overlay_file.is_file(), "Overlay PNG not generated"
    assert yolo_file.is_file(), "YOLO txt not generated"

    # Verify CSV rows == JSON count
    with open(json_file, encoding="utf-8") as f:
        meta_data = json.load(f)
    json_count = meta_data["detection_summary"]["count"]

    with open(csv_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["x1", "y1", "x2", "y2", "score", "width", "height"]
        csv_rows = len(list(reader))

    assert csv_rows == json_count, f"CSV rows ({csv_rows}) != JSON count ({json_count})"


def test_cli_smoke_full_micrograph(tmp_path: Path):
    """Run CLI predict on a full 3840x2160 micrograph and verify outputs."""
    micrograph_path = Path("data/real_raw/plant_8/img/B6_0808.tif")
    assert micrograph_path.is_file(), f"Test micrograph not found at {micrograph_path}"

    out_dir = tmp_path / "out_micrograph"
    argv = [
        "predict",
        "--input", str(micrograph_path),
        "--model", "E3",
        "--conf", "auto",
        "--output-dir", str(out_dir),
        "--save-overlay",
        "--save-csv",
        "--batch", "16",
    ]

    ret = main(argv)
    assert ret == 0, "CLI returned non-zero exit code"

    stem = micrograph_path.stem
    csv_file = out_dir / f"{stem}_detections.csv"
    json_file = out_dir / f"{stem}_meta.json"
    overlay_file = out_dir / f"{stem}_overlay.png"

    assert csv_file.is_file()
    assert json_file.is_file()
    assert overlay_file.is_file()

    with open(json_file, encoding="utf-8") as f:
        meta_data = json.load(f)
    json_count = meta_data["detection_summary"]["count"]

    with open(csv_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        csv_rows = len(list(reader))

    assert csv_rows == json_count, f"CSV rows ({csv_rows}) != JSON count ({json_count})"
