"""
Command Line Interface (CLI) for plant-cell detection on micrographs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from . import LIMITATIONS_NOTICE
from .export import export_csv, export_json, export_overlay, export_yolo
from .imageio import read_image
from .infer import detect_full_image
from .models import DOCUMENTED_THRESHOLDS, load_model, model_info


def print_banner() -> None:
    """Print standard header and limitations notice."""
    print("=" * 80)
    print("PLANT CELL DETECTION INFERENCE TOOL (YOLOv8 Tiled Micrograph Inference)")
    print("=" * 80)
    print(f"LIMITATIONS NOTICE:\n{LIMITATIONS_NOTICE}")
    print("=" * 80 + "\n")


def find_image_files(input_path: Path) -> List[Path]:
    """Find supported image files from a file path or directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if input_path.is_file():
        if input_path.suffix.lower() not in valid_exts:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    elif input_path.is_dir():
        found = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in valid_exts]
        if not found:
            raise FileNotFoundError(f"No supported images ({valid_exts}) found in directory: {input_path}")
        return found
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_predict(args: argparse.Namespace) -> int:
    """Execute prediction pipeline."""
    print_banner()

    # Determine confidence threshold
    model_name_input = args.model
    if args.conf.lower() == "auto":
        info = model_info(model_name_input)
        thresh = info.get("best_f1_conf")
        if thresh is None:
            print(
                f"Error: Model '{info['name']}' has no documented best-F1 threshold. "
                f"Please specify --conf explicitly (e.g. --conf 0.10).",
                file=sys.stderr,
            )
            return 1
        conf_val = float(thresh)
        print(f"[*] Using documented best-F1 confidence threshold for {info['name']}: {conf_val:.3f}")
    else:
        try:
            conf_val = float(args.conf)
        except ValueError:
            print(f"Error: Invalid confidence value '{args.conf}'. Must be a float or 'auto'.", file=sys.stderr)
            return 1

    input_path = Path(args.input)
    try:
        image_paths = find_image_files(input_path)
    except Exception as e:
        print(f"Error resolving input images: {e}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading model '{model_name_input}' (device={args.device}, half={args.half})...")
    try:
        model, canonical_name = load_model(
            model_name_input, device=args.device, half=args.half
        )
    except Exception as e:
        print(f"Error loading model '{model_name_input}': {e}", file=sys.stderr)
        return 1

    print(f"[*] Loaded model: {canonical_name}")
    print(f"[*] Processing {len(image_paths)} image(s)...")

    success_count = 0
    failure_count = 0

    for img_p in image_paths:
        t0 = time.perf_counter()
        try:
            rgb_img, img_meta = read_image(img_p)
            read_time = time.perf_counter() - t0

            result = detect_full_image(
                model=model,
                image=rgb_img,
                conf=conf_val,
                iou=args.iou,
                tile=args.tile,
                overlap=args.overlap,
                batch=args.batch,
                max_det=args.max_det,
                half=args.half,
                drop_border_boxes=args.drop_border_boxes,
                model_name=canonical_name,
            )
            result.timing["read"] = round(read_time, 4)
            result.timing["total"] = round(time.perf_counter() - t0, 4)

            stem = img_p.stem

            # Always save JSON metadata
            json_path = output_dir / f"{stem}_meta.json"
            export_json(result, img_meta, json_path)

            if args.save_overlay:
                overlay_path = output_dir / f"{stem}_overlay.png"
                export_overlay(rgb_img, result, overlay_path)

            if args.save_csv:
                csv_path = output_dir / f"{stem}_detections.csv"
                export_csv(result, csv_path)

            if args.save_yolo:
                yolo_path = output_dir / f"{stem}_yolo.txt"
                export_yolo(result, rgb_img.shape[:2], yolo_path)

            print(
                f"[OK] {img_p.name}: {result.count} cells detected across "
                f"{result.n_tiles} tiles in {result.timing['total']:.2f}s"
            )
            success_count += 1

        except Exception as e:
            print(f"[FAIL] {img_p.name}: {e}", file=sys.stderr)
            failure_count += 1

    print(f"\n[*] Complete: {success_count} succeeded, {failure_count} failed. Outputs saved in {output_dir}")
    return 0 if failure_count == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m plantcell_tool",
        description="Overlapping tile inference for high-resolution plant-cell micrographs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # predict subcommand
    pred_parser = subparsers.add_parser("predict", help="Run inference on an image or directory.")
    pred_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an input image file or directory of images.",
    )
    pred_parser.add_argument(
        "--model",
        type=str,
        default="E5a",
        help="Model name ('E5a' [default], 'E3' [fastest], 'E1', 'E8', etc.) or path to .pt file.",
    )
    pred_parser.add_argument(
        "--conf",
        type=str,
        default="auto",
        help="Confidence threshold: 'auto' uses documented per-model threshold (E5a: 0.063, E3: 0.114, E1: 0.108), or specify a float.",
    )
    pred_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_tool/run1",
        help="Directory to save output files.",
    )
    pred_parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for cross-tile NMS (default: 0.45).")
    pred_parser.add_argument("--tile", type=int, default=640, help="Tile size in pixels (default: 640).")
    pred_parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap fraction (default: 0.25).")
    pred_parser.add_argument("--batch", type=int, default=16, help="Batch size for tile inference (default: 16).")
    pred_parser.add_argument("--max-det", type=int, default=300, help="Max detections per tile (default: 300).")
    pred_parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu').")
    pred_parser.add_argument("--half", action="store_true", help="Enable FP16 half precision on GPU.")
    pred_parser.add_argument(
        "--drop-border-boxes",
        action="store_true",
        help="Discard boxes within 3px of interior tile edges before NMS.",
    )
    pred_parser.add_argument("--save-overlay", action="store_true", help="Save annotated overlay PNG.")
    pred_parser.add_argument("--save-csv", action="store_true", help="Save CSV of detections in pixel coordinates.")
    pred_parser.add_argument("--save-yolo", action="store_true", help="Save normalized YOLO format txt.")

    # info subcommand
    info_parser = subparsers.add_parser("info", help="Inspect model checkpoint metadata and thresholds.")
    info_parser.add_argument("model", type=str, help="Model name ('E1', 'E3', 'E5a', 'E8') or .pt path.")

    return parser


def run_info(args: argparse.Namespace) -> int:
    """Print model metadata and best-F1 threshold."""
    try:
        info = model_info(args.model)
    except Exception as e:
        print(f"Error inspecting model: {e}", file=sys.stderr)
        return 1

    print("\n--- MODEL METADATA ---")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")
    print("----------------------\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        return run_predict(args)
    elif args.command == "info":
        return run_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
