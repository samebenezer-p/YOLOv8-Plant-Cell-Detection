"""
Evaluation protocol audit script.

Prints the exact evaluation settings (confidence threshold, NMS IoU, max_det,
tiling settings, imgsz) used by each experiment stage, sourced from:
  1. The per-stage YAML config files in configs/
  2. The saved results.json (if available) in results/

Usage:
    python src/eval_protocol_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ──────────────────────────────────────────────────────────────────────────────

STAGES = [
    ("E1", "e1_baseline"),
    ("E2", "e2_tiling_p2"),
    ("E3", "e3_fusion_neck"),
    ("E4", "e4_boundary"),
    ("E5a", "e5a_shapeiou"),
    ("E5b", "e5b_mpdiou"),
    ("E6", "e6_context_refine"),
    ("E7", "e7_dcn_optional"),
    ("E8", "e8_full_model"),
]


def load_config_settings(config_path: Path) -> dict:
    """Load relevant eval settings from a YAML config."""
    if not config_path.exists():
        return {"status": "CONFIG_MISSING"}
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    settings = {
        "conf_thresh": cfg.get("conf_thresh", "NOT_SET"),
        "iou_thresh": cfg.get("iou_thresh", "NOT_SET"),
        "imgsz": cfg.get("imgsz", "NOT_SET"),
        "epochs": cfg.get("epochs", "NOT_SET"),
        "batch_size": cfg.get("batch_size", "NOT_SET"),
        # Tiling is only active in E2 (uses yolov8n-p2.yaml which adds stride-4 head;
        # but explicit tile-based sliding window is not currently in any config)
        "tiling_enabled": cfg.get("tiling", {}).get("enabled", False),
        "tile_size": cfg.get("tiling", {}).get("tile_size", "N/A"),
        "tile_overlap": cfg.get("tiling", {}).get("overlap", "N/A"),
        # max_det is an Ultralytics default — not set in our configs => uses YOLO default of 300
        "max_det": cfg.get("max_det", "default_300"),
        "model_variant": cfg.get("model_variant", "?"),
        "optimizer": cfg.get("optimizer", "?"),
        "lr0": cfg.get("lr0", "?"),
        "lrf": cfg.get("lrf", "?"),
    }
    return settings


def load_runtime_settings(results_json: Path) -> dict:
    """Load evaluation settings actually used at runtime from results.json."""
    if not results_json.exists():
        return {"status": "RESULTS_MISSING"}
    with open(results_json) as f:
        data = json.load(f)

    # Dig into the saved config if present
    saved_cfg = data.get("config", {})
    eval_cfg = data.get("evaluation", {})

    return {
        "runtime_conf": eval_cfg.get("conf_thresh", saved_cfg.get("conf_thresh", "?")),
        "runtime_iou": eval_cfg.get("iou_thresh", saved_cfg.get("iou_thresh", "?")),
        "runtime_split": eval_cfg.get("split", "?"),
        "runtime_mAP50": eval_cfg.get("mAP50", "?"),
        "runtime_mAP50_95": eval_cfg.get("mAP50_95", "?"),
    }


def audit_evaluation_protocol():
    configs_dir = PROJECT_ROOT / "configs"
    results_dir = PROJECT_ROOT / "results"

    print("=" * 80)
    print("  EVALUATION PROTOCOL AUDIT")
    print("  Source: configs/*.yaml + results/*/seed_42/results.json")
    print("=" * 80)
    print()

    all_records = []

    for stage_id, name in STAGES:
        config_path = configs_dir / f"{name}.yaml"
        results_json = results_dir / name / "seed_42" / "results.json"

        cfg_settings = load_config_settings(config_path)
        runtime = load_runtime_settings(results_json)

        record = {
            "stage": stage_id,
            "experiment": name,
            **cfg_settings,
            **runtime,
        }
        all_records.append(record)

        print(f"  Stage: {stage_id} ({name})")
        print(f"    Config file:       {config_path}")
        print(f"    --- Config settings ---")
        print(f"    conf_thresh:       {cfg_settings['conf_thresh']}")
        print(f"    iou_thresh (NMS):  {cfg_settings['iou_thresh']}")
        print(f"    imgsz:             {cfg_settings['imgsz']}")
        print(f"    max_det:           {cfg_settings['max_det']}")
        print(f"    tiling_enabled:    {cfg_settings['tiling_enabled']}")
        print(f"    tile_size:         {cfg_settings['tile_size']}")
        print(f"    tile_overlap:      {cfg_settings['tile_overlap']}")
        print(f"    model_variant:     {cfg_settings['model_variant']}")
        print(f"    optimizer:         {cfg_settings['optimizer']}")
        print(f"    lr0:               {cfg_settings['lr0']}")
        print(f"    lrf:               {cfg_settings['lrf']}")
        print(f"    --- Runtime (from results.json) ---")
        print(f"    runtime_conf:      {runtime.get('runtime_conf', 'N/A')}")
        print(f"    runtime_iou:       {runtime.get('runtime_iou', 'N/A')}")
        print(f"    runtime_split:     {runtime.get('runtime_split', 'N/A')}")
        print(f"    runtime_mAP50:     {runtime.get('runtime_mAP50', 'N/A')}")
        print(f"    runtime_mAP50_95:  {runtime.get('runtime_mAP50_95', 'N/A')}")
        print()

    # ── Consistency check ──────────────────────────────────────────────
    print("=" * 80)
    print("  CONSISTENCY CHECK (E1 vs all other stages)")
    print("=" * 80)

    ref = None
    for r in all_records:
        if r["stage"] == "E1":
            ref = r
            break

    if ref is None:
        print("  ERROR: E1 record not found in configs.")
        return all_records

    fairness_keys = ["conf_thresh", "iou_thresh", "imgsz", "optimizer", "lr0", "lrf"]
    all_ok = True

    for r in all_records:
        if r["stage"] == "E1":
            continue
        mismatches = []
        for k in fairness_keys:
            ref_val = ref.get(k)
            stage_val = r.get(k)
            if str(ref_val) != str(stage_val):
                mismatches.append(f"{k}: E1={ref_val} vs {r['stage']}={stage_val}")
        if mismatches:
            print(f"  MISMATCH in {r['stage']} ({r['experiment']}):")
            for m in mismatches:
                print(f"    - {m}")
            all_ok = False
        else:
            print(f"  OK: {r['stage']} matches E1 on all fairness-protocol fields.")

    if all_ok:
        print("\n  All stages pass the fairness-protocol consistency check.")

    # ── E2 tiling note ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  TILING SETTINGS NOTE")
    print("=" * 80)
    print("""
  E2 uses the yolov8n-p2.yaml architecture which adds a stride-4 detection
  head (P2 level) to detect smaller objects within a single forward pass.
  This is NOT a sliding-window tile approach.

  True sliding-window tiling (splitting large images into overlapping tiles
  and running separate forward passes) is NOT currently enabled in any stage.
  The 'tiling_enabled' field in all configs is False / absent.

  If sliding-window tiling is needed for real data (large microscopy images),
  set:
    tiling:
      enabled: true
      tile_size: 640
      overlap: 0.2   # 20% overlap
  in the relevant config and implement the tile-stitching logic in evaluate.py.

  max_det:
  Ultralytics default is 300 detections per image. None of the E1-E8 configs
  override this, so all stages use max_det=300.
""")

    return all_records


if __name__ == "__main__":
    audit_evaluation_protocol()
