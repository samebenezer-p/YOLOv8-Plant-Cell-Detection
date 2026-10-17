"""
E1: Stock YOLOv8 baseline detector wrapper.

Wraps the Ultralytics YOLO API with a standardized interface for training,
prediction, and results export. All other ablation stages build on or
extend this baseline.

Includes memory-safe training features for 4GB VRAM GPUs:
- Automatic OOM retry with batch size halving
- Peak VRAM logging per epoch
- Mixed precision (AMP), disk caching, reduced workers
- Gradient accumulation for effective batch size
- Early stopping with configurable patience

Usage:
    from src.models.yolov8_baseline import BaselineDetector
    detector = BaselineDetector("yolov8n.pt")
    detector.train(data_yaml="data.yaml", epochs=100)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# VRAM logging callback
# ---------------------------------------------------------------------------

def _vram_epoch_end_callback(trainer):
    """Ultralytics callback: log peak VRAM at the end of each epoch."""
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        current_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        epoch = trainer.epoch + 1
        print(f"  [VRAM] Epoch {epoch}: peak={peak_mb:.1f} MB, current={current_mb:.1f} MB")

        # Persist to a log file next to the training outputs
        vram_log = Path(trainer.save_dir) / "vram_log.jsonl"
        with open(vram_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "peak_vram_mb": round(peak_mb, 1),
                "current_vram_mb": round(current_mb, 1),
            }) + "\n")

        # Reset peak stats for next epoch
        torch.cuda.reset_peak_memory_stats()


class BaselineDetector:
    """Stock YOLOv8 wrapper for the E1 baseline experiment.

    Parameters
    ----------
    model_variant : str
        YOLOv8 model name or path (e.g., ``"yolov8n.pt"``).
    num_classes : int
        Number of object classes.
    device : str, optional
        Torch device. ``None`` for auto-detection.
    """

    def __init__(
        self,
        model_variant: str = "yolov8n.pt",
        num_classes: int = 3,
        device: Optional[str] = None,
    ):
        self.model_variant = model_variant
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load or initialize the YOLO model."""
        self.model = YOLO(self.model_variant)

    def train(
        self,
        data_yaml: str,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train the model with memory-safe OOM retry logic.

        Parameters
        ----------
        data_yaml : str
            Path to the data.yaml file.
        cfg : dict
            Training configuration. Expected keys (with defaults):
            - ``epochs`` (int): 100
            - ``batch_size`` (int): 4
            - ``imgsz`` (int): 640
            - ``optimizer`` (str): "AdamW"
            - ``lr0`` (float): 0.001
            - ``lrf`` (float): 0.01
            - ``seed`` (int): 42
            - ``project_dir`` (str): "results"
            - ``experiment_name`` (str): "e1_baseline"
            - ``augmentation`` (dict): augmentation settings
            - ``resume`` (bool): False
            - ``accumulate`` (int): gradient accumulation steps
            - ``patience`` (int): early stopping patience
            - ``close_mosaic`` (int): epochs before end to close mosaic

        Returns
        -------
        dict
            Training results summary from Ultralytics.
        """
        assert self.model is not None, "Model not loaded"

        batch_size = cfg.get("batch_size", 4)
        max_retries = cfg.get("oom_max_retries", 3)

        for attempt in range(max_retries + 1):
            try:
                train_args = self._build_train_args(data_yaml, cfg, batch_size)

                # Register VRAM logging callback
                if torch.cuda.is_available():
                    self.model.add_callback(
                        "on_train_epoch_end", _vram_epoch_end_callback
                    )

                # Train
                results = self.model.train(**train_args)
                return self._extract_train_results(results, train_args)

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < max_retries:
                    old_batch = batch_size
                    batch_size = max(1, batch_size // 2)
                    print(f"\n{'!'*60}")
                    print(f"  CUDA OOM detected! (attempt {attempt + 1}/{max_retries})")
                    print(f"  Halving batch size: {old_batch} -> {batch_size}")
                    print(f"  Clearing CUDA cache and retrying with resume=True...")
                    print(f"{'!'*60}\n")

                    torch.cuda.empty_cache()
                    cfg["resume"] = True

                    # Re-load model from last checkpoint if available
                    save_dir = Path(cfg.get("project_dir", "results")) / cfg.get("experiment_name", "run")
                    last_pt = save_dir / "weights" / "last.pt"
                    if last_pt.exists():
                        self.model = YOLO(str(last_pt))
                    else:
                        # Reset model for fresh retry
                        self._load_model()
                else:
                    raise

        # Should not reach here, but just in case
        raise RuntimeError(f"Training failed after {max_retries} OOM retries")

    def _build_train_args(
        self,
        data_yaml: str,
        cfg: Dict[str, Any],
        batch_size: int,
    ) -> Dict[str, Any]:
        """Build the Ultralytics training argument dict.

        Enforces memory-safe defaults for 4GB VRAM GPUs.
        """
        train_args: Dict[str, Any] = {
            "data": data_yaml,
            "epochs": cfg.get("epochs", 100),
            "batch": batch_size,
            "imgsz": cfg.get("imgsz", 640),
            "optimizer": cfg.get("optimizer", "AdamW"),
            "lr0": cfg.get("lr0", 0.001),
            "lrf": cfg.get("lrf", 0.01),
            "seed": cfg.get("seed", 42),
            "device": self.device,
            "project": cfg.get("project_dir", "results"),
            "name": cfg.get("experiment_name", "e1_baseline"),
            "exist_ok": True,
            "verbose": True,
            "save": True,
            "save_period": -1,  # save only best + last
            "plots": True,
            "resume": cfg.get("resume", False),
            # --- Memory-safe defaults ---
            "amp": cfg.get("amp", True),              # Mixed precision
            "cache": cfg.get("cache", False),          # No RAM caching
            "workers": cfg.get("workers", 2),          # Low worker count
            "close_mosaic": cfg.get("close_mosaic", 10),  # Close mosaic last 10 epochs
            "patience": cfg.get("patience", 20),       # Early stopping
        }

        # Gradient accumulation: Ultralytics uses 'nbs' (nominal batch size)
        # to auto-compute accumulation. If accumulate is specified, we set
        # nbs = batch_size * accumulate so Ultralytics computes the right
        # accumulation factor internally.
        accumulate = cfg.get("accumulate")
        if accumulate and accumulate > 1:
            train_args["nbs"] = batch_size * accumulate

        # Augmentation overrides
        aug = cfg.get("augmentation", {})
        for key in [
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
            "scale", "fliplr", "flipud", "mosaic", "mixup",
        ]:
            if key in aug:
                train_args[key] = aug[key]

        return train_args

    def _extract_train_results(
        self,
        results: Any,
        train_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract a serializable results summary from training."""
        save_dir = Path(train_args["project"]) / train_args["name"]
        best_pt = save_dir / "weights" / "best.pt"

        result = {
            "experiment_name": train_args["name"],
            "model_variant": self.model_variant,
            "epochs": train_args["epochs"],
            "batch_size": train_args["batch"],
            "best_weights": str(best_pt) if best_pt.exists() else None,
            "save_dir": str(save_dir),
        }

        # Include peak VRAM info if available
        vram_log = save_dir / "vram_log.jsonl"
        if vram_log.exists():
            try:
                vram_entries = []
                with open(vram_log, "r") as f:
                    for line in f:
                        vram_entries.append(json.loads(line.strip()))
                if vram_entries:
                    peaks = [e["peak_vram_mb"] for e in vram_entries]
                    result["vram_peak_max_mb"] = max(peaks)
                    result["vram_peak_mean_mb"] = round(sum(peaks) / len(peaks), 1)
            except Exception:
                pass

        return result

    def predict(
        self,
        source: Union[str, np.ndarray, List[str]],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
    ) -> List[Dict[str, Any]]:
        """Run inference on images.

        Parameters
        ----------
        source : str, np.ndarray, or list of str
            Image path(s) or numpy array.
        conf : float
            Confidence threshold.
        iou : float
            NMS IoU threshold.
        imgsz : int
            Inference image size.
        max_det : int
            Max detections per image.

        Returns
        -------
        list of dict
            Each dict has keys: ``"image"``, ``"detections"`` (list of
            dicts with bbox, confidence, class_id).
        """
        assert self.model is not None, "Model not loaded"

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=self.device,
            verbose=False,
        )

        predictions: List[Dict[str, Any]] = []
        for r in results:
            boxes = r.boxes
            det_list = []
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    c = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    det_list.append({
                        "bbox": xyxy,
                        "confidence": c,
                        "class_id": cls,
                    })
            predictions.append({
                "image": r.path if hasattr(r, "path") else str(source),
                "detections": det_list,
            })

        return predictions

    def load_weights(self, weights_path: str) -> None:
        """Load trained weights from a checkpoint.

        Parameters
        ----------
        weights_path : str
            Path to a ``.pt`` weights file.
        """
        self.model = YOLO(weights_path)

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Write results dict to a JSON file.

        Parameters
        ----------
        results : dict
            Results dictionary to serialize.
        output_path : path-like
            Output JSON file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")

    @property
    def info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_variant": self.model_variant,
            "num_classes": self.num_classes,
            "device": self.device,
        }
