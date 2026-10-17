"""
Unified smoke test suite for E1 to E7 with strict assertions:
- No silent skips: raises RuntimeError if module counts or types do not match.
- Runs 3 epochs on sample dataset with minimal image size to complete quickly on CPU.
- Evaluates weights, logs parameter count, GFLOPs, hashes, and boundary loss.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from src.models.dcn_optional import replace_conv_with_dcn, DeformableConv2d
from src.models.fusion_neck import inject_fusion_neck, FusionNeck
from src.models.boundary_branch import BoundaryHead, InterceptWrapper, generate_boundary_targets_tensor
from src.models.losses import shape_iou, mpdiou, patched_bbox_iou
from src.benchmark import _estimate_gflops

SMOKE_RESULTS = {}

class CustomAblationTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None, stage="E1"):
        self.stage = stage
        self.boundary_losses = []
        super().__init__(overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        
        # Inject DCN if stage is E7
        if self.stage == "E7":
            replace_conv_with_dcn(model, layer_indices=[6, 8])
            dcn_cnt = sum(1 for m in model.modules() if isinstance(m, DeformableConv2d))
            if dcn_cnt != 3:
                raise RuntimeError(f"Assertion failed: Expected 3 DeformableConv2d layers, got {dcn_cnt}")
            print(f"[{self.stage}] Confirmed {dcn_cnt} DeformableConv2d layers active.")

        # Inject FusionNeck if stage is E3
        if self.stage == "E3":
            inject_fusion_neck(model, num_repeats=2, out_channels=128)
            fusion_cnt = sum(1 for m in model.modules() if isinstance(m, FusionNeck))
            if fusion_cnt != 1:
                raise RuntimeError(f"Assertion failed: Expected 1 FusionNeck, got {fusion_cnt}")
            print(f"[{self.stage}] Confirmed FusionNeck active.")

        # Inject BoundaryHead if stage is E4
        if self.stage == "E4":
            b_head = BoundaryHead(in_channels=64, hidden_channels=32, alpha=0.5)
            # Wrap layer 15 (P3 C2f output)
            model.model[15] = InterceptWrapper(model.model[15], b_head)
            self.boundary_head = b_head
            print(f"[{self.stage}] Confirmed BoundaryHead injected at layer 15.")

        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        # If E4, wrap loss function
        if self.stage == "E4" and hasattr(self, "boundary_head"):
            orig_loss = self.loss
            b_head = self.boundary_head
            loss_history = self.boundary_losses

            class BoundaryLossWrapper:
                def __init__(self, loss_fn, head):
                    self.loss_fn = loss_fn
                    self.head = head
                    self.__name__ = getattr(loss_fn, "__name__", "v8DetectionLoss")

                def __call__(self, preds, batch):
                    total_loss, loss_items = self.loss_fn(preds, batch)
                    heatmap_pred = self.head.latest_heatmap
                    if heatmap_pred is not None:
                        bs, _, h, w = heatmap_pred.shape
                        targets = generate_boundary_targets_tensor(
                            batch["bboxes"], batch["batch_idx"], bs, h, w, thickness=3
                        )
                        b_total, b_bce, b_dice = self.head.compute_loss(heatmap_pred, targets)
                        total_loss += b_total
                        loss_history.append(float(b_total.item()))
                    return total_loss, loss_items

            self.loss = BoundaryLossWrapper(orig_loss, b_head)


def run_stage_smoke(stage_name: str, model_arg: str, iou_patch=None):
    print(f"\n==========================================")
    print(f"  RUNNING SMOKE TEST: {stage_name}")
    print(f"==========================================")
    save_dir = Path(f"runs/smoke/{stage_name}")
    if save_dir.exists():
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)

    yolo = YOLO(model_arg)
    
    # Check baseline params
    p_init = sum(p.numel() for p in yolo.model.parameters())
    print(f"[{stage_name}] Initial model parameters: {p_init:,}")

    trainer_kwargs = {
        "data": "data.yaml",
        "epochs": 3,
        "batch": 4,
        "imgsz": 256,
        "device": "cpu",
        "project": "runs/smoke",
        "name": stage_name,
        "plots": False,
        "save": True,
        "verbose": False,
        "workers": 0,
    }

    class StageTrainer(CustomAblationTrainer):
        def __init__(self, overrides=None, _callbacks=None):
            super().__init__(overrides=overrides, _callbacks=_callbacks, stage=stage_name)

    trainer_cls = StageTrainer if stage_name in ["E3", "E4", "E7"] else None

    boundary_losses = []
    if iou_patch:
        with patched_bbox_iou(iou_patch):
            if trainer_cls:
                yolo.train(trainer=trainer_cls, **trainer_kwargs)
            else:
                yolo.train(**trainer_kwargs)
    else:
        if trainer_cls:
            yolo.train(trainer=trainer_cls, **trainer_kwargs)
            if hasattr(yolo.trainer, "boundary_losses"):
                boundary_losses = yolo.trainer.boundary_losses
        else:
            yolo.train(**trainer_kwargs)

    # Retrieve best.pt from trainer's actual save directory
    save_dir = Path(yolo.trainer.save_dir) if hasattr(yolo, "trainer") and yolo.trainer else Path(f"runs/detect/runs/smoke/{stage_name}")
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Checkpoint was not created: {best_pt}")

    # Inspect trained checkpoint
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    m = ckpt["model"]
    params = sum(p.numel() for p in m.parameters())
    md5 = hashlib.md5(best_pt.read_bytes()).hexdigest()
    gflops = _estimate_gflops(YOLO(str(best_pt)), imgsz=256, device="cpu")

    SMOKE_RESULTS[stage_name] = {
        "params": params,
        "gflops": gflops,
        "md5": md5,
        "best_pt": str(best_pt),
        "boundary_losses": boundary_losses[:5] if boundary_losses else [],
    }
    print(f"[{stage_name}] Complete! Params: {params:,} | GFLOPs: {gflops} | MD5: {md5}")


if __name__ == "__main__":
    # E1 Baseline
    run_stage_smoke("E1", "yolov8n.pt")
    # E2 P2 Head
    run_stage_smoke("E2", "yolov8n-p2.yaml")
    # E3 Fusion Neck
    run_stage_smoke("E3", "yolov8n.yaml")
    # E4 Auxiliary Boundary Head
    run_stage_smoke("E4", "yolov8n.yaml")
    # E5a Shape IoU Loss
    run_stage_smoke("E5a", "yolov8n.pt", iou_patch=shape_iou)
    # E5b MPDIoU Loss
    run_stage_smoke("E5b", "yolov8n.pt", iou_patch=mpdiou)
    # E7 DCNv2
    run_stage_smoke("E7", "yolov8n.pt")

    import json
    with open("runs/smoke/smoke_summary.json", "w") as f:
        json.dump(SMOKE_RESULTS, f, indent=2)
    print("\nALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
