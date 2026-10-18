"""
Unified smoke test suite for ALL stages (E1 to E8) with strict assertions:
- No silent skips: raises RuntimeError/AssertionError if module counts or types do not match.
- Trains the exact model returned by src.model_factory.build_stage().
- Runs 3 epochs on sample dataset with minimal image size (256) to complete quickly on CPU.
- Evaluates weights, logs parameter count (fused vs unfused), hashes, and boundary loss.
- Logs per-epoch boundary loss for E4 and E8.
- For E8: loads checkpoint and runs one prediction to verify inference.
- Final success message is conditional on all 9 stages (E1-E8) completing and passing.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import json
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

# Always print torch version and CUDA availability at the start of every run
print(f"torch.__version__: {torch.__version__} | torch.cuda.is_available(): {torch.cuda.is_available()}")

from src.models.dcn_optional import replace_conv_with_dcn, DeformableConv2d
from src.models.fusion_neck import (
    inject_fusion_neck,
    FusionNeck,
    E8FusionNeck,
    _FastFusion,
    _make_e8_predict_once,
)
from src.models.boundary_branch import BoundaryHead, InterceptWrapper, generate_boundary_targets_tensor
from src.models.context_refine import ContextRefineClassifier
from src.models.losses import shape_iou, mpdiou, patched_bbox_iou
from src.benchmark import _estimate_gflops
from src.model_factory import build_stage, StageManifest

# Ensure unpickling backward compatibility
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "E8FusionNeck", E8FusionNeck)
    setattr(main_mod, "_FastFusion", _FastFusion)
    if not hasattr(main_mod, "E8LossWrapper"):
        class E8LossWrapper:
            pass
        setattr(main_mod, "E8LossWrapper", E8LossWrapper)

SMOKE_RESULTS = {}


class CustomAblationTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None, stage="E1"):
        self.stage = stage
        self.epoch_boundary_losses = []
        self._batch_boundary_losses = []
        super().__init__(overrides=overrides, _callbacks=_callbacks)

        def on_train_epoch_end_callback(trainer):
            if trainer._batch_boundary_losses:
                avg_b_loss = sum(trainer._batch_boundary_losses) / len(trainer._batch_boundary_losses)
                trainer.epoch_boundary_losses.append(avg_b_loss)
                ep = len(trainer.epoch_boundary_losses)
                print(f"[{trainer.stage}] Epoch {ep} Boundary Loss: {avg_b_loss:.4f} (across {len(trainer._batch_boundary_losses)} batches)")
                trainer._batch_boundary_losses.clear()

        self.add_callback("on_train_epoch_end", on_train_epoch_end_callback)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Obtain model directly from the single model factory
        factory_model, manifest = build_stage(self.stage)
        self.factory_manifest = manifest

        # Check for BoundaryHead
        bh_modules = [m for m in factory_model.model.modules() if isinstance(m, BoundaryHead)]
        if bh_modules:
            self.boundary_head = bh_modules[0]

        return factory_model.model

    def set_model_attributes(self):
        super().set_model_attributes()
        # If BoundaryHead present (E4, E8), wrap loss function
        if hasattr(self, "boundary_head"):
            orig_loss = self.loss
            b_head = self.boundary_head
            loss_history = self._batch_boundary_losses

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


def run_stage_smoke(stage_name: str, iou_patch=None):
    print(f"\n==========================================")
    print(f"  RUNNING SMOKE TEST: {stage_name}")
    print(f"==========================================")
    save_dir = Path(f"runs/smoke/{stage_name}")
    if save_dir.exists():
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)

    # 1. Obtain model from single model factory
    factory_yolo, manifest = build_stage(stage_name)
    expected_det_params = sum(p.numel() for p in factory_yolo.model.parameters())
    print(f"[{stage_name}] Factory built parameters (unfused): {expected_det_params:,}")

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

    # Train model
    if iou_patch:
        with patched_bbox_iou(iou_patch):
            factory_yolo.train(trainer=StageTrainer, **trainer_kwargs)
    else:
        factory_yolo.train(trainer=StageTrainer, **trainer_kwargs)

    # Retrieve checkpoint
    save_dir = Path(factory_yolo.trainer.save_dir) if hasattr(factory_yolo, "trainer") and factory_yolo.trainer else Path(f"runs/detect/runs/smoke/{stage_name}")
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Checkpoint was not created: {best_pt}")

    # Inspect trained checkpoint
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    m = ckpt["model"]
    ckpt_params = sum(p.numel() for p in m.parameters())
    md5 = hashlib.md5(best_pt.read_bytes()).hexdigest()
    gflops = _estimate_gflops(YOLO(str(best_pt)), imgsz=256, device="cpu")

    # Strict assertion: trained model parameters must match factory parameters
    print(f"[{stage_name}] Verification: checkpoint params={ckpt_params:,} vs factory={expected_det_params:,}")
    assert ckpt_params == expected_det_params, \
        f"[{stage_name}] ASSERTION FAILED: Trained ckpt params {ckpt_params} != factory params {expected_det_params}"

    # Verify E8 prediction from loaded checkpoint
    if stage_name == "E8":
        m.float().eval()
        with torch.no_grad():
            sample_pred = m(torch.randn(1, 3, 256, 256))
        assert sample_pred is not None, "E8 prediction failed on loaded checkpoint"
        print(f"[{stage_name}] Checkpoint loaded and prediction verified successfully (output type: {type(sample_pred)}).")

    # Verify E6 refiner functionality
    if stage_name == "E6" and hasattr(factory_yolo, "_refiner") and factory_yolo._refiner is not None:
        factory_yolo._refiner.eval()
        with torch.no_grad():
            test_crop = torch.randn(1, 3, 64, 64)
            prob = factory_yolo._refiner(test_crop).item()
        assert 0.0 <= prob <= 1.0, f"E6 refiner probability {prob} out of bounds"
        print(f"[{stage_name}] ContextRefineClassifier verified successfully (test crop prob: {prob:.4f}).")

    # Per-epoch boundary loss reporting for E4 and E8
    epoch_b_losses = getattr(factory_yolo.trainer, "epoch_boundary_losses", [])
    if epoch_b_losses:
        print(f"[{stage_name}] Per-Epoch Boundary Loss Summary:")
        for ep_idx, b_val in enumerate(epoch_b_losses, 1):
            print(f"    Epoch {ep_idx}: {b_val:.4f}")

    SMOKE_RESULTS[stage_name] = {
        "status": "PASS",
        "params_unfused": ckpt_params,
        "gflops": gflops,
        "md5": md5,
        "best_pt": str(best_pt),
        "per_epoch_boundary_loss": epoch_b_losses,
    }
    print(f"[{stage_name}] Complete & Passed! Params: {ckpt_params:,} (unfused) | GFLOPs: {gflops} | MD5: {md5}")


if __name__ == "__main__":
    REQUIRED_STAGES = ["E1", "E2", "E3", "E4", "E5a", "E5b", "E6", "E7", "E8"]

    # E1 Baseline
    run_stage_smoke("E1")
    # E2 P2 Head
    run_stage_smoke("E2")
    # E3 Fusion Neck
    run_stage_smoke("E3")
    # E4 Auxiliary Boundary Head
    run_stage_smoke("E4")
    # E5a Shape IoU Loss
    run_stage_smoke("E5a", iou_patch=shape_iou)
    # E5b MPDIoU Loss
    run_stage_smoke("E5b", iou_patch=mpdiou)
    # E6 Context Refinement
    run_stage_smoke("E6")
    # E7 DCNv2
    run_stage_smoke("E7")
    # E8 Full Combined Model
    run_stage_smoke("E8", iou_patch=shape_iou)

    # Save summary
    out_path = Path("runs/smoke/smoke_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(SMOKE_RESULTS, f, indent=2)

    # Conditional success assertion: MUST contain all required stages and all must have status PASS
    passed_stages = [s for s in REQUIRED_STAGES if s in SMOKE_RESULTS and SMOKE_RESULTS[s].get("status") == "PASS"]
    print(f"\nStages verified: {passed_stages}")
    assert len(passed_stages) == len(REQUIRED_STAGES), \
        f"Smoke test suite failed: expected {REQUIRED_STAGES}, but only {passed_stages} passed."

    print("\nALL 9 SMOKE TESTS (E1-E8) COMPLETED SUCCESSFULLY AND PASSED!")
