"""
E6 Context Refiner — Hard-Negative Retraining Script.

Mines hard negatives from the TRAIN split:
  1. Fiber crops (class 1 GT boxes) — morphologically similar to parenchyma
  2. Background crops (random non-overlapping patches)
  3. False-positive detector boxes mined from TRAIN predictions
     (boxes predicted as parenchyma that have IoU < 0.3 with any GT parenchyma)

Training:
  - Positive:  GT parenchyma crops + margin (class 2)
  - Negative:  fiber crops + background patches + mined FP crops

Evaluation on VAL split:
  - Crop-level accuracy and AUC-ROC
  - Probability distribution histogram (checks for pathological clustering near 0.56)
  - Precision / recall / F1 at 0.5 threshold

Run:
    python src/train_e6_refiner.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

from src.models.context_refine import ContextRefineClassifier, extract_crops_from_gt


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_YAML = "data.yaml"
CROP_SIZE = 64
MARGIN_RATIO = 0.25
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = "cpu"
CLASSIFIER_OUT = "results/context_refine/classifier_hardneg.pt"
RESULTS_OUT = "results/context_refine/hardneg_results.json"
SEED = 42

# Parenchyma class id
PARENCHYMA_CLS = 2
FIBER_CLS = 1

# IOU threshold below which a parenchyma prediction counts as FP
FP_IOU_THRESH = 0.3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# IoU helper
# ──────────────────────────────────────────────────────────────────────────────

def _iou_xyxy(b1, b2):
    """Compute IoU between two xyxy boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _max_iou_vs_gt(det_box, gt_boxes):
    """Max IoU of a detection box vs. a list of GT boxes (xyxy)."""
    if not gt_boxes:
        return 0.0
    return max(_iou_xyxy(det_box, g) for g in gt_boxes)


# ──────────────────────────────────────────────────────────────────────────────
# Crop extraction
# ──────────────────────────────────────────────────────────────────────────────

def _crop_box(img, cx, cy, bw, bh, margin_ratio, crop_size):
    """Extract a crop with margin from a normalised YOLO box."""
    H, W = img.shape[:2]
    mx = bw * margin_ratio
    my = bh * margin_ratio
    x1 = int(max(0, (cx - bw / 2 - mx) * W))
    y1 = int(max(0, (cy - bh / 2 - my) * H))
    x2 = int(min(W, (cx + bw / 2 + mx) * W))
    y2 = int(min(H, (cy + bh / 2 + my) * H))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (crop_size, crop_size))


def _random_bg_crop(img, gt_boxes_xyxy, crop_size, min_iou=0.05, tries=20):
    """Sample a random crop that does not overlap significantly with any GT box."""
    H, W = img.shape[:2]
    size = crop_size  # square crop
    for _ in range(tries):
        x1 = random.randint(0, max(0, W - size))
        y1 = random.randint(0, max(0, H - size))
        x2, y2 = x1 + size, y1 + size
        box = [x1, y1, x2, y2]
        if _max_iou_vs_gt(box, gt_boxes_xyxy) < min_iou:
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                return cv2.resize(crop, (crop_size, crop_size))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(split: str, detector_model=None, mine_fp: bool = True):
    """
    Build positive and negative crop lists for a dataset split.

    Parameters
    ----------
    split : str
        "train" or "val"
    detector_model : YOLO or None
        If provided, used to mine FP crops from predictions.
    mine_fp : bool
        Whether to include mined FP crops as negatives (train only).

    Returns
    -------
    crops : list of np.ndarray  (H, W, 3) uint8
    labels : list of int  (1 = parenchyma, 0 = negative)
    """
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(DATA_YAML).parent
    img_dir = data_root / data_cfg[split]
    lbl_dir = data_root / data_cfg[split].replace("images", "labels")

    img_paths = sorted(p for p in Path(img_dir).iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    crops, labels = [], []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
        rows = []
        if lbl_path.is_file():
            try:
                rows = np.loadtxt(str(lbl_path), ndmin=2).tolist()
            except Exception:
                rows = []

        parenchyma_boxes_xyxy = []
        all_gt_xyxy = []

        for row in rows:
            cls_id, cx, cy, bw, bh = int(row[0]), row[1], row[2], row[3], row[4]
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            all_gt_xyxy.append([x1, y1, x2, y2])

            c = _crop_box(img_rgb, cx, cy, bw, bh, MARGIN_RATIO, CROP_SIZE)
            if c is None:
                continue

            if cls_id == PARENCHYMA_CLS:
                # Positive
                crops.append(c)
                labels.append(1)
                parenchyma_boxes_xyxy.append([x1, y1, x2, y2])

            elif cls_id == FIBER_CLS:
                # Hard negative: fiber crops look like parenchyma
                crops.append(c)
                labels.append(0)

        # Random background crops (2 per image)
        for _ in range(2):
            bg = _random_bg_crop(img_rgb, all_gt_xyxy, CROP_SIZE)
            if bg is not None:
                crops.append(bg)
                labels.append(0)

        # Mine FP crops from detector predictions (train split only)
        if mine_fp and detector_model is not None:
            try:
                preds = detector_model.predict(
                    str(img_path), conf=0.25, iou=0.45, verbose=False
                )
                if preds and len(preds[0].boxes) > 0:
                    for box in preds[0].boxes:
                        cls_pred = int(box.cls.item())
                        if cls_pred != PARENCHYMA_CLS:
                            continue
                        xyxy = box.xyxy[0].tolist()
                        if _max_iou_vs_gt(xyxy, parenchyma_boxes_xyxy) < FP_IOU_THRESH:
                            # This parenchyma prediction has low overlap with GT — it's a FP
                            x1i, y1i, x2i, y2i = (
                                int(max(0, xyxy[0])), int(max(0, xyxy[1])),
                                int(min(W, xyxy[2])), int(min(H, xyxy[3]))
                            )
                            if x2i > x1i and y2i > y1i:
                                crop = img_rgb[y1i:y2i, x1i:x2i]
                                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                                crops.append(crop)
                                labels.append(0)
            except Exception as e:
                print(f"  [warn] FP mining failed on {img_path.name}: {e}")

    print(
        f"  [{split}] {len(crops)} crops: "
        f"{sum(labels)} pos, {len(labels)-sum(labels)} neg"
    )
    return crops, labels


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CropDataset(Dataset):
    def __init__(self, crops, labels, augment=False):
        self.crops = crops
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        img = self.crops[idx].astype(np.float32) / 255.0
        # HWC -> CHW
        img = torch.tensor(img).permute(2, 0, 1)
        if self.augment:
            # Horizontal / vertical flip
            if random.random() > 0.5:
                img = img.flip(-1)
            if random.random() > 0.5:
                img = img.flip(-2)
            # Colour jitter (brightness ±20%)
            img = (img * (1 + (random.random() - 0.5) * 0.4)).clamp(0, 1)
        lbl = torch.tensor(float(self.labels[idx]))
        return img, lbl


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_classifier(train_crops, train_labels, device=DEVICE, epochs=EPOCHS, lr=LR):
    """Train the ContextRefineClassifier with BCE loss."""
    ds = CropDataset(train_crops, train_labels, augment=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = ContextRefineClassifier(input_size=CROP_SIZE, margin_ratio=MARGIN_RATIO)
    model.to(device)

    # Compute class weights for imbalanced data
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Replace sigmoid classifier with raw logits output for BCEWithLogitsLoss
    # (ContextRefineClassifier already ends with Linear(64, 1) before sigmoid)
    # We bypass the sigmoid during training by patching forward
    original_forward = model.forward

    def logit_forward(crops):
        x = model.features(crops)
        return model.classifier(x)  # raw logits, no sigmoid

    model.forward = logit_forward

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n  Training hard-negative classifier: {epochs} epochs")
    print(f"  Dataset: {len(train_crops)} crops | pos_weight={pos_weight.item():.2f}")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}: loss={total_loss/len(loader):.4f}")

    # Restore sigmoid for inference
    model.forward = original_forward

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(model, val_crops, val_labels, device=DEVICE):
    """Evaluate crop-level accuracy, AUC, and probability distribution."""
    model.eval()
    ds = CropDataset(val_crops, val_labels, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            probs = model(imgs).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.numpy().tolist())

    probs = np.array(all_probs)
    labels = np.array(all_labels, dtype=int)

    # Accuracy at 0.5
    preds = (probs >= 0.5).astype(int)
    accuracy = float((preds == labels).mean())

    # Precision / recall / F1
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # AUC-ROC (manual trapezoid)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        # Fallback: simple rank-based AUC
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        auc = float(np.mean([p > n for p in pos_probs for n in neg_probs])) if (len(pos_probs) and len(neg_probs)) else 0.5

    # Probability distribution buckets (10 bins)
    hist, bin_edges = np.histogram(probs, bins=10, range=(0.0, 1.0))
    prob_hist = {
        f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": int(hist[i])
        for i in range(len(hist))
    }

    # Check for clustering near 0.56 (pathological indicator)
    cluster_05_06 = int(((probs >= 0.50) & (probs <= 0.62)).sum())
    cluster_pct = 100.0 * cluster_05_06 / max(len(probs), 1)

    results = {
        "n_val_crops": len(val_crops),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4),
        "prob_mean_pos": round(float(probs[labels == 1].mean()) if labels.sum() > 0 else 0.0, 4),
        "prob_mean_neg": round(float(probs[labels == 0].mean()) if (1-labels).sum() > 0 else 0.0, 4),
        "prob_std_pos": round(float(probs[labels == 1].std()) if labels.sum() > 0 else 0.0, 4),
        "prob_std_neg": round(float(probs[labels == 0].std()) if (1-labels).sum() > 0 else 0.0, 4),
        "prob_histogram": prob_hist,
        "cluster_near_0.5_0.6_count": cluster_05_06,
        "cluster_near_0.5_0.6_pct": round(cluster_pct, 1),
    }

    print("\n  === Val Evaluation ===")
    print(f"  Crops:     {len(val_crops)}  ({int(labels.sum())} pos, {int((1-labels).sum())} neg)")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Mean prob (pos): {results['prob_mean_pos']:.4f} +/- {results['prob_std_pos']:.4f}")
    print(f"  Mean prob (neg): {results['prob_mean_neg']:.4f} +/- {results['prob_std_neg']:.4f}")
    print(f"  Cluster 0.50-0.62: {cluster_05_06} ({cluster_pct:.1f}%)")
    if cluster_pct > 30:
        print("  WARNING: >30% of scores cluster near 0.5-0.6 -- classifier may not separate classes well!")
    else:
        print("  OK: Probability distribution does not pathologically cluster near 0.56")
    print("  Histogram:")
    for bucket, count in prob_hist.items():
        bar = "#" * min(count, 40)
        print(f"    [{bucket}]: {count:4d}  {bar}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  E6 Context Refiner: Hard-Negative Retraining")
    print("=" * 60)

    # Try to load the existing E6 detector for FP mining
    detector = None
    detector_paths = [
        "runs/detect/results/e6_context_refine/seed_42/weights/best.pt",
        "results/e6_context_refine/seed_42/weights/best.pt",
    ]
    for p in detector_paths:
        if Path(p).exists():
            try:
                from ultralytics import YOLO
                detector = YOLO(p)
                print(f"\nLoaded E6 detector from: {p}")
                break
            except Exception as e:
                print(f"  Could not load detector {p}: {e}")

    if detector is None:
        print("\nWARNING: No E6 detector found -- skipping FP mining.")
        print("  (Train E6 first via: python src/train.py --config configs/e6_context_refine.yaml)")

    # Build datasets
    print("\n--- Building TRAIN dataset ---")
    train_crops, train_labels = build_dataset("train", detector_model=detector, mine_fp=(detector is not None))

    print("\n--- Building VAL dataset (no FP mining) ---")
    val_crops, val_labels = build_dataset("val", detector_model=None, mine_fp=False)

    if len(train_crops) == 0:
        print("\nERROR: No training crops found. Check data paths in data.yaml.")
        return

    # Train
    model = train_classifier(train_crops, train_labels)

    # Save
    out_dir = Path(CLASSIFIER_OUT).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFIER_OUT)
    print(f"\n  Saved classifier to: {CLASSIFIER_OUT}")

    # Evaluate on val
    eval_results = evaluate_classifier(model, val_crops, val_labels)

    # Save results
    full_results = {
        "classifier_path": CLASSIFIER_OUT,
        "config": {
            "crop_size": CROP_SIZE,
            "margin_ratio": MARGIN_RATIO,
            "epochs": EPOCHS,
            "lr": LR,
            "fp_iou_thresh": FP_IOU_THRESH,
            "fp_mining": detector is not None,
        },
        "train": {
            "n_crops": len(train_crops),
            "n_pos": sum(train_labels),
            "n_neg": len(train_labels) - sum(train_labels),
        },
        "val": eval_results,
    }

    with open(RESULTS_OUT, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Full results saved to: {RESULTS_OUT}")

    print("\n=== E6 Hard-Negative Retraining COMPLETE ===")


if __name__ == "__main__":
    main()
