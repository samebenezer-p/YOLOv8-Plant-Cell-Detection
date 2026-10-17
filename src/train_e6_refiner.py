"""
E6 Context Refiner — Hard-Negative Retraining + Rescoring + Val-Only Threshold Calibration.

Changes vs. previous version
─────────────────────────────
1. Crops for training are mined from detector proposals (matched TP and
   unmatched FP), not just GT boxes, so the classifier sees realistic input.
   TP crops  : detector boxes with IoU >= FP_IOU_THRESH against GT parenchyma
   FP crops  : detector boxes with IoU <  FP_IOU_THRESH against GT parenchyma
   Both are mined from the TRAIN split only.
2. Two post-processing options are evaluated on the VAL split only:
   a. Hard suppression   : remove all parenchyma boxes where p < threshold
   b. Rescoring          : multiply detector confidence by classifier prob
      (boxes are kept; NMS uses the rescored confidence)
3. Threshold calibration: the operating threshold is chosen on the VAL split
   to maximise F1. It is NEVER touched by test-set data.
4. AP is reported with and without refinement for each option.

Run:
    python src/train_e6_refiner.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml

from src.models.context_refine import ContextRefineClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_YAML       = "data.yaml"
CROP_SIZE       = 64
MARGIN_RATIO    = 0.25
BATCH_SIZE      = 32
EPOCHS          = 30
LR              = 1e-3
DEVICE          = "cpu"
CLASSIFIER_OUT  = "results/context_refine/classifier_hardneg.pt"
RESULTS_OUT     = "results/context_refine/hardneg_results.json"
SEED            = 42

PARENCHYMA_CLS  = 2
FIBER_CLS       = 1
FP_IOU_THRESH   = 0.3    # IoU below which a parenchyma prediction is mined as FP
TP_IOU_THRESH   = 0.5    # IoU at or above which a prediction is a matched TP

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iou_xyxy(b1, b2) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _max_iou_vs_gt(det_box, gt_boxes) -> float:
    if not gt_boxes:
        return 0.0
    return max(_iou_xyxy(det_box, g) for g in gt_boxes)


def _crop_box(img, cx, cy, bw, bh, margin_ratio, crop_size):
    H, W = img.shape[:2]
    mx = bw * margin_ratio;  my = bh * margin_ratio
    x1 = int(max(0, (cx - bw / 2 - mx) * W))
    y1 = int(max(0, (cy - bh / 2 - my) * H))
    x2 = int(min(W, (cx + bw / 2 + mx) * W))
    y2 = int(min(H, (cy + bh / 2 + my) * H))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (crop_size, crop_size))


def _crop_xyxy(img, x1, y1, x2, y2, margin_ratio, crop_size):
    H, W = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1c = int(max(0, x1 - bw * margin_ratio))
    y1c = int(max(0, y1 - bh * margin_ratio))
    x2c = int(min(W, x2 + bw * margin_ratio))
    y2c = int(min(H, y2 + bh * margin_ratio))
    if x2c <= x1c or y2c <= y1c:
        return None
    return cv2.resize(img[y1c:y2c, x1c:x2c], (crop_size, crop_size))


def _random_bg_crop(img, gt_boxes_xyxy, crop_size, min_iou=0.05, tries=20):
    H, W = img.shape[:2]
    for _ in range(tries):
        x1 = random.randint(0, max(0, W - crop_size))
        y1 = random.randint(0, max(0, H - crop_size))
        box = [x1, y1, x1 + crop_size, y1 + crop_size]
        if _max_iou_vs_gt(box, gt_boxes_xyxy) < min_iou:
            crop = img[y1:y1+crop_size, x1:x1+crop_size]
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                return cv2.resize(crop, (crop_size, crop_size))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder (detector-proposal crops)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(split: str, detector_model=None, mine_proposals: bool = True):
    """
    Build crop lists for classifier training/eval.

    Positives: GT parenchyma crops + detector-proposal TP crops (IoU >= TP_IOU_THRESH)
    Negatives: GT fiber crops + background patches + detector-proposal FP crops
               (IoU < FP_IOU_THRESH), ALL from detector proposals

    ``mine_proposals`` should be True for TRAIN only.
    """
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(DATA_YAML).parent
    img_dir = data_root / data_cfg[split]
    lbl_dir = data_root / data_cfg[split].replace("images", "labels")

    img_paths = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    crops: List[np.ndarray] = []
    labels: List[int] = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
        rows: List[List[float]] = []
        if lbl_path.is_file():
            try:
                rows = np.loadtxt(str(lbl_path), ndmin=2).tolist()
            except Exception:
                rows = []

        parenchyma_gt_xyxy: List[List[float]] = []
        all_gt_xyxy: List[List[float]] = []

        for row in rows:
            cls_id = int(row[0])
            cx, cy, bw, bh = row[1], row[2], row[3], row[4]
            x1 = (cx - bw / 2) * W;  y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W;  y2 = (cy + bh / 2) * H
            all_gt_xyxy.append([x1, y1, x2, y2])

            c = _crop_box(img_rgb, cx, cy, bw, bh, MARGIN_RATIO, CROP_SIZE)
            if c is None:
                continue

            if cls_id == PARENCHYMA_CLS:
                crops.append(c);  labels.append(1)
                parenchyma_gt_xyxy.append([x1, y1, x2, y2])
            elif cls_id == FIBER_CLS:
                crops.append(c);  labels.append(0)

        # Background patches
        for _ in range(2):
            bg = _random_bg_crop(img_rgb, all_gt_xyxy, CROP_SIZE)
            if bg is not None:
                crops.append(bg);  labels.append(0)

        # Detector-proposal crops (TRAIN only)
        if mine_proposals and detector_model is not None:
            try:
                preds = detector_model.predict(
                    str(img_path), conf=0.10, iou=0.45, verbose=False
                )
                if preds and len(preds[0].boxes) > 0:
                    for box in preds[0].boxes:
                        cls_pred = int(box.cls.item())
                        if cls_pred != PARENCHYMA_CLS:
                            continue
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf.item())
                        max_iou = _max_iou_vs_gt(xyxy, parenchyma_gt_xyxy)
                        crop = _crop_xyxy(img_rgb, *[int(v) for v in xyxy],
                                          MARGIN_RATIO, CROP_SIZE)
                        if crop is None:
                            continue
                        if max_iou >= TP_IOU_THRESH:
                            # Detector TP proposal → positive
                            crops.append(crop);  labels.append(1)
                        elif max_iou < FP_IOU_THRESH:
                            # Hard FP negative
                            crops.append(crop);  labels.append(0)
            except Exception as e:
                print(f"  [warn] proposal mining failed on {img_path.name}: {e}")

    print(
        f"  [{split}] {len(crops)} crops: "
        f"{sum(labels)} pos, {len(labels)-sum(labels)} neg"
    )
    return crops, labels


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class CropDataset(Dataset):
    def __init__(self, crops, labels, augment=False):
        self.crops   = crops
        self.labels  = labels
        self.augment = augment

    def __len__(self):  return len(self.crops)

    def __getitem__(self, idx):
        img = self.crops[idx].astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)
        if self.augment:
            if random.random() > 0.5:  img = img.flip(-1)
            if random.random() > 0.5:  img = img.flip(-2)
            img = (img * (1 + (random.random() - 0.5) * 0.4)).clamp(0, 1)
        return img, torch.tensor(float(self.labels[idx]))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(train_crops, train_labels, device=DEVICE, epochs=EPOCHS, lr=LR):
    ds = CropDataset(train_crops, train_labels, augment=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = ContextRefineClassifier(input_size=CROP_SIZE, margin_ratio=MARGIN_RATIO)
    model.to(device)

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def logit_forward(crops):
        x = model.features(crops)
        return model.classifier(x)          # raw logits

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n  Training hard-negative classifier ({epochs} epochs)")
    print(f"  Dataset: {len(train_crops)} crops | pos_weight={pos_weight.item():.2f}")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = logit_forward(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}: loss={total_loss/len(loader):.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Classifier evaluation (crop-level)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(model, val_crops, val_labels, device=DEVICE):
    model.eval()
    ds     = CropDataset(val_crops, val_labels, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_probs: List[float] = []
    all_lbls:  List[int]   = []
    with torch.no_grad():
        for imgs, lbls in loader:
            probs = model(imgs.to(device)).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_lbls.extend(lbls.numpy().tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_lbls, dtype=int)

    preds    = (probs >= 0.5).astype(int)
    accuracy = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        pos_p = probs[labels == 1];  neg_p = probs[labels == 0]
        auc = float(np.mean([p > n for p in pos_p for n in neg_p])) \
              if (len(pos_p) and len(neg_p)) else 0.5

    hist, edges = np.histogram(probs, bins=10, range=(0.0, 1.0))
    prob_hist = {f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(hist[i]) for i in range(len(hist))}
    cluster_05_06 = int(((probs >= 0.50) & (probs <= 0.62)).sum())
    cluster_pct   = 100.0 * cluster_05_06 / max(len(probs), 1)

    results = {
        "n_val_crops": len(val_crops),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "accuracy":   round(accuracy, 4),
        "precision":  round(prec, 4),
        "recall":     round(rec, 4),
        "f1":         round(f1, 4),
        "auc_roc":    round(auc, 4),
        "prob_mean_pos": round(float(probs[labels == 1].mean()) if labels.sum() > 0 else 0.0, 4),
        "prob_mean_neg": round(float(probs[labels == 0].mean()) if (1-labels).sum() > 0 else 0.0, 4),
        "prob_std_pos":  round(float(probs[labels == 1].std())  if labels.sum() > 0 else 0.0, 4),
        "prob_std_neg":  round(float(probs[labels == 0].std())  if (1-labels).sum() > 0 else 0.0, 4),
        "prob_histogram": prob_hist,
        "cluster_near_0.5_0.6_count": cluster_05_06,
        "cluster_near_0.5_0.6_pct":   round(cluster_pct, 1),
    }

    print("\n  === Crop-level Val Evaluation ===")
    print(f"  Crops: {len(val_crops)} ({int(labels.sum())} pos, {int((1-labels).sum())} neg)")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Mean prob — pos: {results['prob_mean_pos']:.4f} ± {results['prob_std_pos']:.4f}")
    print(f"  Mean prob — neg: {results['prob_mean_neg']:.4f} ± {results['prob_std_neg']:.4f}")
    cluster_msg = "WARNING: >30% cluster near 0.5-0.6" if cluster_pct > 30 else "OK"
    print(f"  Cluster 0.50-0.62: {cluster_05_06} ({cluster_pct:.1f}%)  {cluster_msg}")
    for bucket, count in prob_hist.items():
        print(f"    [{bucket}]: {count:4d}  {'#' * min(count, 40)}")

    return results, probs, labels


# ─────────────────────────────────────────────────────────────────────────────
# Val-only threshold calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_threshold_on_val(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, Dict]:
    """
    Sweep thresholds on VAL split and return the one maximising F1.
    NEVER called on the test split.

    Returns
    -------
    best_thresh : float
    calibration_results : dict
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.9, 0.05)]

    best_thresh, best_f1 = 0.5, -1.0
    sweep: List[Dict] = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        sweep.append({"threshold": round(t, 2), "precision": round(prec, 4),
                      "recall": round(rec, 4), "f1": round(f1, 4)})
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\n  === Val-only Threshold Calibration ===")
    print(f"  Best threshold: {best_thresh:.2f}  (F1={best_f1:.4f})")
    for row in sweep:
        marker = " ◀ BEST" if row["threshold"] == best_thresh else ""
        print(f"    t={row['threshold']:.2f}  P={row['precision']:.4f}  "
              f"R={row['recall']:.4f}  F1={row['f1']:.4f}{marker}")

    return best_thresh, {"best_threshold": best_thresh, "best_f1": round(best_f1, 4),
                          "sweep": sweep}


# ─────────────────────────────────────────────────────────────────────────────
# Detector-level AP comparison (hard suppression vs rescoring)
# ─────────────────────────────────────────────────────────────────────────────

def compare_refinement_ap(
    detector_path: str,
    classifier: ContextRefineClassifier,
    best_threshold: float,
    split: str = "val",
) -> Dict:
    """
    Evaluate parenchyma AP50/AP50-95 for three modes:
      (1) No refinement  — raw detector output
      (2) Hard suppression — remove parenchyma boxes where clf_prob < best_threshold
      (3) Rescoring       — multiply detector confidence by clf_prob; keep all boxes

    Operates on the VAL split only (split="val").
    """
    from ultralytics import YOLO
    import yaml as _yaml

    with open(DATA_YAML) as f:
        data_cfg = _yaml.safe_load(f)
    data_root = Path(DATA_YAML).parent
    img_dir   = data_root / data_cfg[split]

    detector = YOLO(detector_path)
    img_paths = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    lbl_dir = data_root / data_cfg[split].replace("images", "labels")

    # Collect detections and ground-truth per image
    records: List[Dict] = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
        gt_rows: List[List[float]] = []
        if lbl_path.is_file():
            try:
                gt_rows = np.loadtxt(str(lbl_path), ndmin=2).tolist()
            except Exception:
                gt_rows = []

        preds = detector.predict(str(img_path), conf=0.10, iou=0.45, verbose=False)
        if not preds or len(preds[0].boxes) == 0:
            det_boxes = []
        else:
            det_boxes = [
                {
                    "xyxy":      box.xyxy[0].tolist(),
                    "conf":      float(box.conf.item()),
                    "cls":       int(box.cls.item()),
                }
                for box in preds[0].boxes
            ]

        # Score parenchyma boxes with classifier
        H, W = img_rgb.shape[:2]
        for det in det_boxes:
            if det["cls"] == PARENCHYMA_CLS:
                crop = _crop_xyxy(img_rgb, *[int(v) for v in det["xyxy"]],
                                  MARGIN_RATIO, CROP_SIZE)
                if crop is not None:
                    t = torch.tensor(crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        det["clf_prob"] = float(classifier(t).item())
                else:
                    det["clf_prob"] = 1.0   # can't crop → keep
            else:
                det["clf_prob"] = 1.0

        records.append({"gt": gt_rows, "dets": det_boxes, "img": str(img_path), "hw": (H, W)})

    # Compute AP for each mode
    modes = {
        "no_refinement":    lambda d: (d["conf"],              True),
        "hard_suppression": lambda d: (d["conf"],              d["cls"] != PARENCHYMA_CLS
                                       or d["clf_prob"] >= best_threshold),
        "rescoring":        lambda d: (d["conf"] * d["clf_prob"], True),
    }

    ap_results: Dict[str, Dict] = {}
    for mode_name, fn in modes.items():
        ap50, ap50_95 = _compute_parenchyma_ap(records, fn, PARENCHYMA_CLS)
        # Count suppression stats for hard_suppression mode
        if mode_name == "hard_suppression":
            suppressed = sum(
                1 for rec in records for d in rec["dets"]
                if d["cls"] == PARENCHYMA_CLS and d["clf_prob"] < best_threshold
            )
            # TP: suppressed boxes that had high IoU with a GT parenchyma
            supp_tp = sum(
                1 for rec in records for d in rec["dets"]
                if d["cls"] == PARENCHYMA_CLS and d["clf_prob"] < best_threshold
                and _max_iou_vs_gt(
                    d["xyxy"],
                    [[(r[1]-(r[3]/2))*rec["hw"][1], (r[2]-(r[4]/2))*rec["hw"][0],
                      (r[1]+(r[3]/2))*rec["hw"][1], (r[2]+(r[4]/2))*rec["hw"][0]]
                     for r in rec["gt"] if int(r[0]) == PARENCHYMA_CLS]
                ) >= TP_IOU_THRESH
            )
            ap_results[mode_name] = {
                "parenchyma_AP50":    round(ap50, 4),
                "parenchyma_AP50_95": round(ap50_95, 4),
                "suppressed_boxes":   suppressed,
                "suppressed_TP":      supp_tp,
                "suppressed_FP":      suppressed - supp_tp,
                "threshold":          round(best_threshold, 4),
            }
        else:
            ap_results[mode_name] = {
                "parenchyma_AP50":    round(ap50, 4),
                "parenchyma_AP50_95": round(ap50_95, 4),
            }

    print("\n  === Detector-level Refinement AP Comparison (VAL) ===")
    print(f"  {'Mode':<22}  AP50    AP50-95")
    for mode, res in ap_results.items():
        print(f"  {mode:<22}  {res['parenchyma_AP50']:.4f}  {res['parenchyma_AP50_95']:.4f}")
        if "suppressed_boxes" in res:
            print(f"    Suppressed: {res['suppressed_boxes']} boxes  "
                  f"(TP={res['suppressed_TP']}, FP={res['suppressed_FP']})  "
                  f"@ threshold={res['threshold']:.2f}")

    return ap_results


def _compute_parenchyma_ap(
    records: List[Dict],
    box_filter_fn,
    target_cls: int,
    iou_thresh: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute AP50 and AP50-95 for a single class using the VOC 11-point method.
    ``box_filter_fn(det) -> (score, keep_bool)``
    """
    thresholds_95 = [0.5 + 0.05 * i for i in range(10)]

    def _ap_at_iou(iou_t: float) -> float:
        all_tp: List[Tuple[float, int]] = []   # (score, is_tp)
        n_gt = 0
        for rec in records:
            H, W = rec["hw"]
            gt_para = [
                [(r[1]-(r[3]/2))*W, (r[2]-(r[4]/2))*H,
                 (r[1]+(r[3]/2))*W, (r[2]+(r[4]/2))*H]
                for r in rec["gt"] if int(r[0]) == target_cls
            ]
            n_gt += len(gt_para)
            matched = [False] * len(gt_para)
            # Filter + score dets
            scored = []
            for d in rec["dets"]:
                if d["cls"] != target_cls:
                    continue
                score, keep = box_filter_fn(d)
                if keep:
                    scored.append((score, d["xyxy"]))
            # Sort by score descending
            scored.sort(key=lambda x: -x[0])
            for score, xyxy in scored:
                if not gt_para:
                    all_tp.append((score, 0))
                    continue
                ious = [_iou_xyxy(xyxy, g) for g in gt_para]
                best_idx = int(np.argmax(ious))
                if ious[best_idx] >= iou_t and not matched[best_idx]:
                    matched[best_idx] = True
                    all_tp.append((score, 1))
                else:
                    all_tp.append((score, 0))

        if n_gt == 0:
            return 0.0
        all_tp.sort(key=lambda x: -x[0])
        tp_arr = np.array([t for _, t in all_tp])
        fp_arr = 1 - tp_arr
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(fp_arr)
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)
        recall    = cum_tp / n_gt
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = precision[recall >= t].max() if (recall >= t).any() else 0.0
            ap += p / 11.0
        return ap

    ap50    = _ap_at_iou(0.5)
    ap50_95 = float(np.mean([_ap_at_iou(t) for t in thresholds_95]))
    return ap50, ap50_95


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  E6 Context Refiner — Hard-Negative Retraining + Rescoring")
    print("=" * 65)

    # Load E6 detector for proposal mining
    detector = None
    detector_path = None
    detector_candidates = [
        "runs/detect/results/e6_context_refine/seed_42/weights/best.pt",
        "runs/detect/results/e1_baseline/seed_42/weights/best.pt",  # fallback
    ]
    for p in detector_candidates:
        if Path(p).exists():
            try:
                from ultralytics import YOLO as _YOLO
                detector = _YOLO(p)
                detector_path = p
                print(f"\nLoaded detector from: {p}")
                break
            except Exception as e:
                print(f"  Could not load {p}: {e}")

    if detector is None:
        print("\nWARNING: No detector found — skipping proposal mining.")

    # Build datasets
    print("\n--- Building TRAIN dataset (with proposal mining) ---")
    train_crops, train_labels = build_dataset(
        "train", detector_model=detector, mine_proposals=(detector is not None)
    )
    print("\n--- Building VAL dataset (GT + fibers only, no mining) ---")
    val_crops, val_labels = build_dataset("val", detector_model=None, mine_proposals=False)

    if len(train_crops) == 0:
        print("\nERROR: No training crops found. Check data paths in data.yaml.")
        return

    # Train
    classifier = train_classifier(train_crops, train_labels)

    # Save
    out_dir = Path(CLASSIFIER_OUT).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), CLASSIFIER_OUT)
    print(f"\n  Saved classifier → {CLASSIFIER_OUT}")

    # Crop-level evaluation on val
    crop_results, val_probs, val_lbls = evaluate_classifier(
        classifier, val_crops, val_labels
    )

    # Val-only threshold calibration
    best_thresh, calib_results = calibrate_threshold_on_val(val_probs, val_lbls)

    # Detector-level AP comparison (val split)
    ap_comparison: Dict = {}
    if detector_path is not None:
        ap_comparison = compare_refinement_ap(
            detector_path, classifier, best_thresh, split="val"
        )
    else:
        print("\n  Skipping detector-level AP comparison (no detector checkpoint).")

    # Save full results
    full_results = {
        "classifier_path":   CLASSIFIER_OUT,
        "best_threshold_val": round(best_thresh, 4),
        "config": {
            "crop_size":     CROP_SIZE,
            "margin_ratio":  MARGIN_RATIO,
            "epochs":        EPOCHS,
            "lr":            LR,
            "fp_iou_thresh": FP_IOU_THRESH,
            "tp_iou_thresh": TP_IOU_THRESH,
            "proposal_mining": detector is not None,
        },
        "train": {
            "n_crops": len(train_crops),
            "n_pos":   sum(train_labels),
            "n_neg":   len(train_labels) - sum(train_labels),
        },
        "val_crop_eval":     crop_results,
        "val_threshold_cal": calib_results,
        "val_ap_comparison": ap_comparison,
    }

    with open(RESULTS_OUT, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Full results saved → {RESULTS_OUT}")
    print("\n=== E6 Hard-Negative Retraining + Rescoring COMPLETE ===")


if __name__ == "__main__":
    main()
