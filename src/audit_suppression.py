import cv2
import json
import numpy as np
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from src.models.context_refine import ContextRefineClassifier

def box_iou_matrix(b1, b2):
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.maximum(0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-6)

clf = ContextRefineClassifier(input_size=64, margin_ratio=0.25)
clf.load_state_dict(torch.load('results/context_refine/classifier_hardneg.pt', map_location='cpu'))
clf.eval()

detector = YOLO('runs/detect/results/e1_baseline/seed_42/weights/best.pt')
with open('data.yaml') as f:
    data_cfg = yaml.safe_load(f)
test_imgs = sorted(Path(data_cfg['test']).glob('*.*'))

suppressed_records = []

for img_p in test_imgs:
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]
    lbl_p = Path('labels/test') / (img_p.stem + '.txt')
    gt_b = []
    if lbl_p.exists():
        for line in lbl_p.read_text().splitlines():
            p = line.strip().split()
            if not p: continue
            c, cx, cy, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
            if c == 2:
                gt_b.append([(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h])
    gt_b = np.array(gt_b, dtype=np.float32) if gt_b else np.zeros((0, 4), dtype=np.float32)
    
    preds = detector(img, imgsz=640, conf=0.25, iou=0.45, max_det=300, verbose=False)[0]
    boxes = preds.boxes.xyxy.cpu().numpy()
    clss = preds.boxes.cls.cpu().numpy().astype(int)
    confs = preds.boxes.conf.cpu().numpy()
    
    p_mask = (clss == 2)
    p_b, p_s = boxes[p_mask], confs[p_mask]
    if len(p_b) == 0:
        continue
        
    order = np.argsort(-p_s)
    
    # Ultralytics standard matching (greedy in descending confidence order)
    matched_gt_50 = set()
    is_primary_tp = {}
    ious = box_iou_matrix(p_b, gt_b) if len(gt_b) > 0 else np.zeros((len(p_b), 0))
    for pi in order:
        best_iou, best_gi = -1.0, -1
        for gi in range(len(gt_b)):
            if gi in matched_gt_50: continue
            if ious[pi, gi] >= 0.5 and ious[pi, gi] > best_iou:
                best_iou = ious[pi, gi]
                best_gi = gi
        if best_gi >= 0:
            is_primary_tp[pi] = (True, best_gi, best_iou)
            matched_gt_50.add(best_gi)
        else:
            max_iou = float(np.max(ious[pi])) if len(gt_b) > 0 else 0.0
            is_primary_tp[pi] = (False, -1, max_iou)
            
    for pi, (b, s) in enumerate(zip(p_b, p_s)):
        x1, y1, x2, y2 = b
        bw, bh = x2 - x1, y2 - y1
        mw, mh = bw * 0.25, bh * 0.25
        cx1, cy1 = int(max(0, x1 - mw)), int(max(0, y1 - mh))
        cx2, cy2 = int(min(w, x2 + mw)), int(min(h, y2 + mh))
        crop = cv2.resize(img[cy1:cy2, cx1:cx2], (64, 64))
        crop_t = torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            prob = clf(crop_t).item()
        if prob < 0.5:
            was_matched, gt_idx, gt_iou = is_primary_tp[pi]
            suppressed_records.append({
                'image': img_p.name,
                'box': [round(float(v), 1) for v in b.tolist()],
                'conf': round(float(s), 4),
                'prob': round(float(prob), 4),
                'gt_iou': round(float(gt_iou), 4),
                'is_primary_tp_at_0_5': was_matched,
                'gt_idx': int(gt_idx),
            })

print(f'Total suppressed records: {len(suppressed_records)}')
primary_tps = sum(1 for r in suppressed_records if r['is_primary_tp_at_0_5'])
print(f'Suppressed boxes that were primary TP at IoU=0.5: {primary_tps}')
Path('results/context_refine/suppressed_43_boxes.json').write_text(json.dumps(suppressed_records, indent=2))
for i, r in enumerate(suppressed_records):
    print(f"{i+1:2d}. img={r['image']} conf={r['conf']:.4f} p={r['prob']:.4f} iou={r['gt_iou']:.4f} primary_TP={r['is_primary_tp_at_0_5']}")
