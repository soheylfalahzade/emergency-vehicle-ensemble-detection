import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO, RTDETR
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

RESULTS = Path.home() / "Portfolio/Smart_Traffic_System/results"
HOLDOUT_IMG = Path.home() / "Portfolio/Smart_Traffic_System/datasets/emergency_merged/test/images"
HOLDOUT_LBL = Path.home() / "Portfolio/Smart_Traffic_System/datasets/emergency_merged/test/labels"
OUT = Path.home() / "Portfolio/Smart_Traffic_System/datasets/confusion_matrix"
CLASSES = ["ambulance", "police", "firetruck", "car"]
NC = len(CLASSES)
IOU_THRESH = 0.5
CONF_THRESH = 0.25
MAX_SIDE = 800
FOLD_FOR_CM = 0  # use fold 0's model as the representative model for each architecture

def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def load_gt(lbl_path, w, h):
    boxes, labels = [], []
    if lbl_path.exists():
        for line in open(lbl_path):
            parts = line.split()
            if not parts:
                continue
            cid = int(parts[0]); cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx-bw/2)*w; y1 = (cy-bh/2)*h; x2 = (cx+bw/2)*w; y2 = (cy+bh/2)*h
            boxes.append([x1,y1,x2,y2]); labels.append(cid)
    return boxes, labels

def build_cm(preds_per_image, gts_per_image):
    # cm[gt_class][pred_class], with an extra "background" row/col (index NC) for FP/FN
    cm = np.zeros((NC+1, NC+1), dtype=int)
    for preds, (gt_boxes, gt_labels) in zip(preds_per_image, gts_per_image):
        matched_gt = set()
        for pbox, pcls, pconf in preds:
            if pconf < CONF_THRESH:
                continue
            best_iou, best_j = 0, -1
            for j, gbox in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                v = iou(pbox, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= IOU_THRESH:
                matched_gt.add(best_j)
                cm[gt_labels[best_j]][pcls] += 1
            else:
                cm[NC][pcls] += 1  # false positive (background predicted as object)
        for j, gl in enumerate(gt_labels):
            if j not in matched_gt:
                cm[gl][NC] += 1  # false negative (missed detection)
    return cm

def plot_cm(cm, title, path):
    labels = CLASSES + ["background"]
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def get_gts(img_paths):
    gts = []
    for p in img_paths:
        with Image.open(p) as im:
            w, h = im.size
        gts.append(load_gt(HOLDOUT_LBL / (p.stem + ".txt"), w, h))
    return gts

img_paths = sorted(HOLDOUT_IMG.glob("*"))
gts = get_gts(img_paths)

results_summary = {}

# ---- YOLOv8n ----
model = YOLO(str(RESULTS / f"yolov8n_fold{FOLD_FOR_CM}" / "weights" / "best.pt"))
preds_all = []
for p in img_paths:
    r = model.predict(str(p), verbose=False)[0]
    preds = [(box.tolist(), int(cls), float(conf)) for box, cls, conf in
             zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    preds_all.append(preds)
cm = build_cm(preds_all, gts)
plot_cm(cm, "YOLOv8n Confusion Matrix (holdout)", OUT / "cm_yolov8n.png")
results_summary["yolov8n"] = cm.tolist()
print("YOLOv8n done")

# ---- RT-DETR-l ----
model = RTDETR(str(RESULTS / f"rtdetr_l_fold{FOLD_FOR_CM}" / "weights" / "best.pt"))
preds_all = []
for p in img_paths:
    r = model.predict(str(p), verbose=False)[0]
    preds = [(box.tolist(), int(cls), float(conf)) for box, cls, conf in
             zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    preds_all.append(preds)
cm = build_cm(preds_all, gts)
plot_cm(cm, "RT-DETR-l Confusion Matrix (holdout)", OUT / "cm_rtdetr_l.png")
results_summary["rtdetr_l"] = cm.tolist()
print("RT-DETR-l done")

# ---- Faster R-CNN ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frcnn = fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, NC + 1)
frcnn.load_state_dict(torch.load(RESULTS / f"fasterrcnn_fold{FOLD_FOR_CM}" / "last.pt", map_location=device))
frcnn.to(device).eval()

preds_all = []
with torch.no_grad():
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        w0, h0 = img.size
        scale = min(1.0, MAX_SIDE / max(w0, h0))
        img_r = img.resize((max(1,int(w0*scale)), max(1,int(h0*scale))), Image.BILINEAR) if scale < 1.0 else img
        tensor = F.to_tensor(img_r).to(device)
        out = frcnn([tensor])[0]
        preds = []
        for box, label, score in zip(out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy()):
            box_orig = [c / scale for c in box]  # rescale back to original image coords
            preds.append((box_orig, int(label) - 1, float(score)))  # -1: undo the +1 offset used in training
        preds_all.append(preds)
cm = build_cm(preds_all, gts)
plot_cm(cm, "Faster R-CNN Confusion Matrix (holdout)", OUT / "cm_fasterrcnn.png")
results_summary["fasterrcnn"] = cm.tolist()
print("Faster R-CNN done")

with open(OUT / "confusion_matrices.json", "w") as f:
    json.dump({"classes": CLASSES + ["background"], "matrices": results_summary}, f, indent=2)
