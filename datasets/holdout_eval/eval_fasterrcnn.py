from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image

RESULTS = Path.home() / "Portfolio/Smart_Traffic_System/results"
HOLDOUT_IMG = Path.home() / "Portfolio/Smart_Traffic_System/datasets/emergency_merged/test/images"
HOLDOUT_LBL = Path.home() / "Portfolio/Smart_Traffic_System/datasets/emergency_merged/test/labels"
OUT = Path.home() / "Portfolio/Smart_Traffic_System/datasets/holdout_eval"
CLASSES = ["ambulance", "police", "firetruck", "car"]
K = 5
MAX_SIDE = 800

class HoldoutDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.imgs = sorted(self.img_dir.glob("*"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size
        scale = min(1.0, MAX_SIDE / max(w0, h0))
        if scale < 1.0:
            img = img.resize((max(1, int(w0 * scale)), max(1, int(h0 * scale))), Image.BILINEAR)
        w, h = img.size
        boxes, labels = [], []
        if lbl_path.exists():
            for line in open(lbl_path):
                parts = line.split()
                if not parts:
                    continue
                cid = int(parts[0]); cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2]); labels.append(cid + 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }
        return F.to_tensor(img), target

def collate_fn(batch):
    return tuple(zip(*batch))

def build_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = HoldoutDataset(HOLDOUT_IMG, HOLDOUT_LBL)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)

all_results = []
for fold in range(K):
    weights_path = RESULTS / f"fasterrcnn_fold{fold}" / "last.pt"
    if not weights_path.exists():
        continue
    model = build_model().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    metric = MeanAveragePrecision(class_metrics=True)
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            metric.update(preds, targets)
    result = metric.compute()

    per_class = {}
    map_per_class = result.get("map_per_class")
    mar_per_class = result.get("mar_100_per_class")
    for i, cname in enumerate(CLASSES):
        per_class[cname] = {
            "AP": float(map_per_class[i]) if map_per_class is not None and map_per_class.ndim > 0 else None,
            "AR100": float(mar_per_class[i]) if mar_per_class is not None and mar_per_class.ndim > 0 else None,
        }
    fold_result = {"fold": fold, "map": float(result["map"]), "map_50": float(result["map_50"]), "per_class": per_class}
    all_results.append(fold_result)
    print(f"fasterrcnn fold{fold} holdout: mAP={result['map']:.4f} mAP50={result['map_50']:.4f}")
    print(f"  per_class: {per_class}")

with open(OUT / "holdout_results_fasterrcnn.json", "w") as f:
    json.dump(all_results, f, indent=2)
