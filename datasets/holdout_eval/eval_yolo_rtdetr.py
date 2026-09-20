from pathlib import Path
from ultralytics import YOLO, RTDETR
import json

RESULTS = Path.home() / "Portfolio/Smart_Traffic_System/results"
DATA = Path.home() / "Portfolio/Smart_Traffic_System/datasets/holdout_eval/holdout_data.yaml"
OUT = Path.home() / "Portfolio/Smart_Traffic_System/datasets/holdout_eval"
K = 5

all_results = {}

for model_name, ctor in [("yolov8n", YOLO), ("rtdetr_l", RTDETR)]:
    all_results[model_name] = []
    for fold in range(K):
        weights = RESULTS / f"{model_name}_fold{fold}" / "weights" / "best.pt"
        if not weights.exists():
            continue
        model = ctor(str(weights))
        metrics = model.val(data=str(DATA), split="val", project=str(OUT), name=f"{model_name}_fold{fold}_holdout", exist_ok=True)
        per_class = {}
        names = metrics.names
        for i, cname in names.items():
            per_class[cname] = {
                "AP50": float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else None,
                "AP50-95": float(metrics.box.ap[i]) if i < len(metrics.box.ap) else None,
                "precision": float(metrics.box.p[i]) if i < len(metrics.box.p) else None,
                "recall": float(metrics.box.r[i]) if i < len(metrics.box.r) else None,
            }
        all_results[model_name].append({"fold": fold, "per_class": per_class, "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)})
        print(f"{model_name} fold{fold} holdout: mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")

with open(OUT / "holdout_results_yolo_rtdetr.json", "w") as f:
    json.dump(all_results, f, indent=2)
