# Emergency Vehicle Ensemble Detection

Phase 1 of the thesis "Adaptive Traffic Signal Management with ML-Driven Fleet Prioritization"
(MSc Computer Science / Algorithms, Yazd University — advisors: Dr. Mohammad Farshi, Dr. Saeed Alikhani; consultant: Dr. Elham Abbasi).

## Contents
- `datasets/` — dataset preparation, merging, and k-fold generation scripts (raw data excluded via .gitignore; see Data section below)
- `datasets/train_yolo_rtdetr.py`, `datasets/train_fasterrcnn.py` — resumable training scripts for YOLOv8n, RT-DETR-l, Faster R-CNN
- `datasets/holdout_eval/` — NOT independent: a near-duplicate audit found that about 95% of its test images also appear in `datasets/folds_v3` (reports in `docs/leakage_audit/`). Use `datasets/holdout_v4` instead (no overlap with `folds_v4` or `folds_v5` in the audit).
- `datasets/confusion_matrix/` — confusion matrix generation for all 3 models

## Data
Datasets are not committed (size + license terms). Sources used:
- KITTI (car class)
- Roboflow: emergency-vehicle-detection-zk8pk, emergency-vehicles-det (Univ. of Bahrain), emergency-vehicles-detection-xockh-af7sr, policeCars_100more, emergency-vehicle-detection-ve1bg (all CC BY 4.0)

## Status
Phase 1 base model training + evaluation: done.
Next: ByteTrack integration, stacking meta-learner, track-guided adaptive cascade.
