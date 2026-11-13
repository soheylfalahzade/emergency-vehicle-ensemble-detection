# Emergency vehicle detection - Phase 1 base training (superseded)

Training and evaluation scripts for a four-class detector (ambulance, police, firetruck, car)
with YOLOv8n, RT-DETR-l and Faster R-CNN, trained with 5-fold cross-validation.

This repository is superseded by https://github.com/soheylfalahzade/emergency-cascade-detection-v3,
which contains the current pipeline, frozen evaluation protocol and results.

## Contents
- `datasets/` - dataset preparation, merging and k-fold generation scripts (raw data excluded via .gitignore)
- `datasets/train_yolo_rtdetr.py`, `datasets/train_fasterrcnn.py` - resumable training scripts
- `datasets/holdout_eval/` - NOT independent: a near-duplicate audit found that about 95% of its test images also appear in `datasets/folds_v3` (reports in `docs/leakage_audit/`). Do not use it for reported results.
- `datasets/confusion_matrix/` - confusion matrix generation for all 3 models

## Data
Datasets are not committed (size and license terms). Sources used:
- KITTI (car class)
- Roboflow: emergency-vehicle-detection-zk8pk, emergency-vehicles-det, emergency-vehicles-detection-xockh-af7sr, policeCars_100more, emergency-vehicle-detection-ve1bg (all CC BY 4.0)
