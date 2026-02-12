# Deep_Learning_Group 3

## 1. Project Goal
This repository implements a two-stage pipeline for Helicobacter pathology analysis:

1. Patch-level Presence classification (`Presence in {-1, 1}`)
2. Patient-level 3-class classification (`DENSITAT in {NEGATIVA, BAIXA, ALTA}`)

Core constraints used in this project:

- Dataset is read-only under `/hhome/ricse03/HelicoData`
- Mapping rules:
  - `Pat_ID = patient_folder.split('_')[0]`
  - `Window_ID` is canonicalized from image stem / xlsx value
- HoldOut is final evaluation only (not used for training/tuning)
- Splits are grouped by patient to avoid leakage

---

## 2. Data Inputs (Absolute)

- `DATA_ROOT=/hhome/ricse03/HelicoData`
- `XLSX=/hhome/ricse03/HelicoData/HP_WSI-CoordAnnotatedAllPatches.xlsx`
- `DIAG=/hhome/ricse03/HelicoData/PatientDiagnosis.csv`
- Patch images used for training/validation/testing: `CrossValidation/Annotated`
- Final inference set: `HoldOut`

---

## 3. Pipeline Overview

1. Build patient labels
   - `src/data/build_patient_labels.py`
   - Output: `manifests/patient_labels.csv`

2. Build clean patch table + patient-grouped split CSVs
   - `src/data/rebuild_clean_patch_splits.py`
   - Outputs:
     - `manifests/patch_train.csv`
     - `manifests/patch_val.csv`
     - `manifests/patch_test.csv`

3. Train patch model (Task1)
   - `src/train/train_patch_presence.py`
   - Best retained checkpoint in this repo snapshot:
     - `outputs/patch_matrix/resnet18_s42/run_001/best.ckpt`

4. Build patient embedding features from patch checkpoint
   - `src/infer/build_patient_embedding_features.py`
   - Outputs under:
     - `outputs/patient_features_full/resnet18_s42/`

5. Train patient classifier (Task2)
   - `src/train/train_patient_classifier_embed_cv.py`
   - Best retained run in this snapshot:
     - `outputs/patient_full/resnet18_s42/run_001/`

6. Patch interpretability (Grad-CAM)
   - `src/infer/generate_patch_gradcam_heatmaps.py`
   - Output:
     - `outputs/patch_heatmaps/run_001/`

7. Patient prediction report export (`xlsx`)
   - `src/report/export_patient_prediction_report.py`
   - Output:
     - `outputs/patient_full/resnet18_s42/run_001/patient_prediction_report.xlsx`

---

## 4. Current Best Result Snapshot

Best patient run: `outputs/patient_full/resnet18_s42/run_001/metrics.json`

- Accuracy: `0.7758620689655172`
- Macro-F1: `0.7492134459347574`
- Balanced Accuracy: `0.7367872436330855`
- Macro OvR ROC-AUC: `0.7977895349147603`
- Macro OvR PR-AUC: `0.670905397446111`
- BAIXA:
  - Precision: `0.625`
  - Recall: `0.5882352941176471`
  - F1: `0.6060606060606061`
- Confusion Matrix:
  - `[[53, 5, 0], [11, 20, 3], [0, 7, 17]]`

Global summary files:

- `outputs/patient_full/summary_metrics.csv`
- `outputs/patient_full/summary_top.json`

---

## 5. File-by-File Guide

### 5.1 Root Files

- `AGENTS.md`
  - Repository operation constraints, data rules, and safety requirements.
- `README.md`
  - This documentation.

### 5.2 Config Files

- `configs/patch.yaml`
  - Base patch training config.
- `configs/patch_tuned.yaml`
  - Tuned patch training config.
- `configs/patch_tuned_quick.yaml`
  - Faster patch tuning config.
- `configs/patch_upstream.yaml`
  - Upstream patch config variant.
- `configs/patch_upstream_fast.yaml`
  - Fast upstream variant.
- `configs/patch_upstream_stable.yaml`
  - Stable upstream variant.
- `configs/patient_mil.yaml`
  - Patient MIL training config.
- `configs/patch_matrix/*.yaml`
  - Patch model matrix experiment configs:
    - `resnet18_s42.yaml`
    - `resnet18_s52.yaml`
    - `resnet34_s42.yaml`
    - `resnet34_s52.yaml`
    - `resnet50_s42.yaml`
    - `resnet50_s52.yaml`

### 5.3 Data Build Scripts (`src/data`)

- `src/data/build_patch_index.py`
  - Build patch index from annotated images + xlsx labels with strict join/audit logic.
- `src/data/build_patient_labels.py`
  - Build `Pat_ID, DENSITAT` table from diagnosis CSV.
- `src/data/sanity_manifest.py`
  - Validate manifests and print counts/balance/null/dup checks.
- `src/data/make_patch_splits.py`
  - Build patient-grouped split ID files for patch task.
- `src/data/rebuild_clean_patch_splits.py`
  - Rebuild clean patch split CSVs (canonical Window_ID handling + stratified patient split).
- `src/data/build_patient_features.py`
  - Aggregate patch probabilities into patient-level statistical features.

### 5.4 Model Definitions (`src/models`)

- `src/models/resnet_presence.py`
  - ResNet18/34/50 binary Presence model construction.

### 5.5 Inference / Feature Scripts (`src/infer`)

- `src/infer/infer_patch_presence.py`
  - Patch-level inference for HoldOut and optional annotated splits.
- `src/infer/build_patient_embedding_features.py`
  - Extract patient-level embedding + probability features from patch checkpoint.
- `src/infer/generate_patch_gradcam_heatmaps.py`
  - Generate patch-level Grad-CAM heatmap visualizations and index CSV.

### 5.6 Training Scripts (`src/train`)

- `src/train/train_patch_presence.py`
  - Train patch Presence classifier, save checkpoint and patch metrics.
- `src/train/train_patient_classifier_embed_cv.py`
  - Main patient 3-class classifier with CV model selection and rich metrics export.
- `src/train/train_patient_classifier.py`
  - Baseline patient classifier script.
- `src/train/train_patient_classifier_cv.py`
  - Patient classifier CV variant.
- `src/train/train_patient_classifier_rf_margin.py`
  - RF + BAIXA margin rule variant.
- `src/train/train_patient_mil.py`
  - Patient MIL training/inference pipeline.

### 5.7 Reporting (`src/report`)

- `src/report/export_patient_prediction_report.py`
  - Export patient prediction `xlsx` report with truth/pred/correctness/confidence/errors.
- `src/report/build_paper_artifacts.py`
  - Build paper-ready artifacts:
    - main result table
    - per-class patient table
    - top error case table
    - figure inventory
    - consolidated xlsx + markdown summary
  - Optionally copy these artifacts into `/hhome/ricse03/reslut/paper`.

### 5.8 Manifest / Split Outputs

- `manifests/patient_labels.csv`
  - Ground-truth patient labels (`Pat_ID`, `DENSITAT`).
- `manifests/patch_train.csv`
- `manifests/patch_val.csv`
- `manifests/patch_test.csv`
  - Clean patch rows with image path, patient id, canonical window id, presence label.

### 5.9 Retained Result Outputs (Current Snapshot)

- `outputs/patch_matrix_top3.tsv`
  - Top3 patch checkpoints from matrix search.
- `outputs/patch_matrix/resnet18_s42/run_001/`
  - Best retained patch model artifacts:
    - `best.ckpt`
    - `metrics.json`
    - `confusion_matrix.png`
    - `roc_curve.png`
    - `pr_curve.png`

- `outputs/patient_features_full/resnet18_s42/`
  - Patient feature tables for each split:
    - `train_features.csv`
    - `val_features.csv`
    - `test_features.csv`
    - `holdout_features.csv`

- `outputs/patient_full/resnet18_s42/run_001/`
  - Best retained patient model artifacts:
    - `metrics.json`
    - `model.pkl`
    - `preds_holdout.csv`
    - `confusion_matrix.png`
    - `patient_prediction_report.xlsx`

- `outputs/patient_full/summary_metrics.csv`
  - Combined metrics across available runs.
- `outputs/patient_full/summary_top.json`
  - Best run snapshots by different criteria.

- `outputs/patch_heatmaps/run_001/`
  - Patch Grad-CAM outputs:
    - `presence_pos/*.png`
    - `presence_neg/*.png`
    - `heatmaps_index.csv`
    - `summary.json`

- `outputs/paper/run_001/`
  - Paper-ready reporting package:
    - `paper_main_results.csv`
    - `paper_patient_per_class.csv`
    - `paper_error_cases_top.csv`
    - `paper_figure_inventory.csv`
    - `paper_tables.xlsx`
    - `paper_summary.md`
    - `paper_artifacts_manifest.json`

---

## 6. Quick Reproduce Commands

Run from `/hhome/ricse03/Deep_Learning_Group 3`.

1. Rebuild clean patch splits:

```bash
python3 src/data/rebuild_clean_patch_splits.py --seed 42
```

2. Build patient embedding features (GPU example):

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/infer/build_patient_embedding_features.py \
  --checkpoint outputs/patch_matrix/resnet18_s42/run_001/best.ckpt \
  --output-dir outputs/patient_features_full/resnet18_s42 \
  --device cuda --batch-size 192 --num-workers 8 --seed 42
```

3. Train patient classifier:

```bash
python3 src/train/train_patient_classifier_embed_cv.py \
  --train-feats outputs/patient_features_full/resnet18_s42/train_features.csv \
  --val-feats outputs/patient_features_full/resnet18_s42/val_features.csv \
  --test-feats outputs/patient_features_full/resnet18_s42/test_features.csv \
  --holdout-feats outputs/patient_features_full/resnet18_s42/holdout_features.csv \
  --output-root outputs/patient_full/resnet18_s42 \
  --seed 42 --candidate-family trees --baixa-prob-steps 66 --margin-steps 50
```

4. Generate patch heatmaps:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/infer/generate_patch_gradcam_heatmaps.py \
  --checkpoint outputs/patch_matrix/resnet18_s42/run_001/best.ckpt \
  --split-csv manifests/patch_test.csv \
  --output-dir outputs/patch_heatmaps/run_001 \
  --max-per-class 25 --seed 42 --device cuda
```

5. Export patient xlsx report:

```bash
python3 src/report/export_patient_prediction_report.py \
  --pred-csv outputs/patient_full/resnet18_s42/run_001/preds_holdout.csv \
  --labels-csv manifests/patient_labels.csv \
  --metrics-json outputs/patient_full/resnet18_s42/run_001/metrics.json \
  --out-xlsx outputs/patient_full/resnet18_s42/run_001/patient_prediction_report.xlsx
```

6. Build paper artifact package:

```bash
python3 src/report/build_paper_artifacts.py \
  --output-dir outputs/paper/run_001 \
  --reslut-paper-dir /hhome/ricse03/reslut/paper \
  --top-errors 30
```

---

## 7. Notes

- This repository snapshot is already cleaned to keep only key artifacts for reporting.
- HoldOut is used as final evaluation output only.
- No files are generated under `/hhome/ricse03/HelicoData`.
- Packaged deliverables are available under `/hhome/ricse03/reslut/`.
