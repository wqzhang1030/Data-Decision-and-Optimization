太好了 👍 我给你做一个**完整、正式、学术英文翻译版本**（保持结构、不过度删减、表达自然，不是直译中式英语）。

---

# A Two-Stage Patch–Patient Learning Framework for *Helicobacter pylori*–Related Diagnosis in Histopathology

---

## Abstract

This study focuses on histopathological image analysis for *Helicobacter pylori*–related diagnosis and proposes a complete two-stage framework bridging patch-level prediction and patient-level classification.

In the first stage, a binary classifier is trained using patch-level labels (`Presence ∈ {1, -1}`). In the second stage, patch-level outputs and deep embedding features are aggregated at the patient level to perform three-class classification (`NEGATIVA / BAIXA / ALTA`).

Under strict patient-level grouping (grouped by `Pat_ID`) and a final HoldOut evaluation protocol, the best-performing patient-level model achieves the following results on the HoldOut set:
Accuracy = 0.7759, Macro-F1 = 0.7492, Balanced Accuracy = 0.7368, and Macro OvR ROC-AUC = 0.7978.

For the intermediate BAIXA class, the model achieves Precision = 0.6250, Recall = 0.5882, and F1 = 0.6061.

Additionally, patch-level Grad-CAM heatmaps and a patient-level prediction report (xlsx) were generated to support interpretability and error tracking.

---

## 1. Research Objectives and Problem Definition

### 1.1 Task Objectives

The objective of this project is to construct a reproducible and interpretable histopathological image analysis pipeline composed of two hierarchical tasks:

1. **Patch-Level Task (Task 1):**
   Predict the `Presence` label (positive/negative) for each patch.

2. **Patient-Level Task (Task 2):**
   Predict the patient-level `DENSITAT` three-class label based on aggregated patch information from the same patient.

---

### 1.2 Label System

* **Patch labels:** Extracted from the annotation xlsx file, where `Presence ∈ {1, -1}`. Samples with `Presence == 0` are treated as invalid and discarded.

* **Patient labels:** Extracted from `PatientDiagnosis.csv`, where
  `DENSITAT ∈ {NEGATIVA, BAIXA, ALTA}`.

---

## 2. Data and Mapping Rules

### 2.1 Data Sources

* Image directory:
  `/hhome/ricse03/HelicoData/CrossValidation/Annotated`

* Patch annotation file:
  `/hhome/ricse03/HelicoData/HP_WSI-CoordAnnotatedAllPatches.xlsx`

* Patient diagnosis file:
  `/hhome/ricse03/HelicoData/PatientDiagnosis.csv`

---

### 2.2 Key Mapping Rules (Engineering Constraints)

1. `Pat_ID = patient_folder.split('_')[0]`
2. `Window_ID` is matched through normalized mapping between image filename stems and xlsx `Window_ID`
3. Only `Presence ∈ {1, -1}` is retained; `Presence == 0` is ignored

---

### 2.3 Data Cleaning and Sample Retention

During the clean rebuild process:

* Total xlsx rows: 2695
* Retained patches: 2676

Discarded samples:

* Invalid `Presence` labels (not in `{1, -1}`): 4
* Missing PNG matches: 15

Overall patient label distribution (`patient_labels.csv`):

* NEGATIVA: 151
* ALTA: 86
* BAIXA: 72

---

## 3. Overall Technical Framework

### 3.1 Two-Stage Architecture

#### Stage A: Patch-Level Binary Classification

* Backbone: ResNet family (best retained model: `resnet18_s42`)
* Output: patch-level positive probability `p_pos`

---

#### Stage B: Patient-Level Three-Class Classification

Input features:

* Statistical summaries of patch probabilities within the same patient
* Aggregated embedding features (mean and standard deviation)

Classifier:

* Tree-based model selected via cross-validation
* Best-performing run: `et_baixa_boost`

Output:

* Patient-level predicted class (`pred_class`)
* Three-class probability vector

---

### 3.2 Data Leakage Prevention

* Train/validation/test splits are performed strictly at the **patient level**
* No patient appears in more than one split
* The HoldOut set is reserved exclusively for final evaluation and is not used in training or hyperparameter tuning

---

## 4. Implementation Details

### 4.1 Patch-Level Implementation

* Training script: `src/train/train_patch_presence.py`
* Model definition: `src/models/resnet_presence.py`
* Best checkpoint:
  `outputs/patch_matrix/resnet18_s42/run_001/best.ckpt`

Clean split statistics:

* Train: 1676 patches / 107 patients
* Validation: 305 patches / 23 patients
* Test: 695 patches / 24 patients

Presence label distribution:

* Train: +1:1086, -1:590
* Validation: +1:221, -1:84
* Test: +1:153, -1:542

---

### 4.2 Patient-Level Feature Construction

* Feature script: `src/infer/build_patient_embedding_features.py`

Feature sources:

1. Statistical features of patch probabilities (mean, variance, quantiles, top-k, etc.)
2. Aggregated deep embedding features (mean and standard deviation)

Final feature dimensionality under the best configuration: 1033.

---

### 4.3 Patient-Level Classification and Evaluation

* Training script: `src/train/train_patient_classifier_embed_cv.py`

Strategy:

* Cross-validation-based model selection
* BAIXA-oriented threshold and model search (balancing Macro-F1 and BAIXA recall)

Best result directory:
`outputs/patient_full/resnet18_s42/run_001/`

---

### 4.4 Interpretability and Reporting

* Patch Grad-CAM generation:
  `src/infer/generate_patch_gradcam_heatmaps.py`
  Output: `outputs/patch_heatmaps/run_001/`
  Generated: 50 heatmaps (25 positive + 25 negative)

* Patient-level prediction report:
  `src/report/export_patient_prediction_report.py`
  Output: `patient_prediction_report.xlsx`
  Includes ground truth, prediction, correctness, confidence, and error case tables.

---

## 5. Experimental Results

### 5.1 Patch-Level Results (Best Patch Run)

Source:
`outputs/patch_matrix/resnet18_s42/run_001/metrics.json`

Validation:

* Accuracy: 0.9804
* F1: 0.9853
* ROC-AUC: 0.9947
* PR-AUC: 0.9977

Test:

* Accuracy: 0.9505
* F1: 0.9499
* ROC-AUC: 0.9792
* PR-AUC: 0.9863

---

### 5.2 Patient-Level Results (Best Patient Run)

Source:
`outputs/patient_full/resnet18_s42/run_001/metrics.json`

Overall performance:

* Accuracy: 0.7759
* Macro-F1: 0.7492
* Macro-Precision: 0.7677
* Macro-Recall: 0.7368
* Balanced Accuracy: 0.7368
* Macro OvR ROC-AUC: 0.7978
* Macro OvR PR-AUC: 0.6709
* Brier Score: 0.4384
* LogLoss: 2.5991
* ECE: 0.1296

Per-class metrics:

* NEGATIVA: P 0.8281 / R 0.9138 / F1 0.8689
* BAIXA: P 0.6250 / R 0.5882 / F1 0.6061
* ALTA: P 0.8500 / R 0.7083 / F1 0.7727

Confusion matrix:

```
[[53, 5, 0],
 [11, 20, 3],
 [0, 7, 17]]
```

---

### 5.3 Comparison with Earlier Baselines (BAIXA Improvement)

Compared with earlier baseline runs (e.g., `outputs/patient/run_009`), BAIXA recall improved from approximately 0.235 to 0.588. At the same time, overall Macro-F1 increased to 0.749.

This demonstrates that patient-level stratified splitting and BAIXA-oriented model selection significantly improve intermediate-class recognition.

---

## 6. Error Analysis and Interpretability

### 6.1 Error Overview

Among 116 HoldOut patients, 26 were misclassified (~22.4%).

From the error reports (`patient_prediction_report.xlsx` and `paper_error_cases_top.csv`):

* High-confidence errors are primarily concentrated in confusion between BAIXA and adjacent classes (NEGATIVA / ALTA).
* This indicates overlapping feature representations for intermediate disease states.

---

### 6.2 Grad-CAM Observations

From `outputs/patch_heatmaps/run_001/`:

* Both positive and negative samples are covered.
* Heatmaps provide visual confirmation that model attention aligns with histopathological cues.

---

## 7. Reproducibility and Artifacts

### 7.1 Key Result Directories

* Patch best run:
  `outputs/patch_matrix/resnet18_s42/run_001/`

* Patient best run:
  `outputs/patient_full/resnet18_s42/run_001/`

* Summary metrics:
  `outputs/patient_full/summary_metrics.csv`
  `outputs/patient_full/summary_top.json`

* Interpretability outputs:
  `outputs/patch_heatmaps/run_001/`

* Paper artifact bundle:
  `outputs/paper/run_001/`

---

## 8. Conclusion and Future Work

This work establishes a complete engineering pipeline from patch-level modeling to patient-level diagnosis, achieving Accuracy = 0.7759 and Macro-F1 = 0.7492 on the patient-level three-class task.

The primary limitation lies in distinguishing the intermediate BAIXA class from adjacent categories.

Future directions include:

1. Hard-example reweighting based on high-confidence errors
2. Stain normalization and stronger domain augmentation
3. Sequence- or attention-based bag modeling at the patient level
4. External validation on independent cohorts


