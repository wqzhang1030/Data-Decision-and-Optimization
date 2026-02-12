# Agent Instructions for This Repository

## Scope and Safety
- Read this file first and follow it strictly.
- Do not modify, move, or write anything under `/hhome/ricse03/HelicoData` (read-only dataset).
- All generated artifacts must be written under this project root only: `/hhome/ricse03/Deep_Learning_Group 3`.

## Absolute Data Paths
- `DATA_ROOT=/hhome/ricse03/HelicoData`
- `XLSX=/hhome/ricse03/HelicoData/HP_WSI-CoordAnnotatedAllPatches.xlsx`
- `DIAG=/hhome/ricse03/HelicoData/PatientDiagnosis.csv`

## Hard Mapping Rules
- Patient mapping:
  - `patient_key = folder_name.split('_')[0]`
  - Example: `B22-25_0 -> B22-25`
  - `patient_key` must match xlsx `Pat_ID`.
- Window mapping:
  - `window_id = int(filename_stem)`
  - Examples: `02816.png -> 2816`, `00099.png -> 99`
  - `window_id` must match xlsx `Window_ID`.
- Patch labels from xlsx `Presence`:
  - Use only `Presence in {1, -1}`.
  - Ignore/drop `Presence == 0`.
  - Always log/report the count of dropped `Presence == 0` rows.

## Split and Leakage Rules
- All data splits must be grouped by patient (`Pat_ID`) to prevent leakage.
- No patient may appear across multiple splits.
- `HoldOut` is final test only.
- `HoldOut` must never be used for:
  - training
  - validation
  - hyperparameter tuning
  - threshold selection/calibration

## Output Layout (Project Root Only)
Write all generated files only under:
- `/hhome/ricse03/Deep_Learning_Group 3/manifests/`
- `/hhome/ricse03/Deep_Learning_Group 3/splits/`
- `/hhome/ricse03/Deep_Learning_Group 3/outputs/`

No generated artifacts may be written outside these directories.

## Quick Sanity Checks (Python Scripts)
Run from project root `/hhome/ricse03/Deep_Learning_Group 3`.

```bash
python check_data.py
python demo.py
```

If script arguments are implemented, pass the required absolute paths above and verify:
- patient-key and window-id mapping behavior matches rules
- `Presence == 0` is dropped and count is reported
- split files are grouped by `Pat_ID`
- `HoldOut` is excluded from any tuning/thresholding flow
