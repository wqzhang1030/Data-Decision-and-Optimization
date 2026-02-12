#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT_DEFAULT = Path('/hhome/ricse03/Deep_Learning_Group 3')


def main() -> int:
    parser = argparse.ArgumentParser(description='Sanity checks for manifests/patch_index.csv and manifests/patient_labels.csv')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT_DEFAULT)
    args = parser.parse_args()

    manifests_dir = args.project_root / 'manifests'
    patch_path = manifests_dir / 'patch_index.csv'
    patient_path = manifests_dir / 'patient_labels.csv'

    if not patch_path.exists():
        raise FileNotFoundError(f'patch_index not found: {patch_path}')
    if not patient_path.exists():
        raise FileNotFoundError(f'patient_labels not found: {patient_path}')

    patch_df = pd.read_csv(patch_path)
    patient_df = pd.read_csv(patient_path)

    patch_required = ['image_path', 'Pat_ID', 'Window_ID', 'Presence']
    patient_required = ['Pat_ID', 'DENSITAT']

    missing_patch_cols = [c for c in patch_required if c not in patch_df.columns]
    if missing_patch_cols:
        raise RuntimeError(f'patch_index missing required columns: {missing_patch_cols}')

    missing_patient_cols = [c for c in patient_required if c not in patient_df.columns]
    if missing_patient_cols:
        raise RuntimeError(f'patient_labels missing required columns: {missing_patient_cols}')

    if patch_df.empty:
        raise RuntimeError('patch_index.csv is empty')
    if patient_df.empty:
        raise RuntimeError('patient_labels.csv is empty')

    patch_df['Pat_ID'] = patch_df['Pat_ID'].astype(str).str.strip()
    patch_df['image_path'] = patch_df['image_path'].astype(str).str.strip()
    patient_df['Pat_ID'] = patient_df['Pat_ID'].astype(str).str.strip()
    patient_df['DENSITAT'] = patient_df['DENSITAT'].astype(str).str.strip()

    null_checks = {
        'patch.image_path': int(patch_df['image_path'].isna().sum() + patch_df['image_path'].eq('').sum()),
        'patch.Pat_ID': int(patch_df['Pat_ID'].isna().sum() + patch_df['Pat_ID'].eq('').sum()),
        'patch.Window_ID': int(patch_df['Window_ID'].isna().sum()),
        'patch.Presence': int(patch_df['Presence'].isna().sum()),
        'patient.Pat_ID': int(patient_df['Pat_ID'].isna().sum() + patient_df['Pat_ID'].eq('').sum()),
        'patient.DENSITAT': int(patient_df['DENSITAT'].isna().sum() + patient_df['DENSITAT'].eq('').sum()),
    }
    bad_null = {k: v for k, v in null_checks.items() if v > 0}
    if bad_null:
        raise RuntimeError(f'Null/empty required fields detected: {bad_null}')

    dup_paths = int(patch_df['image_path'].duplicated().sum())
    if dup_paths > 0:
        raise RuntimeError(f'Duplicate image_path rows found: {dup_paths}')

    patch_df['Presence'] = pd.to_numeric(patch_df['Presence'], errors='raise').astype(int)
    invalid_presence = sorted(set(patch_df.loc[~patch_df['Presence'].isin([-1, 1]), 'Presence'].tolist()))
    if invalid_presence:
        raise RuntimeError(f'Invalid Presence values (expected only -1,1): {invalid_presence}')

    patient_label_conflicts = patient_df.groupby('Pat_ID')['DENSITAT'].nunique()
    conflicting = patient_label_conflicts[patient_label_conflicts > 1].index.tolist()
    if conflicting:
        raise RuntimeError(f'Conflicting DENSITAT labels for Pat_ID values: {conflicting[:20]}')

    patient_map = patient_df.drop_duplicates(subset=['Pat_ID', 'DENSITAT']).set_index('Pat_ID')['DENSITAT'].to_dict()

    patch_patients = sorted(set(patch_df['Pat_ID'].tolist()))
    missing_patients = [pid for pid in patch_patients if pid not in patient_map]
    if missing_patients:
        raise RuntimeError(f'Missing patient labels for {len(missing_patients)} Pat_ID values. Sample: {missing_patients[:20]}')

    presence_balance = Counter(patch_df['Presence'].tolist())
    densitat_balance = Counter(patient_map[pid] for pid in patch_patients)

    print(f'number of patches: {len(patch_df)}')
    print(f'number of unique Pat_ID: {len(patch_patients)}')

    print('Presence label balance:')
    for label in sorted(presence_balance):
        print(f'  {label}: {presence_balance[label]}')

    print('DENSITAT balance over Pat_ID in patch_index:')
    for label, count in sorted(densitat_balance.items()):
        print(f'  {label}: {count}')

    print(f'duplicate image_path check: OK (duplicates={dup_paths})')
    print('null checks for required fields: OK')

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
