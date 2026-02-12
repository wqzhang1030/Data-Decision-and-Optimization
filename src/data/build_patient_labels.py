#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DIAG_DEFAULT = Path('/hhome/ricse03/HelicoData/PatientDiagnosis.csv')
PROJECT_ROOT_DEFAULT = Path('/hhome/ricse03/Deep_Learning_Group 3')


def normalize_pat_id(value: str) -> str:
    text = '' if value is None else str(value)
    text = text.strip().upper()
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r'\s+', '', text)
    text = text.replace('_', '-')
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description='Build patient_labels.csv from PatientDiagnosis.csv')
    parser.add_argument('--diag', type=Path, default=DIAG_DEFAULT)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT_DEFAULT)
    args = parser.parse_args()

    if not args.diag.exists():
        raise FileNotFoundError(f'DIAG CSV not found: {args.diag}')

    manifests_dir = args.project_root / 'manifests'
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifests_dir / 'patient_labels.csv'
    patch_index_path = manifests_dir / 'patch_index.csv'

    diag_df = pd.read_csv(args.diag)
    required = ['CODI', 'DENSITAT']
    missing = [c for c in required if c not in diag_df.columns]
    if missing:
        raise RuntimeError(f'DIAG missing required columns: {missing}')

    out_df = diag_df[required].copy()
    raw_codi = out_df['CODI'].astype(str)
    out_df['Pat_ID'] = raw_codi.map(normalize_pat_id)
    out_df['DENSITAT'] = out_df['DENSITAT'].astype(str).str.strip()

    normalized_count = int((raw_codi.str.strip() != out_df['Pat_ID']).sum())

    out_df = out_df.loc[out_df['Pat_ID'] != '', ['Pat_ID', 'DENSITAT']].copy()
    if out_df['DENSITAT'].eq('').any():
        bad = out_df.loc[out_df['DENSITAT'].eq(''), 'Pat_ID'].head(10).tolist()
        raise RuntimeError(f'Empty DENSITAT found for patients: {bad}')

    conflicts = out_df.groupby('Pat_ID')['DENSITAT'].nunique()
    conflicting_ids = conflicts[conflicts > 1].index.tolist()
    if conflicting_ids:
        raise RuntimeError(f'Conflicting DENSITAT labels for Pat_ID values: {conflicting_ids[:20]}')

    out_df = out_df.drop_duplicates(subset=['Pat_ID', 'DENSITAT']).sort_values('Pat_ID').reset_index(drop=True)
    out_df.to_csv(out_path, index=False)

    print(f'Input diagnosis rows: {len(diag_df)}')
    print(f'Output unique patients: {len(out_df)}')
    print(f'Pat_ID normalized count: {normalized_count}')
    if normalized_count > 0:
        print('Normalization applied: trim, uppercase, remove spaces, convert underscore/long-dash to hyphen.')

    if patch_index_path.exists():
        patch_df = pd.read_csv(patch_index_path)
        if 'Pat_ID' not in patch_df.columns:
            raise RuntimeError(f'patch_index missing Pat_ID column: {patch_index_path}')
        patch_patients = sorted(set(patch_df['Pat_ID'].astype(str).str.strip().tolist()))
        label_patients = set(out_df['Pat_ID'].tolist())
        missing_patients = [pid for pid in patch_patients if pid not in label_patients]
        if missing_patients:
            print('Missing Pat_ID in patient_labels for patch_index:', file=sys.stderr)
            for pid in missing_patients[:50]:
                print(f'  {pid}', file=sys.stderr)
            raise RuntimeError(f'Missing {len(missing_patients)} Pat_ID values required by patch_index.')
        print(f'Coverage check passed: all {len(patch_patients)} patch_index patients found in patient_labels.')
    else:
        print(f'Coverage check skipped: patch_index not found at {patch_index_path}')

    print(f'Output: {out_path}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
