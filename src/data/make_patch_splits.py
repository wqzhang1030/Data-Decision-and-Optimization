#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT_DEFAULT = Path('/hhome/ricse03/Deep_Learning_Group 3')


def _alloc_counts(n, val_ratio, test_ratio):
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    if n_test + n_val >= n:
        overflow = n_test + n_val - (n - 1)
        while overflow > 0 and (n_test > 0 or n_val > 0):
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            overflow -= 1

    return n_val, n_test


def _pick_strat_col(patient_df):
    patient_df = patient_df.copy()
    patient_df['presence_stratum'] = np.where(
        patient_df['pos_patches'] == 0,
        'neg_only',
        np.where(patient_df['neg_patches'] == 0, 'pos_only', 'mixed'),
    )
    counts = patient_df['presence_stratum'].value_counts()

    # Feasible if at least two strata exist and each has >=2 patients.
    if len(counts) >= 2 and int(counts.min()) >= 2:
        return 'presence_stratum', patient_df
    return 'DENSITAT', patient_df


def _split_patients(patient_df, strat_col, seed, val_ratio, test_ratio):
    rng = np.random.RandomState(seed)

    train_ids = []
    val_ids = []
    test_ids = []

    for _, group in patient_df.groupby(strat_col):
        ids = group['Pat_ID'].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_val, n_test = _alloc_counts(n, val_ratio, test_ratio)

        test_chunk = ids[:n_test]
        val_chunk = ids[n_test:n_test + n_val]
        train_chunk = ids[n_test + n_val:]

        train_ids.extend(train_chunk)
        val_ids.extend(val_chunk)
        test_ids.extend(test_chunk)

    # Ensure requested non-empty splits if possible.
    all_ids = train_ids + val_ids + test_ids
    n_total = len(all_ids)
    if n_total >= 3:
        if test_ratio > 0 and len(test_ids) == 0 and len(train_ids) > 1:
            test_ids.append(train_ids.pop())
        if val_ratio > 0 and len(val_ids) == 0 and len(train_ids) > 1:
            val_ids.append(train_ids.pop())

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def _format_split_summary(name, split_ids, patch_df, patient_df):
    split_patch = patch_df[patch_df['Pat_ID'].isin(split_ids)]
    split_patient = patient_df[patient_df['Pat_ID'].isin(split_ids)]

    presence_counts = split_patch['Presence'].value_counts().to_dict()
    densitat_counts = split_patient['DENSITAT'].value_counts().to_dict()

    lines = [
        f'[{name}]',
        f'patients={len(split_ids)}',
        f'patches={len(split_patch)}',
        f'presence_balance={presence_counts}',
        f'densitat_balance={densitat_counts}',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Create grouped patient splits for patch-level task.')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args = parser.parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or (args.val_ratio + args.test_ratio) >= 1.0:
        raise ValueError('Require 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1.0')

    patch_path = args.project_root / 'manifests' / 'patch_index.csv'
    patient_path = args.project_root / 'manifests' / 'patient_labels.csv'

    if not patch_path.exists():
        raise FileNotFoundError(f'Missing {patch_path}')
    if not patient_path.exists():
        raise FileNotFoundError(f'Missing {patient_path}')

    patch_df = pd.read_csv(patch_path)
    patient_df = pd.read_csv(patient_path)

    for col in ['Pat_ID', 'Presence']:
        if col not in patch_df.columns:
            raise RuntimeError(f'patch_index.csv missing required column: {col}')
    for col in ['Pat_ID', 'DENSITAT']:
        if col not in patient_df.columns:
            raise RuntimeError(f'patient_labels.csv missing required column: {col}')

    patch_df = patch_df.copy()
    patient_df = patient_df.copy()

    patch_df['Pat_ID'] = patch_df['Pat_ID'].astype(str).str.strip()
    patch_df['Presence'] = pd.to_numeric(patch_df['Presence'], errors='raise').astype(int)
    patient_df['Pat_ID'] = patient_df['Pat_ID'].astype(str).str.strip()
    patient_df['DENSITAT'] = patient_df['DENSITAT'].astype(str).str.strip()

    bad_presence = sorted(set(patch_df.loc[~patch_df['Presence'].isin([-1, 1]), 'Presence'].tolist()))
    if bad_presence:
        raise RuntimeError(f'patch_index Presence must be only -1/1, found: {bad_presence}')

    patient_stats = patch_df.groupby('Pat_ID').agg(
        patch_count=('Presence', 'size'),
        pos_patches=('Presence', lambda s: int((s == 1).sum())),
        neg_patches=('Presence', lambda s: int((s == -1).sum())),
    ).reset_index()

    patient_stats = patient_stats.merge(
        patient_df[['Pat_ID', 'DENSITAT']].drop_duplicates(),
        on='Pat_ID',
        how='left',
    )

    missing_labels = patient_stats.loc[patient_stats['DENSITAT'].isna(), 'Pat_ID'].tolist()
    if missing_labels:
        raise RuntimeError(f'Missing DENSITAT for {len(missing_labels)} patients. Sample: {missing_labels[:20]}')

    strat_col, patient_stats = _pick_strat_col(patient_stats)

    train_ids, val_ids, test_ids = _split_patients(
        patient_stats,
        strat_col=strat_col,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError('Pat_ID overlap detected across splits')

    split_dir = args.project_root / 'splits' / 'patch'
    split_dir.mkdir(parents=True, exist_ok=True)

    (split_dir / 'train_patients.txt').write_text('\n'.join(train_ids) + ('\n' if train_ids else ''), encoding='utf-8')
    (split_dir / 'val_patients.txt').write_text('\n'.join(val_ids) + ('\n' if val_ids else ''), encoding='utf-8')
    (split_dir / 'test_patients.txt').write_text('\n'.join(test_ids) + ('\n' if test_ids else ''), encoding='utf-8')

    lines = [
        'Patch Split Summary',
        f'seed={args.seed}',
        f'val_ratio={args.val_ratio}',
        f'test_ratio={args.test_ratio}',
        f'stratify_by={strat_col}',
        _format_split_summary('train', train_ids, patch_df, patient_stats),
        _format_split_summary('val', val_ids, patch_df, patient_stats),
        _format_split_summary('test', test_ids, patch_df, patient_stats),
        f'overlap_check={(len(train_set & val_set) == 0 and len(train_set & test_set) == 0 and len(val_set & test_set) == 0)}',
    ]
    summary = '\n\n'.join(lines)

    print(summary)
    (split_dir / 'summary.txt').write_text(summary + '\n', encoding='utf-8')

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
