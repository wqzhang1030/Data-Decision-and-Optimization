#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

XLSX_DEFAULT = Path('/hhome/ricse03/HelicoData/HP_WSI-CoordAnnotatedAllPatches.xlsx')
IMAGE_ROOT_DEFAULT = Path('/hhome/ricse03/HelicoData/CrossValidation/Annotated')
OUTPUT_DIR_DEFAULT = Path('/hhome/ricse03/Deep_Learning_Group 3/manifests')
PATIENT_LABELS_DEFAULT = Path('/hhome/ricse03/Deep_Learning_Group 3/manifests/patient_labels.csv')

NUMERIC_RE = re.compile(r'^\d+(?:\.0+)?$')
DIGIT_SUFFIX_RE = re.compile(r'^(\d+)_(.+)$')


def canonicalize_window_id(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == '':
        return None

    if NUMERIC_RE.fullmatch(s):
        return str(int(float(s)))

    m = DIGIT_SUFFIX_RE.fullmatch(s)
    if m is not None:
        lead = str(int(m.group(1)))
        suffix = m.group(2)
        if suffix == '':
            return None
        return f'{lead}_{suffix}'

    return None


def scan_png_index(image_root: Path):
    if not image_root.exists():
        raise FileNotFoundError(f'IMAGE_ROOT not found: {image_root}')

    png_paths = sorted(image_root.rglob('*.png'))
    index = {}
    duplicate_key_count = 0
    unrecognized_png_stem = 0

    for p in png_paths:
        patient_folder = p.parent.name
        pat_id = patient_folder.split('_')[0]
        canonical = canonicalize_window_id(p.stem)
        if canonical is None:
            unrecognized_png_stem += 1
            continue

        key = (pat_id, canonical)
        abs_path = str(p.resolve())
        if key in index:
            duplicate_key_count += 1
            continue
        index[key] = abs_path

    return index, len(png_paths), unrecognized_png_stem, duplicate_key_count


def allocate_counts(n: int):
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    n_test = n - n_train - n_val

    # For classes with >=3 patients, keep every split non-empty.
    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1

    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    while n_train + n_val + n_test < n:
        n_train += 1

    return n_train, n_val, n_test


def make_splits_by_patient(df: pd.DataFrame, patient_labels: pd.DataFrame, seed: int):
    patient_level = (
        df[['Pat_ID']]
        .drop_duplicates()
        .merge(patient_labels[['Pat_ID', 'DENSITAT']], on='Pat_ID', how='left')
    )
    missing = patient_level[patient_level['DENSITAT'].isna()]['Pat_ID'].tolist()
    if missing:
        raise RuntimeError(f'Missing DENSITAT for {len(missing)} patients. Sample: {missing[:10]}')

    rng = np.random.RandomState(seed)
    train_pat, val_pat, test_pat = [], [], []
    for _, grp in patient_level.groupby('DENSITAT', sort=True):
        ids = grp['Pat_ID'].astype(str).tolist()
        rng.shuffle(ids)
        n_train, n_val, n_test = allocate_counts(len(ids))
        train_pat.extend(ids[:n_train])
        val_pat.extend(ids[n_train:n_train + n_val])
        test_pat.extend(ids[n_train + n_val:n_train + n_val + n_test])

    train_pat = sorted(train_pat)
    val_pat = sorted(val_pat)
    test_pat = sorted(test_pat)

    train_df = df[df['Pat_ID'].isin(train_pat)].copy()
    val_df = df[df['Pat_ID'].isin(val_pat)].copy()
    test_df = df[df['Pat_ID'].isin(test_pat)].copy()

    return train_df, val_df, test_df, train_pat, val_pat, test_pat


def main():
    parser = argparse.ArgumentParser(description='Rebuild clean patch-level dataset splits with canonical Window_ID matching.')
    parser.add_argument('--xlsx', type=Path, default=XLSX_DEFAULT)
    parser.add_argument('--image-root', type=Path, default=IMAGE_ROOT_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--patient-labels', type=Path, default=PATIENT_LABELS_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    png_index, total_png, dropped_png_unrecognized, png_duplicate_keys = scan_png_index(args.image_root)

    if not args.xlsx.exists():
        raise FileNotFoundError(f'XLSX not found: {args.xlsx}')

    xlsx = pd.read_excel(args.xlsx, engine='openpyxl')
    required = {'Pat_ID', 'Window_ID', 'Presence'}
    if not required.issubset(xlsx.columns):
        raise RuntimeError(f'XLSX missing required columns: {required - set(xlsx.columns)}')

    total_rows = len(xlsx)

    work = xlsx[['Pat_ID', 'Window_ID', 'Presence']].copy()
    work['Pat_ID'] = work['Pat_ID'].astype(str).str.strip()
    work['Window_ID_raw'] = work['Window_ID'].apply(lambda x: '' if pd.isna(x) else str(x).strip())

    work['Presence'] = pd.to_numeric(work['Presence'], errors='coerce')
    valid_presence_mask = work['Presence'].isin([-1, 1])
    drop_presence = int((~valid_presence_mask).sum())
    work = work[valid_presence_mask].copy()
    work['Presence'] = work['Presence'].astype(int)

    work['Window_ID_canonical'] = work['Window_ID'].apply(canonicalize_window_id)
    unrecognized_mask = work['Window_ID_canonical'].isna()
    drop_unrecognized = int(unrecognized_mask.sum())
    work = work[~unrecognized_mask].copy()

    keys = list(zip(work['Pat_ID'].tolist(), work['Window_ID_canonical'].tolist()))
    image_paths = [png_index.get(k) for k in keys]
    work['image_path'] = image_paths

    no_match_mask = work['image_path'].isna()
    drop_no_match = int(no_match_mask.sum())
    kept = work[~no_match_mask].copy()

    out = kept[['image_path', 'Pat_ID', 'Window_ID_raw', 'Window_ID_canonical', 'Presence']].copy()
    out = out.sort_values(['Pat_ID', 'Window_ID_canonical', 'image_path']).reset_index(drop=True)

    # Acceptance checks
    if not set(out['Presence'].unique().tolist()).issubset({-1, 1}):
        raise RuntimeError('Presence contains values outside {-1,1}.')

    missing_paths = [p for p in out['image_path'].tolist() if not Path(p).exists()]
    if missing_paths:
        raise RuntimeError(f'image_path contains missing files, sample: {missing_paths[:5]}')

    if not args.patient_labels.exists():
        raise FileNotFoundError(f'patient_labels not found: {args.patient_labels}')
    patient_labels = pd.read_csv(args.patient_labels)
    if not {'Pat_ID', 'DENSITAT'}.issubset(patient_labels.columns):
        missing_cols = {'Pat_ID', 'DENSITAT'} - set(patient_labels.columns)
        raise RuntimeError(f'patient_labels missing required columns: {missing_cols}')
    patient_labels = patient_labels[['Pat_ID', 'DENSITAT']].copy()
    patient_labels['Pat_ID'] = patient_labels['Pat_ID'].astype(str).str.strip()
    patient_labels['DENSITAT'] = patient_labels['DENSITAT'].astype(str).str.strip()
    patient_labels = patient_labels[patient_labels['Pat_ID'] != '']

    train_df, val_df, test_df, train_pat, val_pat, test_pat = make_splits_by_patient(
        out,
        patient_labels=patient_labels,
        seed=args.seed,
    )

    train_set, val_set, test_set = set(train_pat), set(val_pat), set(test_pat)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError('Pat_ID overlap found across splits.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / 'patch_train.csv'
    val_path = args.output_dir / 'patch_val.csv'
    test_path = args.output_dir / 'patch_test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Ground-truth example check
    example_key = ('B22-302', '739_Aug4')
    has_xlsx = bool(((xlsx['Pat_ID'].astype(str).str.strip() == example_key[0]) &
                     (xlsx['Window_ID'].apply(canonicalize_window_id) == example_key[1]) &
                     (pd.to_numeric(xlsx['Presence'], errors='coerce').isin([-1, 1]))).any())
    has_png = example_key in png_index
    has_kept = bool(((out['Pat_ID'] == example_key[0]) & (out['Window_ID_canonical'] == example_key[1])).any())

    if has_xlsx and has_png and not has_kept:
        raise RuntimeError('Example (B22-302, 739_Aug4) exists in XLSX+PNG but was not matched.')

    print(f'total PNG scanned: {total_png}')
    print(f'PNG dropped (unrecognized stem): {dropped_png_unrecognized}')
    print(f'PNG duplicate keys ignored: {png_duplicate_keys}')

    print(f'total XLSX rows: {total_rows}')
    print(f'kept rows: {len(out)}')
    print('dropped rows breakdown:')
    print(f'  Presence not in {{-1,1}}: {drop_presence}')
    print(f'  Window_ID unrecognized: {drop_unrecognized}')
    print(f'  no matching PNG: {drop_no_match}')

    print('split summary:')
    print(f'  train: patients={len(train_pat)}, patches={len(train_df)}')
    print(f'  val:   patients={len(val_pat)}, patches={len(val_df)}')
    print(f'  test:  patients={len(test_pat)}, patches={len(test_df)}')
    for name, pats in [('train', train_pat), ('val', val_pat), ('test', test_pat)]:
        dens = (
            patient_labels[patient_labels['Pat_ID'].isin(pats)]['DENSITAT']
            .value_counts()
            .to_dict()
        )
        print(f'  {name} DENSITAT={dens}')

    print('acceptance checks:')
    print(f'  all image_path exist: {len(missing_paths) == 0}')
    print(f'  Presence only {{-1,1}}: {set(out["Presence"].unique().tolist()).issubset({-1,1})}')
    print(f'  no Pat_ID overlap: {len(train_set & val_set) == 0 and len(train_set & test_set) == 0 and len(val_set & test_set) == 0}')
    print(f'  example B22-302 + 739_Aug4 -> has_xlsx={has_xlsx}, has_png={has_png}, matched={has_kept}')

    print(f'output files: {train_path}, {val_path}, {test_path}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
