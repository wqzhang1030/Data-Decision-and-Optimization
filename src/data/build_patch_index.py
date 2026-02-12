#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

DATA_ROOT_DEFAULT = Path('/hhome/ricse03/HelicoData')
XLSX_DEFAULT = Path('/hhome/ricse03/HelicoData/HP_WSI-CoordAnnotatedAllPatches.xlsx')
PROJECT_ROOT_DEFAULT = Path('/hhome/ricse03/Deep_Learning_Group 3')

PNG_PATTERN = re.compile(r'^\d+\.png$')
DIGITS_ONLY = re.compile(r'^\d+$')


def scan_pngs(data_root: Path) -> pd.DataFrame:
    annotated_root = data_root / 'CrossValidation' / 'Annotated'
    if not annotated_root.exists():
        raise FileNotFoundError('Annotated root not found: {}'.format(annotated_root))

    rows = []
    for p in annotated_root.rglob('*.png'):
        if not PNG_PATTERN.match(p.name):
            continue
        patient_folder = p.parent.name
        pat_id = patient_folder.split('_')[0]
        window_id = int(p.stem)
        rows.append(
            {
                'image_path': str(p.resolve()),
                'patient_folder': patient_folder,
                'Pat_ID': pat_id,
                'Window_ID': window_id,
            }
        )

    if not rows:
        raise RuntimeError('No digit-only PNG files found under {}'.format(annotated_root))

    png_df = pd.DataFrame(rows).sort_values('image_path').reset_index(drop=True)
    if png_df['image_path'].duplicated().any():
        dup_count = int(png_df['image_path'].duplicated().sum())
        raise RuntimeError('Duplicate image_path detected during scan: {}'.format(dup_count))

    return png_df


def load_xlsx(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError('XLSX not found: {}'.format(xlsx_path))

    df = pd.read_excel(xlsx_path, engine='openpyxl')
    required = ['Pat_ID', 'Section_ID', 'Window_ID', 'i', 'j', 'h', 'w', 'Presence']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError('XLSX missing required columns: {}'.format(missing))

    df = df[required].copy()
    df['Pat_ID'] = df['Pat_ID'].astype(str).str.strip()

    window_text = df['Window_ID'].astype(str).str.strip()
    numeric_mask = window_text.str.match(DIGITS_ONLY)
    skipped_non_numeric = int((~numeric_mask).sum())
    df = df.loc[numeric_mask].copy()

    df['Window_ID'] = pd.to_numeric(df['Window_ID'], errors='raise').astype(int)

    for col in ['Section_ID', 'i', 'j', 'h', 'w', 'Presence']:
        df[col] = pd.to_numeric(df[col], errors='raise').astype(int)

    df.attrs['skipped_non_numeric_window_id'] = skipped_non_numeric
    return df


def write_audit_csvs(
    manifests_dir: Path,
    unmatched_png_df: pd.DataFrame,
    unmatched_xlsx_df: pd.DataFrame,
) -> Tuple[Path, Path]:
    manifests_dir.mkdir(parents=True, exist_ok=True)

    unmatched_png_path = manifests_dir / 'unmatched_png.csv'
    unmatched_xlsx_path = manifests_dir / 'unmatched_xlsx.csv'

    unmatched_png_cols = ['Pat_ID', 'Window_ID', 'patient_folder', 'image_path']
    unmatched_png_df = unmatched_png_df.reindex(columns=unmatched_png_cols)
    unmatched_png_df.to_csv(unmatched_png_path, index=False)

    unmatched_xlsx_cols = ['Pat_ID', 'Window_ID', 'Presence', 'Section_ID', 'i', 'j', 'h', 'w']
    unmatched_xlsx_df = unmatched_xlsx_df.reindex(columns=unmatched_xlsx_cols)
    unmatched_xlsx_df.to_csv(unmatched_xlsx_path, index=False)

    return unmatched_png_path, unmatched_xlsx_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Build patch_index.csv from Annotated PNGs + XLSX Presence labels.')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument('--xlsx', type=Path, default=XLSX_DEFAULT)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT_DEFAULT)
    args = parser.parse_args()

    manifests_dir = args.project_root / 'manifests'
    patch_index_path = manifests_dir / 'patch_index.csv'

    png_df = scan_pngs(args.data_root)
    xlsx_df = load_xlsx(args.xlsx)

    key_cols = ['Pat_ID', 'Window_ID']

    png_key_counts = png_df.groupby(key_cols).size().rename('png_key_count').reset_index()
    png_with_key_counts = png_df.merge(png_key_counts, on=key_cols, how='left')

    xlsx_key_counts = xlsx_df.groupby(key_cols).size().rename('xlsx_key_count').reset_index()
    xlsx_with_key_counts = xlsx_df.merge(xlsx_key_counts, on=key_cols, how='left')

    merged = png_with_key_counts.merge(
        xlsx_with_key_counts,
        on=key_cols,
        how='left',
        suffixes=('_png', ''),
        indicator=True,
    )

    unmatched_png_df = merged.loc[
        (merged['_merge'] == 'left_only') | (merged['xlsx_key_count'].fillna(0) != 1),
        ['Pat_ID', 'Window_ID', 'patient_folder', 'image_path'],
    ].drop_duplicates()

    png_key_set = set(zip(png_df['Pat_ID'], png_df['Window_ID']))
    xlsx_unmatched_mask = ~xlsx_df.apply(lambda r: (r['Pat_ID'], int(r['Window_ID'])) in png_key_set, axis=1)
    unmatched_xlsx_df = xlsx_df.loc[
        xlsx_unmatched_mask,
        ['Pat_ID', 'Window_ID', 'Presence', 'Section_ID', 'i', 'j', 'h', 'w'],
    ].copy()

    unmatched_png_path, unmatched_xlsx_path = write_audit_csvs(manifests_dir, unmatched_png_df, unmatched_xlsx_df)

    strict_fail = len(unmatched_png_df) > 0
    if strict_fail:
        print('Scanned PNGs (digit-only filenames): {}'.format(len(png_df)))
        print('XLSX rows after numeric Window_ID filtering: {}'.format(len(xlsx_df)))
        print('XLSX rows skipped (non-numeric Window_ID): {}'.format(xlsx_df.attrs.get('skipped_non_numeric_window_id', 0)))
        print('Unmatched PNG keys (strict failures): {}'.format(len(unmatched_png_df)))
        print('Unmatched XLSX rows (no PNG key): {}'.format(len(unmatched_xlsx_df)))
        print('Audit written: {}'.format(unmatched_png_path))
        print('Audit written: {}'.format(unmatched_xlsx_path))
        print('ERROR: strict join failed; each scanned PNG must match exactly one XLSX row on (Pat_ID, Window_ID).', file=sys.stderr)
        return 1

    matched = merged.loc[
        (merged['_merge'] == 'both') & (merged['xlsx_key_count'] == 1),
        ['image_path', 'Pat_ID', 'Window_ID', 'Presence', 'Section_ID', 'i', 'j', 'h', 'w'],
    ].copy()

    dropped_presence_0 = int((matched['Presence'] == 0).sum())
    patch_index = matched.loc[matched['Presence'].isin([-1, 1])].copy()

    patch_index = patch_index.sort_values(['Pat_ID', 'Window_ID', 'image_path']).reset_index(drop=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    patch_index.to_csv(patch_index_path, index=False)

    print('Scanned PNGs (digit-only filenames): {}'.format(len(png_df)))
    print('Matched rows before Presence filter: {}'.format(len(matched)))
    print('Dropped Presence==0 rows: {}'.format(dropped_presence_0))
    print('Rows written to patch_index.csv: {}'.format(len(patch_index)))
    print('XLSX rows skipped (non-numeric Window_ID): {}'.format(xlsx_df.attrs.get('skipped_non_numeric_window_id', 0)))
    print('Unmatched PNG keys: {}'.format(len(unmatched_png_df)))
    print('Unmatched XLSX rows: {}'.format(len(unmatched_xlsx_df)))
    print('Output: {}'.format(patch_index_path))

    return 0


if __name__ == '__main__':
    sys.exit(main())
