#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
PATCH_PREDS_DIR = PROJECT_ROOT / 'outputs' / 'patch_preds'

PATCH_PREDS_HOLDOUT_DEFAULT = PATCH_PREDS_DIR / 'holdout.csv'
PATCH_PREDS_TRAIN_DEFAULT = PATCH_PREDS_DIR / 'train.csv'
PATCH_PREDS_VAL_DEFAULT = PATCH_PREDS_DIR / 'val.csv'
PATIENT_LABELS_DEFAULT = PROJECT_ROOT / 'manifests' / 'patient_labels.csv'
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features'

FEATURE_COLUMNS = [
    'num_patches',
    'mean_p',
    'std_p',
    'max_p',
    'topk_mean_p',
    'pos_ratio_05',
    'pos_ratio_07',
    'q90',
    'q50',
]


def resolve_optional_input(primary: Path, fallback: Path):
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return None


def load_patch_predictions(path: Path):
    df = pd.read_csv(path)
    required = {'Pat_ID', 'p_pos'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'{path} missing required columns: {missing}')

    out = df[['Pat_ID', 'p_pos']].copy()
    out['Pat_ID'] = out['Pat_ID'].astype(str).str.strip()
    out['p_pos'] = pd.to_numeric(out['p_pos'], errors='coerce')

    out = out.dropna(subset=['Pat_ID', 'p_pos'])
    out = out[out['Pat_ID'] != '']
    out = out.reset_index(drop=True)
    return out


def aggregate_patient_features(df: pd.DataFrame, topk: int = 5):
    rows = []
    for pat_id, grp in df.groupby('Pat_ID', sort=True):
        p = grp['p_pos'].to_numpy(dtype=float)
        p_sorted_desc = np.sort(p)[::-1]
        k = min(topk, len(p_sorted_desc))
        topk_mean = float(np.mean(p_sorted_desc[:k]))

        rows.append(
            {
                'Pat_ID': pat_id,
                'num_patches': int(len(p)),
                'mean_p': float(np.mean(p)),
                'std_p': float(np.std(p, ddof=0)),
                'max_p': float(np.max(p)),
                'topk_mean_p': topk_mean,
                'pos_ratio_05': float(np.mean(p > 0.5)),
                'pos_ratio_07': float(np.mean(p > 0.7)),
                'q90': float(np.quantile(p, 0.90)),
                'q50': float(np.quantile(p, 0.50)),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        out = pd.DataFrame(columns=['Pat_ID'] + FEATURE_COLUMNS)
    return out


def attach_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    out = features_df.merge(labels_df[['Pat_ID', 'DENSITAT']], on='Pat_ID', how='left')
    out['DENSITAT'] = out['DENSITAT'].fillna('')
    return out


def make_feature_table(pred_path: Path, labels_df: pd.DataFrame, topk: int):
    pred_df = load_patch_predictions(pred_path)
    if len(pred_df) == 0:
        feat = pd.DataFrame(columns=['Pat_ID'] + FEATURE_COLUMNS)
        feat = attach_labels(feat, labels_df)
        return pred_df, feat

    feat = aggregate_patient_features(pred_df, topk=topk)
    feat = attach_labels(feat, labels_df)
    return pred_df, feat


def save_empty_feature_table(path: Path):
    empty = pd.DataFrame(columns=['Pat_ID'] + FEATURE_COLUMNS + ['DENSITAT'])
    empty.to_csv(path, index=False)


def print_summary(split_name: str, pred_path: Path, pred_df: pd.DataFrame, feat_df: pd.DataFrame, out_path: Path):
    print(f'[{split_name}]')
    if pred_path is None:
        print('  input: missing -> saved empty feature table')
        print(f'  output: {out_path}')
        return

    in_rows = len(pred_df)
    out_pat = len(feat_df)
    unlabeled = int((feat_df['DENSITAT'] == '').sum()) if 'DENSITAT' in feat_df.columns else out_pat

    print(f'  input: {pred_path}')
    print(f'  input rows (patches): {in_rows}')
    print(f'  output rows (patients): {out_pat}')
    print(f'  unlabeled patients: {unlabeled}')
    print(f'  output: {out_path}')

    if out_pat > 0:
        show_cols = ['Pat_ID'] + FEATURE_COLUMNS + ['DENSITAT']
        print('  example rows:')
        print(feat_df[show_cols].head(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Aggregate patch-level predictions into patient-level feature tables.')
    parser.add_argument('--patch-preds-holdout', type=Path, default=PATCH_PREDS_HOLDOUT_DEFAULT)
    parser.add_argument('--patch-preds-train', type=Path, default=PATCH_PREDS_TRAIN_DEFAULT)
    parser.add_argument('--patch-preds-val', type=Path, default=PATCH_PREDS_VAL_DEFAULT)
    parser.add_argument('--patient-labels', type=Path, default=PATIENT_LABELS_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    if args.topk <= 0:
        raise ValueError('--topk must be > 0')

    if not args.patient_labels.exists():
        raise FileNotFoundError(f'Patient labels CSV not found: {args.patient_labels}')

    if not args.patch_preds_holdout.exists():
        raise FileNotFoundError(f'HoldOut patch predictions CSV not found: {args.patch_preds_holdout}')

    labels_df = pd.read_csv(args.patient_labels)
    required_labels = {'Pat_ID', 'DENSITAT'}
    missing = required_labels - set(labels_df.columns)
    if missing:
        raise RuntimeError(f'{args.patient_labels} missing required columns: {missing}')

    labels_df = labels_df[['Pat_ID', 'DENSITAT']].copy()
    labels_df['Pat_ID'] = labels_df['Pat_ID'].astype(str).str.strip()
    labels_df['DENSITAT'] = labels_df['DENSITAT'].astype(str).str.strip()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_pred_path = resolve_optional_input(
        args.patch_preds_train,
        PATCH_PREDS_DIR / 'patch_train_pred.csv',
    )
    val_pred_path = resolve_optional_input(
        args.patch_preds_val,
        PATCH_PREDS_DIR / 'patch_val_pred.csv',
    )
    holdout_pred_path = args.patch_preds_holdout

    out_train = args.output_dir / 'train_features.csv'
    out_val = args.output_dir / 'val_features.csv'
    out_holdout = args.output_dir / 'holdout_features.csv'

    if train_pred_path is not None:
        train_pred_df, train_feat = make_feature_table(train_pred_path, labels_df, topk=args.topk)
        train_feat.to_csv(out_train, index=False)
    else:
        train_pred_df = pd.DataFrame(columns=['Pat_ID', 'p_pos'])
        train_feat = pd.DataFrame(columns=['Pat_ID'] + FEATURE_COLUMNS + ['DENSITAT'])
        save_empty_feature_table(out_train)

    if val_pred_path is not None:
        val_pred_df, val_feat = make_feature_table(val_pred_path, labels_df, topk=args.topk)
        val_feat.to_csv(out_val, index=False)
    else:
        val_pred_df = pd.DataFrame(columns=['Pat_ID', 'p_pos'])
        val_feat = pd.DataFrame(columns=['Pat_ID'] + FEATURE_COLUMNS + ['DENSITAT'])
        save_empty_feature_table(out_val)

    holdout_pred_df, holdout_feat = make_feature_table(holdout_pred_path, labels_df, topk=args.topk)
    holdout_feat.to_csv(out_holdout, index=False)

    print_summary('train', train_pred_path, train_pred_df, train_feat, out_train)
    print_summary('val', val_pred_path, val_pred_df, val_feat, out_val)
    print_summary('holdout', holdout_pred_path, holdout_pred_df, holdout_feat, out_holdout)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
