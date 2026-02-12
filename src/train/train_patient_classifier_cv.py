#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')

TRAIN_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'train_features.csv'
VAL_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'val_features.csv'
HOLDOUT_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'holdout_features.csv'
PATCH_TEST_PRED_DEFAULT = PROJECT_ROOT / 'outputs' / 'patch_preds' / 'patch_test_pred.csv'
PATIENT_LABELS_DEFAULT = PROJECT_ROOT / 'manifests' / 'patient_labels.csv'
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient' / 'run_002'

CLASS_ORDER = ['NEGATIVA', 'BAIXA', 'ALTA']
BASE_FEATURES = [
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


def load_feat_csv(path: Path):
    df = pd.read_csv(path)
    req = {'Pat_ID'} | set(BASE_FEATURES)
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f'{path} missing columns: {miss}')
    df = df.copy()
    df['Pat_ID'] = df['Pat_ID'].astype(str).str.strip()
    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Pat_ID'] + BASE_FEATURES)
    df = df[df['Pat_ID'] != '']
    keep = ['Pat_ID'] + BASE_FEATURES
    return df[keep].copy()


def aggregate_patch_pred(path: Path, topk: int = 5):
    d = pd.read_csv(path)
    req = {'Pat_ID', 'p_pos'}
    miss = req - set(d.columns)
    if miss:
        raise RuntimeError(f'{path} missing columns: {miss}')

    d = d.copy()
    d['Pat_ID'] = d['Pat_ID'].astype(str).str.strip()
    d['p_pos'] = pd.to_numeric(d['p_pos'], errors='coerce')
    d = d.dropna(subset=['Pat_ID', 'p_pos'])
    d = d[d['Pat_ID'] != '']

    rows = []
    for pid, g in d.groupby('Pat_ID', sort=True):
        p = g['p_pos'].to_numpy(dtype=float)
        s = np.sort(p)[::-1]
        k = min(topk, len(s))
        rows.append(
            {
                'Pat_ID': pid,
                'num_patches': int(len(p)),
                'mean_p': float(np.mean(p)),
                'std_p': float(np.std(p, ddof=0)),
                'max_p': float(np.max(p)),
                'topk_mean_p': float(np.mean(s[:k])),
                'pos_ratio_05': float(np.mean(p > 0.5)),
                'pos_ratio_07': float(np.mean(p > 0.7)),
                'q90': float(np.quantile(p, 0.90)),
                'q50': float(np.quantile(p, 0.50)),
            }
        )
    return pd.DataFrame(rows)


def add_derived_features(df: pd.DataFrame):
    out = df.copy()
    eps = 1e-8
    out['log_num_patches'] = np.log1p(out['num_patches'])
    out['cv_p'] = out['std_p'] / (out['mean_p'] + eps)
    out['iqr_like'] = out['q90'] - out['q50']
    out['peak_gap'] = out['max_p'] - out['q90']
    out['topk_gap'] = out['topk_mean_p'] - out['mean_p']
    return out


def get_feature_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in {'Pat_ID', 'DENSITAT'}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def load_labels(path: Path):
    df = pd.read_csv(path)
    req = {'Pat_ID', 'DENSITAT'}
    if req.issubset(df.columns):
        out = df[['Pat_ID', 'DENSITAT']].copy()
    elif {'CODI', 'DENSITAT'}.issubset(df.columns):
        out = df[['CODI', 'DENSITAT']].rename(columns={'CODI': 'Pat_ID'}).copy()
    else:
        raise RuntimeError(f'Label CSV missing required columns: {path}')

    out['Pat_ID'] = out['Pat_ID'].astype(str).str.strip()
    out['DENSITAT'] = out['DENSITAT'].astype(str).str.strip()
    out = out[out['DENSITAT'].isin(CLASS_ORDER)]
    out = out.drop_duplicates(subset=['Pat_ID'])
    return out


def plot_cm(cm: np.ndarray, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title='HoldOut Confusion Matrix',
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def choose_model(X, y, seed: int):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    candidates = {
        'lr_balanced': Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'model',
                    LogisticRegression(
                        multi_class='multinomial',
                        class_weight='balanced',
                        max_iter=5000,
                        C=3.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        'rf_balanced_leaf4': RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            min_samples_leaf=4,
            max_depth=None,
            random_state=seed,
        ),
    }

    scores = []
    for name, model in candidates.items():
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
        cv_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        scores.append(
            {
                'name': name,
                'cv_macro_f1_mean': float(np.mean(cv_f1)),
                'cv_macro_f1_std': float(np.std(cv_f1)),
                'cv_acc_mean': float(np.mean(cv_acc)),
                'cv_acc_std': float(np.std(cv_acc)),
                'model': model,
            }
        )

    scores = sorted(scores, key=lambda x: x['cv_macro_f1_mean'], reverse=True)
    best = scores[0]
    return best, scores


def ensure_prob_cols(prob: np.ndarray, model_classes, class_order):
    idx = {c: i for i, c in enumerate(model_classes)}
    out = np.zeros((prob.shape[0], len(class_order)), dtype=float)
    for j, c in enumerate(class_order):
        if c in idx:
            out[:, j] = prob[:, idx[c]]
    return out


def main():
    parser = argparse.ArgumentParser(description='Patient-level classifier with CV model selection on annotated only.')
    parser.add_argument('--train-feats', type=Path, default=TRAIN_FEATS_DEFAULT)
    parser.add_argument('--val-feats', type=Path, default=VAL_FEATS_DEFAULT)
    parser.add_argument('--holdout-feats', type=Path, default=HOLDOUT_FEATS_DEFAULT)
    parser.add_argument('--patch-test-pred', type=Path, default=PATCH_TEST_PRED_DEFAULT)
    parser.add_argument('--patient-labels', type=Path, default=PATIENT_LABELS_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    labels = load_labels(args.patient_labels)

    tr = load_feat_csv(args.train_feats)
    va = load_feat_csv(args.val_feats)

    if args.patch_test_pred.exists():
        te = aggregate_patch_pred(args.patch_test_pred)
    else:
        te = pd.DataFrame(columns=['Pat_ID'] + BASE_FEATURES)

    ann = pd.concat([tr, va, te], ignore_index=True).drop_duplicates(subset=['Pat_ID']).copy()
    ann = ann.merge(labels, on='Pat_ID', how='inner')

    ann = add_derived_features(ann)

    hold = load_feat_csv(args.holdout_feats)
    hold = add_derived_features(hold)
    hold = hold.merge(labels, on='Pat_ID', how='left')

    feat_cols = get_feature_cols(ann)
    X = ann[feat_cols].to_numpy(dtype=float)
    y = ann['DENSITAT'].to_numpy()

    best, ranking = choose_model(X, y, seed=args.seed)
    model = best['model']
    model.fit(X, y)

    X_hold = hold[feat_cols].to_numpy(dtype=float)
    hold_pred = model.predict(X_hold)
    hold_prob_raw = model.predict_proba(X_hold)

    model_classes = list(model.named_steps['model'].classes_) if isinstance(model, Pipeline) else list(model.classes_)
    hold_prob = ensure_prob_cols(hold_prob_raw, model_classes, CLASS_ORDER)

    out_pred = pd.DataFrame(
        {
            'Pat_ID': hold['Pat_ID'].astype(str).to_numpy(),
            'pred_class': hold_pred,
            'p_NEGATIVA': hold_prob[:, 0],
            'p_BAIXA': hold_prob[:, 1],
            'p_ALTA': hold_prob[:, 2],
        }
    )

    # HoldOut evaluation is reporting only; model selection done on annotated CV.
    eval_df = hold[hold['DENSITAT'].isin(CLASS_ORDER)].copy()
    eval_pred = out_pred[out_pred['Pat_ID'].isin(set(eval_df['Pat_ID']))].merge(
        eval_df[['Pat_ID', 'DENSITAT']], on='Pat_ID', how='inner'
    )

    hold_acc = float(accuracy_score(eval_pred['DENSITAT'], eval_pred['pred_class'])) if len(eval_pred) else None
    hold_macro_f1 = float(f1_score(eval_pred['DENSITAT'], eval_pred['pred_class'], average='macro')) if len(eval_pred) else None
    cm = confusion_matrix(eval_pred['DENSITAT'], eval_pred['pred_class'], labels=CLASS_ORDER) if len(eval_pred) else np.zeros((3, 3), dtype=int)
    report = (
        classification_report(eval_pred['DENSITAT'], eval_pred['pred_class'], labels=CLASS_ORDER, output_dict=True, zero_division=0)
        if len(eval_pred)
        else {}
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / 'model.pkl'
    metrics_path = args.output_dir / 'metrics.json'
    pred_path = args.output_dir / 'preds_holdout.csv'
    cm_path = args.output_dir / 'confusion_matrix.png'

    joblib.dump(model, model_path)
    out_pred.to_csv(pred_path, index=False)
    plot_cm(cm, CLASS_ORDER, cm_path)

    payload = {
        'seed': args.seed,
        'class_order': CLASS_ORDER,
        'feature_cols': feat_cols,
        'annotated_training': {
            'num_patients': int(len(ann)),
            'label_counts': ann['DENSITAT'].value_counts().to_dict(),
        },
        'cv_model_selection': [
            {
                'name': r['name'],
                'cv_macro_f1_mean': r['cv_macro_f1_mean'],
                'cv_macro_f1_std': r['cv_macro_f1_std'],
                'cv_acc_mean': r['cv_acc_mean'],
                'cv_acc_std': r['cv_acc_std'],
            }
            for r in ranking
        ],
        'selected_model': {
            'name': best['name'],
            'cv_macro_f1_mean': best['cv_macro_f1_mean'],
            'cv_acc_mean': best['cv_acc_mean'],
        },
        'holdout_evaluation': {
            'num_patients_scored': int(len(eval_pred)),
            'accuracy': hold_acc,
            'macro_f1': hold_macro_f1,
            'confusion_matrix': cm.tolist(),
            'per_class_report': {k: report[k] for k in CLASS_ORDER if k in report},
        },
        'paths': {
            'model': str(model_path),
            'preds_holdout': str(pred_path),
            'metrics': str(metrics_path),
            'confusion_matrix': str(cm_path),
        },
    }

    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'selected model: {best["name"]}')
    print(f'cv macro-f1: {best["cv_macro_f1_mean"]:.6f}')
    print(f'holdout accuracy: {hold_acc:.6f}')
    print(f'holdout macro-f1: {hold_macro_f1:.6f}')
    print(f'preds_holdout: {pred_path}')
    print(f'metrics: {metrics_path}')

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
