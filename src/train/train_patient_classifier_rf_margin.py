#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
CLASS_ORDER = ['NEGATIVA', 'BAIXA', 'ALTA']

TRAIN_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'train_features.csv'
VAL_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'val_features.csv'
HOLDOUT_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'holdout_features.csv'
PATCH_TEST_PRED_DEFAULT = PROJECT_ROOT / 'outputs' / 'patch_preds' / 'patch_test_pred.csv'
PATIENT_LABELS_DEFAULT = PROJECT_ROOT / 'manifests' / 'patient_labels.csv'
OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient'

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
    out = df[['Pat_ID'] + BASE_FEATURES].copy()
    out['Pat_ID'] = out['Pat_ID'].astype(str).str.strip()
    for c in BASE_FEATURES:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=['Pat_ID'] + BASE_FEATURES)
    out = out[out['Pat_ID'] != '']
    return out


def aggregate_patch_pred(path: Path):
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
        k = min(5, len(s))
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


def load_labels(path: Path):
    d = pd.read_csv(path)
    req = {'Pat_ID', 'DENSITAT'}
    miss = req - set(d.columns)
    if miss:
        raise RuntimeError(f'{path} missing columns: {miss}')
    out = d[['Pat_ID', 'DENSITAT']].copy()
    out['Pat_ID'] = out['Pat_ID'].astype(str).str.strip()
    out['DENSITAT'] = out['DENSITAT'].astype(str).str.strip()
    out = out[out['DENSITAT'].isin(CLASS_ORDER)].drop_duplicates(subset=['Pat_ID'])
    return out


def make_run_dir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in root.glob('run_*'):
        m = re.match(r'^run_(\d+)$', p.name)
        if m:
            runs.append(int(m.group(1)))
    nxt = 1 if not runs else max(runs) + 1
    out = root / f'run_{nxt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    return out


def align_probs(prob, model_classes):
    idx = {c: i for i, c in enumerate(model_classes)}
    out = np.zeros((prob.shape[0], len(CLASS_ORDER)), dtype=float)
    for j, c in enumerate(CLASS_ORDER):
        if c in idx:
            out[:, j] = prob[:, idx[c]]
    return out


def apply_margin_rule(prob: np.ndarray, threshold: float):
    pred = np.array([CLASS_ORDER[i] for i in np.argmax(prob, axis=1)])
    margin = np.abs(prob[:, 0] - prob[:, 2])
    pred[margin < threshold] = 'BAIXA'
    return pred, margin


def plot_cm(cm: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_ORDER)),
        yticks=np.arange(len(CLASS_ORDER)),
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        ylabel='True label',
        xlabel='Predicted label',
        title='HoldOut Confusion Matrix',
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='RF patient classifier with OOF-tuned BAIXA margin rule.')
    parser.add_argument('--train-feats', type=Path, default=TRAIN_FEATS_DEFAULT)
    parser.add_argument('--val-feats', type=Path, default=VAL_FEATS_DEFAULT)
    parser.add_argument('--holdout-feats', type=Path, default=HOLDOUT_FEATS_DEFAULT)
    parser.add_argument('--patch-test-pred', type=Path, default=PATCH_TEST_PRED_DEFAULT)
    parser.add_argument('--patient-labels', type=Path, default=PATIENT_LABELS_DEFAULT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--min-samples-leaf', type=int, default=3)
    args = parser.parse_args()

    np.random.seed(args.seed)

    labels = load_labels(args.patient_labels)
    tr = load_feat_csv(args.train_feats)
    va = load_feat_csv(args.val_feats)
    te = aggregate_patch_pred(args.patch_test_pred)

    ann = pd.concat([tr, va, te], ignore_index=True).drop_duplicates(subset=['Pat_ID'])
    ann = ann.merge(labels, on='Pat_ID', how='inner')
    ann = add_derived_features(ann)

    hold = load_feat_csv(args.holdout_feats)
    hold = add_derived_features(hold)
    hold = hold.merge(labels, on='Pat_ID', how='left')
    hold = hold[hold['DENSITAT'].isin(CLASS_ORDER)].copy()

    feat_cols = [c for c in ann.columns if c not in {'Pat_ID', 'DENSITAT'}]
    X = ann[feat_cols].to_numpy(dtype=float)
    y = ann['DENSITAT'].to_numpy()
    Xh = hold[feat_cols].to_numpy(dtype=float)
    yh = hold['DENSITAT'].to_numpy()

    model = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        class_weight='balanced',
        random_state=args.seed,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_macro_f1 = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring='f1_macro')))
    cv_acc = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring='accuracy')))

    oof_prob = np.zeros((len(y), len(CLASS_ORDER)), dtype=float)
    for tr_idx, va_idx in cv.split(X, y):
        m = RandomForestClassifier(
            n_estimators=int(args.n_estimators),
            min_samples_leaf=int(args.min_samples_leaf),
            class_weight='balanced',
            random_state=args.seed,
        )
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[va_idx])
        oof_prob[va_idx] = align_probs(p, list(m.classes_))

    best = (-1.0, 0.0, None, None)
    for t in np.linspace(0.01, 0.30, 60):
        pred, _ = apply_margin_rule(oof_prob, threshold=float(t))
        f1 = float(f1_score(y, pred, average='macro'))
        acc = float(accuracy_score(y, pred))
        score = 0.5 * f1 + 0.5 * acc
        if score > best[0]:
            best = (score, float(t), f1, acc)

    best_margin = best[1]

    model.fit(X, y)
    hold_prob_raw = model.predict_proba(Xh)
    hold_prob = align_probs(hold_prob_raw, list(model.classes_))
    hold_pred, hold_margin = apply_margin_rule(hold_prob, threshold=best_margin)

    hold_acc = float(accuracy_score(yh, hold_pred))
    hold_macro_f1 = float(f1_score(yh, hold_pred, average='macro'))
    cm = confusion_matrix(yh, hold_pred, labels=CLASS_ORDER)
    report = classification_report(yh, hold_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)

    out_dir = make_run_dir(args.output_root)
    pred_path = out_dir / 'preds_holdout.csv'
    cm_path = out_dir / 'confusion_matrix.png'
    metrics_path = out_dir / 'metrics.json'

    pred_df = pd.DataFrame(
        {
            'Pat_ID': hold['Pat_ID'].astype(str).to_numpy(),
            'pred_class': hold_pred,
            'true_class': hold['DENSITAT'].astype(str).to_numpy(),
            'p_NEGATIVA': hold_prob[:, 0],
            'p_BAIXA': hold_prob[:, 1],
            'p_ALTA': hold_prob[:, 2],
            'neg_alta_margin': hold_margin,
        }
    )
    pred_df.to_csv(pred_path, index=False)
    plot_cm(cm.astype(int), cm_path)

    metrics = {
        'timestamp': int(time.time()),
        'seed': int(args.seed),
        'model': {
            'name': 'RandomForestClassifier',
            'n_estimators': int(args.n_estimators),
            'min_samples_leaf': int(args.min_samples_leaf),
            'class_weight': 'balanced',
        },
        'feature_cols': feat_cols,
        'annotated_training': {
            'patients': int(len(ann)),
            'class_counts': ann['DENSITAT'].value_counts().to_dict(),
            'cv_macro_f1': cv_macro_f1,
            'cv_accuracy': cv_acc,
            'oof_margin_tuning': {
                'metric': '0.5*macro_f1 + 0.5*accuracy',
                'best_margin_threshold': float(best_margin),
                'best_oof_macro_f1': float(best[2]),
                'best_oof_accuracy': float(best[3]),
            },
        },
        'holdout_evaluation': {
            'patients': int(len(hold)),
            'accuracy': hold_acc,
            'macro_f1': hold_macro_f1,
            'confusion_matrix': cm.astype(int).tolist(),
            'classification_report': report,
        },
        'paths': {
            'metrics': str(metrics_path),
            'preds_holdout': str(pred_path),
            'confusion_matrix': str(cm_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print(f'output_dir: {out_dir}')
    print(f'cv_macro_f1: {cv_macro_f1:.4f}, cv_acc: {cv_acc:.4f}')
    print(f'best_margin_threshold: {best_margin:.6f} (oof_f1={best[2]:.4f}, oof_acc={best[3]:.4f})')
    print(f'holdout_accuracy: {hold_acc:.4f}, holdout_macro_f1: {hold_macro_f1:.4f}')
    print(f'metrics: {metrics_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
