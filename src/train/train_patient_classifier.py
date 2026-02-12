#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')

TRAIN_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'train_features.csv'
VAL_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'val_features.csv'
HOLDOUT_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features' / 'holdout_features.csv'
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient' / 'run_001'

CLASS_ORDER = ['NEGATIVA', 'BAIXA', 'ALTA']
FEATURE_COLS = [
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


def load_features(path: Path, require_label: bool):
    if not path.exists():
        raise FileNotFoundError(f'Feature CSV not found: {path}')

    df = pd.read_csv(path)

    required = {'Pat_ID'} | set(FEATURE_COLS)
    if require_label:
        required |= {'DENSITAT'}

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'{path} missing columns: {missing}')

    out = df.copy()
    out['Pat_ID'] = out['Pat_ID'].astype(str).str.strip()

    for c in FEATURE_COLS:
        out[c] = pd.to_numeric(out[c], errors='coerce')

    out = out.dropna(subset=FEATURE_COLS)
    out = out[out['Pat_ID'] != '']

    if require_label:
        out['DENSITAT'] = out['DENSITAT'].fillna('').astype(str).str.strip()
        out = out[out['DENSITAT'].isin(CLASS_ORDER)]

    out = out.reset_index(drop=True)
    return out


def plot_confusion_matrix(cm: np.ndarray, labels, out_path: Path):
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
        title='Validation Confusion Matrix',
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def ensure_class_prob_columns(prob: np.ndarray, classes_in_model, class_order):
    cls_to_idx = {c: i for i, c in enumerate(classes_in_model)}
    out = np.zeros((prob.shape[0], len(class_order)), dtype=float)
    for j, cls in enumerate(class_order):
        if cls in cls_to_idx:
            out[:, j] = prob[:, cls_to_idx[cls]]
    return out


def main():
    parser = argparse.ArgumentParser(description='Train patient-level 3-class classifier and predict HoldOut.')
    parser.add_argument('--train-feats', type=Path, default=TRAIN_FEATS_DEFAULT)
    parser.add_argument('--val-feats', type=Path, default=VAL_FEATS_DEFAULT)
    parser.add_argument('--holdout-feats', type=Path, default=HOLDOUT_FEATS_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_df = load_features(args.train_feats, require_label=True)
    val_df = load_features(args.val_feats, require_label=True)
    holdout_df = load_features(args.holdout_feats, require_label=False)

    if len(train_df) == 0:
        raise RuntimeError('train features are empty after filtering')
    if len(val_df) == 0:
        raise RuntimeError('val features are empty after filtering')
    if len(holdout_df) == 0:
        raise RuntimeError('holdout features are empty after filtering')

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df['DENSITAT'].to_numpy()

    X_val = val_df[FEATURE_COLS].to_numpy(dtype=float)
    y_val = val_df['DENSITAT'].to_numpy()

    X_holdout = holdout_df[FEATURE_COLS].to_numpy(dtype=float)

    clf = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            (
                'model',
                LogisticRegression(
                    multi_class='multinomial',
                    class_weight='balanced',
                    max_iter=5000,
                    random_state=args.seed,
                    solver='lbfgs',
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    y_val_prob = clf.predict_proba(X_val)

    val_acc = float(accuracy_score(y_val, y_val_pred))
    val_macro_f1 = float(f1_score(y_val, y_val_pred, average='macro'))

    cm = confusion_matrix(y_val, y_val_pred, labels=CLASS_ORDER)
    report = classification_report(y_val, y_val_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)

    holdout_prob = clf.predict_proba(X_holdout)
    holdout_pred = clf.predict(X_holdout)

    classes_in_model = list(clf.named_steps['model'].classes_)
    holdout_prob_full = ensure_class_prob_columns(holdout_prob, classes_in_model, CLASS_ORDER)

    pred_holdout_df = pd.DataFrame(
        {
            'Pat_ID': holdout_df['Pat_ID'].astype(str).to_numpy(),
            'pred_class': holdout_pred,
            'p_NEGATIVA': holdout_prob_full[:, 0],
            'p_BAIXA': holdout_prob_full[:, 1],
            'p_ALTA': holdout_prob_full[:, 2],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / 'model.pkl'
    scaler_path = args.output_dir / 'scaler.pkl'
    metrics_path = args.output_dir / 'metrics.json'
    cm_path = args.output_dir / 'confusion_matrix.png'
    holdout_pred_path = args.output_dir / 'preds_holdout.csv'

    joblib.dump(clf, model_path)
    joblib.dump(clf.named_steps['scaler'], scaler_path)
    pred_holdout_df.to_csv(holdout_pred_path, index=False)
    plot_confusion_matrix(cm, CLASS_ORDER, cm_path)

    payload = {
        'seed': args.seed,
        'class_order': CLASS_ORDER,
        'feature_cols': FEATURE_COLS,
        'train': {
            'num_patients': int(len(train_df)),
            'label_counts': train_df['DENSITAT'].value_counts().to_dict(),
        },
        'val': {
            'num_patients': int(len(val_df)),
            'label_counts': val_df['DENSITAT'].value_counts().to_dict(),
            'accuracy': val_acc,
            'macro_f1': val_macro_f1,
            'confusion_matrix': cm.tolist(),
            'per_class_report': {k: report[k] for k in CLASS_ORDER if k in report},
        },
        'holdout': {
            'num_patients': int(len(holdout_df)),
            'pred_class_counts': pred_holdout_df['pred_class'].value_counts().to_dict(),
        },
        'paths': {
            'model': str(model_path),
            'scaler': str(scaler_path),
            'metrics_json': str(metrics_path),
            'confusion_matrix_png': str(cm_path),
            'preds_holdout_csv': str(holdout_pred_path),
        },
    }

    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'val accuracy: {val_acc:.6f}')
    print(f'val macro_f1: {val_macro_f1:.6f}')
    print(f'val confusion_matrix: {cm.tolist()}')
    print(f'holdout patients: {len(holdout_df)}')
    print(f'holdout preds: {holdout_pred_path}')
    print(f'metrics: {metrics_path}')

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
