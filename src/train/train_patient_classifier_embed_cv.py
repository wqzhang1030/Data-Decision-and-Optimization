#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
TRAIN_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features_embed' / 'train_features.csv'
VAL_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features_embed' / 'val_features.csv'
TEST_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features_embed' / 'test_features.csv'
HOLDOUT_FEATS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features_embed' / 'holdout_features.csv'
OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient'

CLASS_ORDER = ['NEGATIVA', 'BAIXA', 'ALTA']


def one_hot_from_labels(y, class_order):
    idx = {c: i for i, c in enumerate(class_order)}
    arr = np.zeros((len(y), len(class_order)), dtype=float)
    for i, label in enumerate(y):
        if label in idx:
            arr[i, idx[label]] = 1.0
    return arr


def expected_calibration_error_multiclass(y_true, y_pred, y_prob, n_bins: int = 10):
    if len(y_true) == 0:
        return None

    confidence = np.max(y_prob, axis=1)
    correctness = (np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(correctness[mask]))
        bin_conf = float(np.mean(confidence[mask]))
        ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_extended_multiclass_metrics(y_true, y_pred, y_prob, class_order):
    if len(y_true) == 0:
        return {
            'accuracy': None,
            'macro_f1': None,
            'macro_precision': None,
            'macro_recall': None,
            'balanced_accuracy': None,
            'confusion_matrix': np.zeros((len(class_order), len(class_order)), dtype=int).tolist(),
            'per_class_report': {},
            'ovr_roc_auc': {},
            'ovr_pr_auc': {},
            'macro_ovr_roc_auc': None,
            'macro_ovr_pr_auc': None,
            'brier_score_multiclass': None,
            'log_loss': None,
            'ece_toplabel': None,
            'evaluated_patients': 0,
        }

    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_prob_arr = np.clip(y_prob_arr, 1e-12, 1.0)
    row_sum = y_prob_arr.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    y_prob_arr = y_prob_arr / row_sum

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average='macro'))
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=class_order,
        average='macro',
        zero_division=0,
    )
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=class_order)
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )

    ovr_roc_auc = {}
    ovr_pr_auc = {}
    roc_vals = []
    pr_vals = []
    for j, cls in enumerate(class_order):
        y_bin = (y_true_arr == cls).astype(int)
        if y_bin.min() == y_bin.max():
            ovr_roc_auc[cls] = None
            ovr_pr_auc[cls] = None
            continue
        p = y_prob_arr[:, j]
        r_auc = float(roc_auc_score(y_bin, p))
        p_auc = float(average_precision_score(y_bin, p))
        ovr_roc_auc[cls] = r_auc
        ovr_pr_auc[cls] = p_auc
        roc_vals.append(r_auc)
        pr_vals.append(p_auc)

    macro_ovr_roc_auc = float(np.mean(roc_vals)) if roc_vals else None
    macro_ovr_pr_auc = float(np.mean(pr_vals)) if pr_vals else None

    y_onehot = one_hot_from_labels(y_true_arr, class_order)
    brier = float(np.mean(np.sum((y_onehot - y_prob_arr) ** 2, axis=1)))
    ll = float(log_loss(y_true_arr, y_prob_arr, labels=class_order))
    ece = expected_calibration_error_multiclass(y_true_arr, y_pred_arr, y_prob_arr, n_bins=10)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'balanced_accuracy': bal_acc,
        'confusion_matrix': cm.astype(int).tolist(),
        'per_class_report': report,
        'ovr_roc_auc': ovr_roc_auc,
        'ovr_pr_auc': ovr_pr_auc,
        'macro_ovr_roc_auc': macro_ovr_roc_auc,
        'macro_ovr_pr_auc': macro_ovr_pr_auc,
        'brier_score_multiclass': brier,
        'log_loss': ll,
        'ece_toplabel': ece,
        'evaluated_patients': int(len(y_true_arr)),
    }


def load_feat(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Feature file not found: {path}')
    df = pd.read_csv(path)
    if 'Pat_ID' not in df.columns:
        raise RuntimeError(f'{path} missing Pat_ID column')

    out = df.copy()
    out['Pat_ID'] = out['Pat_ID'].astype(str).str.strip()
    out = out[out['Pat_ID'] != ''].copy()
    out = out.drop_duplicates(subset=['Pat_ID'], keep='first')

    if 'DENSITAT' not in out.columns:
        out['DENSITAT'] = ''
    out['DENSITAT'] = out['DENSITAT'].astype(str).str.strip()

    numeric_cols = []
    for c in out.columns:
        if c in {'Pat_ID', 'DENSITAT'}:
            continue
        out[c] = pd.to_numeric(out[c], errors='coerce')
        numeric_cols.append(c)

    out = out.dropna(subset=numeric_cols)
    return out.reset_index(drop=True), numeric_cols


def get_feature_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in {'Pat_ID', 'DENSITAT'}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_run_dir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    run_ids = []
    for p in root.glob('run_*'):
        m = re.match(r'^run_(\d+)$', p.name)
        if m:
            run_ids.append(int(m.group(1)))
    nxt = 1 if not run_ids else max(run_ids) + 1
    out = root / f'run_{nxt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    return out


def build_candidates(seed: int, candidate_family: str):
    candidates = {}

    if candidate_family == 'all':
        for c in [0.1, 0.3, 1.0, 3.0]:
            candidates[f'lr_c{c}'] = Pipeline(
                steps=[
                    ('scaler', StandardScaler()),
                    (
                        'clf',
                        LogisticRegression(
                            multi_class='multinomial',
                            class_weight='balanced',
                            max_iter=10000,
                            C=float(c),
                            random_state=seed,
                        ),
                    ),
                ]
            )

        candidates['lr_pca64_c1'] = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=64, random_state=seed)),
                (
                    'clf',
                    LogisticRegression(
                        multi_class='multinomial',
                        class_weight='balanced',
                        max_iter=10000,
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )

        candidates['kbest50_lr'] = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('kbest', SelectKBest(score_func=f_classif, k=50)),
                (
                    'clf',
                    LogisticRegression(
                        multi_class='multinomial',
                        class_weight='balanced',
                        max_iter=10000,
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )

        candidates['kbest80_lr'] = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('kbest', SelectKBest(score_func=f_classif, k=80)),
                (
                    'clf',
                    LogisticRegression(
                        multi_class='multinomial',
                        class_weight='balanced',
                        max_iter=10000,
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )

        candidates['svm_rbf_c3'] = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('clf', SVC(C=3.0, gamma='scale', class_weight='balanced', probability=True, random_state=seed)),
            ]
        )

    candidates['rf_balanced'] = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=seed,
        n_jobs=-1,
    )

    candidates['et_balanced'] = ExtraTreesClassifier(
        n_estimators=800,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
    )

    # Heavier BAIXA weight to improve minority-class recall.
    baixa_boost = {'NEGATIVA': 1.0, 'BAIXA': 2.5, 'ALTA': 1.2}
    candidates['rf_baixa_boost'] = RandomForestClassifier(
        n_estimators=800,
        min_samples_leaf=2,
        class_weight=baixa_boost,
        random_state=seed,
        n_jobs=-1,
    )

    candidates['et_baixa_boost'] = ExtraTreesClassifier(
        n_estimators=900,
        min_samples_leaf=2,
        class_weight=baixa_boost,
        random_state=seed,
        n_jobs=-1,
    )

    return candidates


def run_cv_model_selection(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    candidate_family: str,
    baixa_prob_steps: int,
    margin_steps: int,
    tuning_objective: str,
    baixa_recall_weight: float,
    min_baixa_recall: float,
):
    class_counts = pd.Series(y).value_counts()
    min_count = int(class_counts.min())
    n_splits = max(2, min(5, min_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ranking = []
    for name, model in build_candidates(seed, candidate_family=candidate_family).items():
        oof_prob = oof_probabilities(model=model, X=X, y=y, cv=cv, class_order=CLASS_ORDER)

        argmax_pred = np.array(CLASS_ORDER, dtype=object)[np.argmax(oof_prob, axis=1)]
        oof_argmax_f1 = float(f1_score(y, argmax_pred, average='macro'))
        oof_argmax_acc = float(accuracy_score(y, argmax_pred))

        tuned = tune_baixa_gate(
            oof_prob=oof_prob,
            y_true=y,
            baixa_prob_steps=baixa_prob_steps,
            margin_steps=margin_steps,
            tuning_objective=tuning_objective,
            baixa_recall_weight=baixa_recall_weight,
            min_baixa_recall=min_baixa_recall,
        )
        tuned_pred = apply_baixa_gate(
            prob=oof_prob,
            baixa_prob_threshold=tuned['baixa_prob_threshold'],
            margin_threshold=tuned['margin_threshold'],
        )
        ranking.append(
            {
                'name': name,
                'oof_argmax_macro_f1': oof_argmax_f1,
                'oof_argmax_accuracy': oof_argmax_acc,
                'oof_tuned_macro_f1': float(tuned['macro_f1']),
                'oof_tuned_accuracy': float(tuned['accuracy']),
                'oof_tuned_baixa_recall': float(tuned['baixa_recall']),
                'oof_tuned_objective_score': float(tuned['objective_score']),
                'tuned_baixa_prob_threshold': float(tuned['baixa_prob_threshold']),
                'tuned_margin_threshold': float(tuned['margin_threshold']),
                'oof_tuned_confusion_matrix': confusion_matrix(y, tuned_pred, labels=CLASS_ORDER).astype(int).tolist(),
                'model': model,
            }
        )

    ranking = sorted(
        ranking,
        key=lambda x: (
            x['oof_tuned_objective_score'],
            x['oof_tuned_macro_f1'],
            x['oof_tuned_baixa_recall'],
            x['oof_tuned_accuracy'],
            x['oof_argmax_macro_f1'],
        ),
        reverse=True,
    )
    best = ranking[0]
    return best, ranking


def align_probabilities(prob: np.ndarray, model_classes, class_order):
    idx = {c: i for i, c in enumerate(model_classes)}
    out = np.zeros((prob.shape[0], len(class_order)), dtype=float)
    for j, c in enumerate(class_order):
        if c in idx:
            out[:, j] = prob[:, idx[c]]
    return out


def get_model_classes(model):
    if hasattr(model, 'classes_'):
        return list(model.classes_)
    if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
        return list(model.named_steps['clf'].classes_)
    raise RuntimeError('Unable to infer model classes for probability alignment.')


def oof_probabilities(model, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold, class_order):
    oof_prob = np.zeros((len(y), len(class_order)), dtype=float)
    for tr_idx, va_idx in cv.split(X, y):
        m = clone(model)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[va_idx])
        oof_prob[va_idx] = align_probabilities(p, get_model_classes(m), class_order)
    return oof_prob


def apply_baixa_gate(prob: np.ndarray, baixa_prob_threshold: float, margin_threshold: float):
    pred = np.array(CLASS_ORDER, dtype=object)[np.argmax(prob, axis=1)]
    margin = np.abs(prob[:, 0] - prob[:, 2])
    gate = (prob[:, 1] >= float(baixa_prob_threshold)) | (margin < float(margin_threshold))
    pred[gate] = 'BAIXA'
    return pred


def tune_baixa_gate(
    oof_prob: np.ndarray,
    y_true: np.ndarray,
    baixa_prob_steps: int,
    margin_steps: int,
    tuning_objective: str,
    baixa_recall_weight: float,
    min_baixa_recall: float,
):
    best = None
    baixa_prob_steps = max(3, int(baixa_prob_steps))
    margin_steps = max(3, int(margin_steps))
    y_true_arr = np.asarray(y_true)

    def objective(macro_f1: float, baixa_recall: float):
        if tuning_objective == 'macro_f1':
            return macro_f1
        if tuning_objective == 'macro_f1_baixa':
            return macro_f1 + baixa_recall_weight * baixa_recall
        if tuning_objective == 'baixa_recall':
            return baixa_recall + 0.1 * macro_f1
        raise ValueError(f'Unknown tuning objective: {tuning_objective}')

    all_candidates = []
    for baixa_prob_threshold in np.linspace(0.05, 0.70, baixa_prob_steps):
        for margin_threshold in np.linspace(0.01, 0.50, margin_steps):
            pred = apply_baixa_gate(
                prob=oof_prob,
                baixa_prob_threshold=float(baixa_prob_threshold),
                margin_threshold=float(margin_threshold),
            )
            f1 = float(f1_score(y_true, pred, average='macro'))
            acc = float(accuracy_score(y_true, pred))
            baixa_mask = y_true_arr == 'BAIXA'
            baixa_recall = float(np.mean(pred[baixa_mask] == 'BAIXA')) if np.any(baixa_mask) else 0.0
            score = float(objective(f1, baixa_recall))
            all_candidates.append((score, f1, acc, baixa_recall, float(baixa_prob_threshold), float(margin_threshold)))

            if baixa_recall < float(min_baixa_recall):
                continue
            if best is None:
                best = (score, f1, acc, baixa_recall, float(baixa_prob_threshold), float(margin_threshold))
                continue
            if (score > best[0]) or (np.isclose(score, best[0]) and f1 > best[1]) or (
                np.isclose(score, best[0]) and np.isclose(f1, best[1]) and acc > best[2]
            ):
                best = (score, f1, acc, baixa_recall, float(baixa_prob_threshold), float(margin_threshold))

    if best is None:
        all_candidates = sorted(all_candidates, key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best = all_candidates[0]

    return {
        'objective_score': best[0],
        'macro_f1': best[1],
        'accuracy': best[2],
        'baixa_recall': best[3],
        'baixa_prob_threshold': best[4],
        'margin_threshold': best[5],
    }


def save_confusion_matrix(cm: np.ndarray, labels, out_path: Path):
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
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train patient 3-class model on embedding features with CV model selection.')
    parser.add_argument('--train-feats', type=Path, default=TRAIN_FEATS_DEFAULT)
    parser.add_argument('--val-feats', type=Path, default=VAL_FEATS_DEFAULT)
    parser.add_argument('--test-feats', type=Path, default=TEST_FEATS_DEFAULT)
    parser.add_argument('--holdout-feats', type=Path, default=HOLDOUT_FEATS_DEFAULT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--candidate-family', type=str, choices=['all', 'trees'], default='trees')
    parser.add_argument('--baixa-prob-steps', type=int, default=33)
    parser.add_argument('--margin-steps', type=int, default=30)
    parser.add_argument(
        '--tuning-objective',
        type=str,
        choices=['macro_f1', 'macro_f1_baixa', 'baixa_recall'],
        default='macro_f1',
    )
    parser.add_argument('--baixa-recall-weight', type=float, default=0.5)
    parser.add_argument('--min-baixa-recall', type=float, default=0.0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_df, _ = load_feat(args.train_feats)
    val_df, _ = load_feat(args.val_feats)
    test_df, _ = load_feat(args.test_feats)
    hold_df, _ = load_feat(args.holdout_feats)

    ann = pd.concat([train_df, val_df, test_df], ignore_index=True)
    ann = ann.drop_duplicates(subset=['Pat_ID'], keep='first').copy()
    ann = ann[ann['DENSITAT'].isin(CLASS_ORDER)].copy()

    if len(ann) == 0:
        raise RuntimeError('No labeled annotated patients found in train/val/test features.')

    feature_cols = get_feature_cols(ann)
    if not feature_cols:
        raise RuntimeError('No numeric feature columns found.')

    # Keep holdout feature set exactly aligned to annotated training columns.
    missing_in_hold = [c for c in feature_cols if c not in hold_df.columns]
    if missing_in_hold:
        raise RuntimeError(f'HoldOut feature file missing columns: {missing_in_hold[:10]}')
    hold_df = hold_df[['Pat_ID', 'DENSITAT'] + feature_cols].copy()

    X_ann = ann[feature_cols].to_numpy(dtype=float)
    y_ann = ann['DENSITAT'].to_numpy()

    best, ranking = run_cv_model_selection(
        X_ann,
        y_ann,
        seed=args.seed,
        candidate_family=args.candidate_family,
        baixa_prob_steps=args.baixa_prob_steps,
        margin_steps=args.margin_steps,
        tuning_objective=args.tuning_objective,
        baixa_recall_weight=args.baixa_recall_weight,
        min_baixa_recall=args.min_baixa_recall,
    )
    model = best['model']
    model.fit(X_ann, y_ann)

    X_hold = hold_df[feature_cols].to_numpy(dtype=float)
    y_prob_raw = model.predict_proba(X_hold)
    y_prob = align_probabilities(y_prob_raw, get_model_classes(model), CLASS_ORDER)

    y_pred = apply_baixa_gate(
        prob=y_prob,
        baixa_prob_threshold=best['tuned_baixa_prob_threshold'],
        margin_threshold=best['tuned_margin_threshold'],
    )

    pred_df = pd.DataFrame(
        {
            'Pat_ID': hold_df['Pat_ID'].astype(str),
            'pred_class': y_pred,
            'p_NEGATIVA': y_prob[:, 0],
            'p_BAIXA': y_prob[:, 1],
            'p_ALTA': y_prob[:, 2],
        }
    )

    eval_df = hold_df[hold_df['DENSITAT'].isin(CLASS_ORDER)].copy()
    eval_join = pred_df.merge(eval_df[['Pat_ID', 'DENSITAT']], on='Pat_ID', how='inner')

    if len(eval_join) == 0:
        hold_metrics = compute_extended_multiclass_metrics(
            y_true=np.array([], dtype=object),
            y_pred=np.array([], dtype=object),
            y_prob=np.zeros((0, len(CLASS_ORDER)), dtype=float),
            class_order=CLASS_ORDER,
        )
    else:
        eval_prob = eval_join[['p_NEGATIVA', 'p_BAIXA', 'p_ALTA']].to_numpy(dtype=float)
        hold_metrics = compute_extended_multiclass_metrics(
            y_true=eval_join['DENSITAT'].to_numpy(dtype=object),
            y_pred=eval_join['pred_class'].to_numpy(dtype=object),
            y_prob=eval_prob,
            class_order=CLASS_ORDER,
        )

    hold_acc = hold_metrics['accuracy']
    hold_macro_f1 = hold_metrics['macro_f1']
    cm = np.array(hold_metrics['confusion_matrix'], dtype=int)

    out_dir = make_run_dir(args.output_root)
    pred_path = out_dir / 'preds_holdout.csv'
    model_path = out_dir / 'model.pkl'
    cm_path = out_dir / 'confusion_matrix.png'
    metrics_path = out_dir / 'metrics.json'

    pred_df.to_csv(pred_path, index=False)
    joblib.dump(model, model_path)
    save_confusion_matrix(cm, CLASS_ORDER, cm_path)

    metrics = {
        'timestamp': int(time.time()),
        'seed': int(args.seed),
        'selected_model': best['name'],
        'candidate_family': args.candidate_family,
        'annotated_patients': int(len(ann)),
        'holdout_patients': int(len(hold_df)),
        'n_features': int(len(feature_cols)),
        'cv_ranking': [
            {
                'name': r['name'],
                'oof_argmax_macro_f1': r['oof_argmax_macro_f1'],
                'oof_argmax_accuracy': r['oof_argmax_accuracy'],
                'oof_tuned_macro_f1': r['oof_tuned_macro_f1'],
                'oof_tuned_accuracy': r['oof_tuned_accuracy'],
                'oof_tuned_baixa_recall': r['oof_tuned_baixa_recall'],
                'oof_tuned_objective_score': r['oof_tuned_objective_score'],
                'tuned_baixa_prob_threshold': r['tuned_baixa_prob_threshold'],
                'tuned_margin_threshold': r['tuned_margin_threshold'],
                'oof_tuned_confusion_matrix': r['oof_tuned_confusion_matrix'],
            }
            for r in ranking
        ],
        'selected_model_tuning': {
            'tuned_baixa_prob_threshold': best['tuned_baixa_prob_threshold'],
            'tuned_margin_threshold': best['tuned_margin_threshold'],
            'oof_tuned_macro_f1': best['oof_tuned_macro_f1'],
            'oof_tuned_accuracy': best['oof_tuned_accuracy'],
            'oof_tuned_baixa_recall': best['oof_tuned_baixa_recall'],
            'oof_tuned_objective_score': best['oof_tuned_objective_score'],
            'tuning_objective': args.tuning_objective,
            'baixa_recall_weight': args.baixa_recall_weight,
            'min_baixa_recall': args.min_baixa_recall,
        },
        'holdout_eval': {
            **hold_metrics,
        },
        'artifacts': {
            'preds_holdout_csv': str(pred_path),
            'model_pkl': str(model_path),
            'confusion_matrix_png': str(cm_path),
        },
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print(f'out_dir: {out_dir}')
    print(f'selected_model: {best["name"]}')
    print(f'annotated_patients: {len(ann)}, holdout_patients: {len(hold_df)}, n_features: {len(feature_cols)}')
    print(f'holdout_accuracy: {hold_acc}, holdout_macro_f1: {hold_macro_f1}')
    print(f'predictions: {pred_path}')
    print(f'metrics: {metrics_path}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
