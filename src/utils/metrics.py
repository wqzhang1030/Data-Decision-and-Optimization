#!/usr/bin/env python3
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        'acc': _safe_float(accuracy_score(y_true, y_pred)),
        'f1': _safe_float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float('nan'),
        'pr_auc': float('nan'),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    if len(np.unique(y_true)) >= 2:
        out['roc_auc'] = _safe_float(roc_auc_score(y_true, y_prob))
        out['pr_auc'] = _safe_float(average_precision_score(y_true, y_prob))

    return out


def roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List[float]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return {'fpr': [0.0, 1.0], 'tpr': [0.0, 1.0], 'thresholds': [1.0, 0.0]}
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thr.tolist()}


def pr_points(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List[float]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    return {'precision': prec.tolist(), 'recall': rec.tolist(), 'thresholds': thr.tolist()}
