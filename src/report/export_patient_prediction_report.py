#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
DEFAULT_PRED_CSV = PROJECT_ROOT / 'outputs' / 'patient_full' / 'resnet18_s42' / 'run_001' / 'preds_holdout.csv'
DEFAULT_LABELS_CSV = PROJECT_ROOT / 'manifests' / 'patient_labels.csv'
DEFAULT_METRICS_JSON = PROJECT_ROOT / 'outputs' / 'patient_full' / 'resnet18_s42' / 'run_001' / 'metrics.json'
DEFAULT_OUT_XLSX = PROJECT_ROOT / 'outputs' / 'patient_full' / 'resnet18_s42' / 'run_001' / 'patient_prediction_report.xlsx'

CLASS_ORDER = ['NEGATIVA', 'BAIXA', 'ALTA']


def main():
    parser = argparse.ArgumentParser(description='Export patient-level prediction report (xlsx) with truth/pred/correctness.')
    parser.add_argument('--pred-csv', type=Path, default=DEFAULT_PRED_CSV)
    parser.add_argument('--labels-csv', type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument('--metrics-json', type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument('--out-xlsx', type=Path, default=DEFAULT_OUT_XLSX)
    args = parser.parse_args()

    if not args.pred_csv.exists():
        raise FileNotFoundError(f'Prediction CSV not found: {args.pred_csv}')
    if not args.labels_csv.exists():
        raise FileNotFoundError(f'Labels CSV not found: {args.labels_csv}')

    pred = pd.read_csv(args.pred_csv)
    required_pred = {'Pat_ID', 'pred_class', 'p_NEGATIVA', 'p_BAIXA', 'p_ALTA'}
    missing_pred = required_pred - set(pred.columns)
    if missing_pred:
        raise RuntimeError(f'Prediction CSV missing columns: {missing_pred}')

    labels = pd.read_csv(args.labels_csv)
    required_labels = {'Pat_ID', 'DENSITAT'}
    missing_labels = required_labels - set(labels.columns)
    if missing_labels:
        raise RuntimeError(f'Labels CSV missing columns: {missing_labels}')

    pred = pred[['Pat_ID', 'pred_class', 'p_NEGATIVA', 'p_BAIXA', 'p_ALTA']].copy()
    pred['Pat_ID'] = pred['Pat_ID'].astype(str).str.strip()
    pred['pred_class'] = pred['pred_class'].astype(str).str.strip()
    for c in ['p_NEGATIVA', 'p_BAIXA', 'p_ALTA']:
        pred[c] = pd.to_numeric(pred[c], errors='coerce')

    labels = labels[['Pat_ID', 'DENSITAT']].copy()
    labels['Pat_ID'] = labels['Pat_ID'].astype(str).str.strip()
    labels['DENSITAT'] = labels['DENSITAT'].astype(str).str.strip()

    rep = pred.merge(labels, on='Pat_ID', how='left')
    rep['has_true_label'] = rep['DENSITAT'].isin(CLASS_ORDER)
    rep['correct'] = np.where(rep['has_true_label'], rep['pred_class'] == rep['DENSITAT'], np.nan)
    rep['correct'] = rep['correct'].astype('object')
    rep.loc[rep['correct'].notna(), 'correct'] = rep.loc[rep['correct'].notna(), 'correct'].astype(bool)
    rep['confidence'] = rep[['p_NEGATIVA', 'p_BAIXA', 'p_ALTA']].max(axis=1)
    rep['true_label'] = rep['DENSITAT']

    view_cols = [
        'Pat_ID',
        'true_label',
        'pred_class',
        'correct',
        'confidence',
        'p_NEGATIVA',
        'p_BAIXA',
        'p_ALTA',
    ]
    rep_out = rep[view_cols].copy()
    rep_out = rep_out.sort_values(['correct', 'confidence', 'Pat_ID'], ascending=[True, False, True], na_position='last')

    eval_df = rep[rep['has_true_label']].copy()
    if len(eval_df) > 0:
        acc = float(accuracy_score(eval_df['DENSITAT'], eval_df['pred_class']))
        macro_f1 = float(f1_score(eval_df['DENSITAT'], eval_df['pred_class'], average='macro'))
        cm = confusion_matrix(eval_df['DENSITAT'], eval_df['pred_class'], labels=CLASS_ORDER)
    else:
        acc = None
        macro_f1 = None
        cm = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)

    summary_rows = [
        {'item': 'pred_csv', 'value': str(args.pred_csv)},
        {'item': 'labels_csv', 'value': str(args.labels_csv)},
        {'item': 'out_xlsx', 'value': str(args.out_xlsx)},
        {'item': 'n_patients_predictions', 'value': int(len(pred))},
        {'item': 'n_patients_with_truth', 'value': int(len(eval_df))},
        {'item': 'accuracy_recomputed', 'value': acc},
        {'item': 'macro_f1_recomputed', 'value': macro_f1},
    ]

    if args.metrics_json.exists():
        try:
            m = json.loads(args.metrics_json.read_text(encoding='utf-8'))
            hold = m.get('holdout_eval', {})
            summary_rows.extend(
                [
                    {'item': 'metrics_json', 'value': str(args.metrics_json)},
                    {'item': 'accuracy_metrics_json', 'value': hold.get('accuracy')},
                    {'item': 'macro_f1_metrics_json', 'value': hold.get('macro_f1')},
                    {'item': 'balanced_accuracy_metrics_json', 'value': hold.get('balanced_accuracy')},
                    {'item': 'macro_ovr_roc_auc_metrics_json', 'value': hold.get('macro_ovr_roc_auc')},
                    {'item': 'macro_ovr_pr_auc_metrics_json', 'value': hold.get('macro_ovr_pr_auc')},
                    {'item': 'ece_metrics_json', 'value': hold.get('ece_toplabel')},
                ]
            )
        except Exception as exc:
            summary_rows.append({'item': 'metrics_json_parse_error', 'value': str(exc)})
    else:
        summary_rows.append({'item': 'metrics_json', 'value': f'not found: {args.metrics_json}'})

    summary_df = pd.DataFrame(summary_rows)
    cm_df = pd.DataFrame(cm, index=[f'true_{c}' for c in CLASS_ORDER], columns=[f'pred_{c}' for c in CLASS_ORDER])
    err_df = rep_out[rep_out['correct'] == False].copy()  # noqa: E712

    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx, engine='openpyxl') as writer:
        rep_out.to_excel(writer, sheet_name='predictions_all', index=False)
        err_df.to_excel(writer, sheet_name='errors_only', index=False)
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        cm_df.to_excel(writer, sheet_name='confusion_matrix')

    print(f'xlsx: {args.out_xlsx}')
    print(f'patients: total={len(pred)}, with_truth={len(eval_df)}, errors={len(err_df)}')
    print(f'acc={acc}, macro_f1={macro_f1}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
