#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')

PATCH_METRICS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patch_matrix' / 'resnet18_s42' / 'run_001' / 'metrics.json'
PATIENT_METRICS_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_full' / 'resnet18_s42' / 'run_001' / 'metrics.json'
PATIENT_REPORT_XLSX_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_full' / 'resnet18_s42' / 'run_001' / 'patient_prediction_report.xlsx'
HEATMAP_INDEX_DEFAULT = PROJECT_ROOT / 'outputs' / 'patch_heatmaps' / 'run_001' / 'heatmaps_index.csv'
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / 'outputs' / 'paper' / 'run_001'
RESLUT_PAPER_DIR_DEFAULT = Path('/hhome/ricse03/reslut/paper')


def ensure_exists(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f'{name} not found: {path}')


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def to_relative_str(path: Path, base: Path):
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def build_main_table(patch_metrics: dict, patient_metrics: dict):
    patch = patch_metrics['metrics']
    hold = patient_metrics['holdout_eval']

    rows = []
    for split in ['val', 'test']:
        m = patch[split]
        rows.append(
            {
                'task': f'Patch Presence ({split})',
                'accuracy': m.get('acc'),
                'f1_or_macro_f1': m.get('f1'),
                'balanced_accuracy': None,
                'roc_auc': m.get('roc_auc'),
                'pr_auc': m.get('pr_auc'),
                'macro_precision': None,
                'macro_recall': None,
                'ece': None,
                'brier': None,
                'log_loss': None,
                'notes': 'Binary patch-level metric',
            }
        )

    rows.append(
        {
            'task': 'Patient 3-class (holdout)',
            'accuracy': hold.get('accuracy'),
            'f1_or_macro_f1': hold.get('macro_f1'),
            'balanced_accuracy': hold.get('balanced_accuracy'),
            'roc_auc': hold.get('macro_ovr_roc_auc'),
            'pr_auc': hold.get('macro_ovr_pr_auc'),
            'macro_precision': hold.get('macro_precision'),
            'macro_recall': hold.get('macro_recall'),
            'ece': hold.get('ece_toplabel'),
            'brier': hold.get('brier_score_multiclass'),
            'log_loss': hold.get('log_loss'),
            'notes': 'Multiclass patient-level metric',
        }
    )
    return pd.DataFrame(rows)


def build_per_class_table(patient_metrics: dict):
    hold = patient_metrics['holdout_eval']
    rep = hold.get('per_class_report', {})
    rows = []
    for cls in ['NEGATIVA', 'BAIXA', 'ALTA']:
        r = rep.get(cls, {})
        rows.append(
            {
                'class': cls,
                'precision': r.get('precision'),
                'recall': r.get('recall'),
                'f1': r.get('f1-score'),
                'support': r.get('support'),
                'ovr_roc_auc': hold.get('ovr_roc_auc', {}).get(cls),
                'ovr_pr_auc': hold.get('ovr_pr_auc', {}).get(cls),
            }
        )
    return pd.DataFrame(rows)


def build_error_table(report_xlsx: Path, top_k: int = 30):
    pred_df = pd.read_excel(report_xlsx, sheet_name='predictions_all')
    required = {'Pat_ID', 'true_label', 'pred_class', 'correct', 'confidence', 'p_NEGATIVA', 'p_BAIXA', 'p_ALTA'}
    missing = required - set(pred_df.columns)
    if missing:
        raise RuntimeError(f'predictions_all missing required columns: {missing}')

    err = pred_df[pred_df['correct'] == False].copy()  # noqa: E712
    err = err.sort_values(['confidence'], ascending=False).reset_index(drop=True)
    if top_k > 0:
        err = err.head(top_k).copy()

    err.insert(0, 'error_rank', range(1, len(err) + 1))
    return err


def build_figure_inventory(project_root: Path, heatmap_index_path: Path):
    fig_rows = []

    fixed_figures = [
        ('patch_confusion_matrix', project_root / 'outputs' / 'patch_matrix' / 'resnet18_s42' / 'run_001' / 'confusion_matrix.png', 'Patch', 'Patch-level confusion matrix'),
        ('patch_roc_curve', project_root / 'outputs' / 'patch_matrix' / 'resnet18_s42' / 'run_001' / 'roc_curve.png', 'Patch', 'Patch-level ROC curve'),
        ('patch_pr_curve', project_root / 'outputs' / 'patch_matrix' / 'resnet18_s42' / 'run_001' / 'pr_curve.png', 'Patch', 'Patch-level PR curve'),
        ('patient_confusion_matrix', project_root / 'outputs' / 'patient_full' / 'resnet18_s42' / 'run_001' / 'confusion_matrix.png', 'Patient', 'Patient-level confusion matrix'),
    ]
    for fig_id, path, category, caption in fixed_figures:
        if path.exists():
            fig_rows.append(
                {
                    'figure_id': fig_id,
                    'category': category,
                    'artifact_path': str(path),
                    'caption': caption,
                }
            )

    if heatmap_index_path.exists():
        hm = pd.read_csv(heatmap_index_path)
        if {'true_presence', 'heatmap_path'}.issubset(hm.columns):
            hm_pos = hm[hm['true_presence'] == 1].head(3)
            hm_neg = hm[hm['true_presence'] == -1].head(3)
            for i, (_, row) in enumerate(hm_pos.iterrows(), start=1):
                fig_rows.append(
                    {
                        'figure_id': f'heatmap_pos_{i}',
                        'category': 'Heatmap',
                        'artifact_path': str(row['heatmap_path']),
                        'caption': f'Patch Grad-CAM positive example #{i}',
                    }
                )
            for i, (_, row) in enumerate(hm_neg.iterrows(), start=1):
                fig_rows.append(
                    {
                        'figure_id': f'heatmap_neg_{i}',
                        'category': 'Heatmap',
                        'artifact_path': str(row['heatmap_path']),
                        'caption': f'Patch Grad-CAM negative example #{i}',
                    }
                )
    return pd.DataFrame(fig_rows)


def build_markdown_summary(main_df: pd.DataFrame, per_class_df: pd.DataFrame, out_md: Path):
    def df_to_md(df: pd.DataFrame):
        cols = list(df.columns)
        lines = []
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
        for _, row in df.iterrows():
            vals = [str(row[c]) for c in cols]
            lines.append('| ' + ' | '.join(vals) + ' |')
        return '\n'.join(lines)

    lines = []
    lines.append('# Paper Result Summary')
    lines.append('')
    lines.append('## Main Results')
    lines.append('')
    lines.append(df_to_md(main_df))
    lines.append('')
    lines.append('## Patient Per-Class Results')
    lines.append('')
    lines.append(df_to_md(per_class_df))
    lines.append('')
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def copy_to_reslut(output_dir: Path, reslut_dir: Path):
    if reslut_dir.exists():
        shutil.rmtree(reslut_dir)
    shutil.copytree(output_dir, reslut_dir)


def main():
    parser = argparse.ArgumentParser(description='Build paper-ready result tables/artifacts.')
    parser.add_argument('--patch-metrics', type=Path, default=PATCH_METRICS_DEFAULT)
    parser.add_argument('--patient-metrics', type=Path, default=PATIENT_METRICS_DEFAULT)
    parser.add_argument('--patient-report-xlsx', type=Path, default=PATIENT_REPORT_XLSX_DEFAULT)
    parser.add_argument('--heatmap-index', type=Path, default=HEATMAP_INDEX_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--reslut-paper-dir', type=Path, default=RESLUT_PAPER_DIR_DEFAULT)
    parser.add_argument('--top-errors', type=int, default=30)
    parser.add_argument('--skip-copy-to-reslut', action='store_true')
    args = parser.parse_args()

    ensure_exists(args.patch_metrics, 'patch metrics')
    ensure_exists(args.patient_metrics, 'patient metrics')
    ensure_exists(args.patient_report_xlsx, 'patient prediction report xlsx')

    patch_metrics = load_json(args.patch_metrics)
    patient_metrics = load_json(args.patient_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_df = build_main_table(patch_metrics, patient_metrics)
    per_class_df = build_per_class_table(patient_metrics)
    error_df = build_error_table(args.patient_report_xlsx, top_k=args.top_errors)
    fig_df = build_figure_inventory(PROJECT_ROOT, args.heatmap_index)

    main_csv = args.output_dir / 'paper_main_results.csv'
    per_class_csv = args.output_dir / 'paper_patient_per_class.csv'
    error_csv = args.output_dir / 'paper_error_cases_top.csv'
    fig_csv = args.output_dir / 'paper_figure_inventory.csv'
    xlsx_path = args.output_dir / 'paper_tables.xlsx'
    md_path = args.output_dir / 'paper_summary.md'
    manifest_path = args.output_dir / 'paper_artifacts_manifest.json'

    main_df.to_csv(main_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)
    error_df.to_csv(error_csv, index=False)
    fig_df.to_csv(fig_csv, index=False)

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        main_df.to_excel(writer, sheet_name='main_results', index=False)
        per_class_df.to_excel(writer, sheet_name='patient_per_class', index=False)
        error_df.to_excel(writer, sheet_name='error_cases_top', index=False)
        fig_df.to_excel(writer, sheet_name='figure_inventory', index=False)

    build_markdown_summary(main_df, per_class_df, md_path)

    manifest = {
        'patch_metrics': str(args.patch_metrics),
        'patient_metrics': str(args.patient_metrics),
        'patient_report_xlsx': str(args.patient_report_xlsx),
        'heatmap_index': str(args.heatmap_index),
        'artifacts': {
            'paper_main_results_csv': str(main_csv),
            'paper_patient_per_class_csv': str(per_class_csv),
            'paper_error_cases_top_csv': str(error_csv),
            'paper_figure_inventory_csv': str(fig_csv),
            'paper_tables_xlsx': str(xlsx_path),
            'paper_summary_md': str(md_path),
        },
        'counts': {
            'main_rows': int(len(main_df)),
            'per_class_rows': int(len(per_class_df)),
            'error_rows': int(len(error_df)),
            'figure_rows': int(len(fig_df)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    if not args.skip_copy_to_reslut:
        copy_to_reslut(args.output_dir, args.reslut_paper_dir)

    print(f'output_dir: {args.output_dir}')
    print(f'main_table: {main_csv}')
    print(f'per_class_table: {per_class_csv}')
    print(f'error_table: {error_csv}')
    print(f'figure_inventory: {fig_csv}')
    print(f'xlsx: {xlsx_path}')
    print(f'markdown: {md_path}')
    print(f'manifest: {manifest_path}')
    if not args.skip_copy_to_reslut:
        print(f'copied_to_reslut: {args.reslut_paper_dir}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
