#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.resnet_presence import build_resnet_presence  # noqa: E402

DATA_ROOT_DEFAULT = Path('/hhome/ricse03/HelicoData')
HOLDOUT_ROOT_DEFAULT = Path('/hhome/ricse03/HelicoData/HoldOut')
OUTPUT_DIR_DEFAULT = Path('/hhome/ricse03/Deep_Learning_Group 3/outputs/patch_preds')


class InferenceDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        image_path = row['image_path']
        pat_id = row['Pat_ID']
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            x = self.transform(img)
        return x, pat_id, image_path


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_checkpoint(project_root: Path, checkpoint: Path = None):
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        return checkpoint

    report = project_root / 'outputs' / 'patch' / 'final_training_report.json'
    if report.exists():
        data = json.loads(report.read_text(encoding='utf-8'))
        best_run = data.get('best_run_recommendation')
        if best_run:
            ckpt = project_root / 'outputs' / 'patch' / best_run / 'best.ckpt'
            if ckpt.exists():
                return ckpt

    run_ckpts = []
    for p in (project_root / 'outputs' / 'patch').glob('run_*'):
        m = re.match(r'^run_(\d+)$', p.name)
        ckpt = p / 'best.ckpt'
        if m and ckpt.exists():
            run_ckpts.append((int(m.group(1)), ckpt))

    if not run_ckpts:
        raise FileNotFoundError('No best.ckpt found under outputs/patch/run_*/best.ckpt')

    run_ckpts.sort(key=lambda x: x[0])
    return run_ckpts[-1][1]


def load_model_and_transform(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('config', {})
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('train', {})

    backbone = model_cfg.get('backbone', 'resnet18')
    dropout = float(model_cfg.get('dropout', 0.0))
    image_size = int(train_cfg.get('image_size', 224))

    model = build_resnet_presence(backbone=backbone, pretrained=False, dropout=dropout)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, tfm, image_size


def scan_holdout(holdout_root: Path):
    if not holdout_root.exists():
        raise FileNotFoundError(f'HOLDOUT_ROOT not found: {holdout_root}')

    rows = []
    for p in holdout_root.rglob('*.png'):
        rel = p.relative_to(holdout_root)
        if len(rel.parts) < 1:
            continue
        patient_folder = rel.parts[0]
        pat_id = patient_folder.split('_')[0]
        rows.append({'Pat_ID': pat_id, 'image_path': str(p.resolve())})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f'No PNG files found under {holdout_root}')
    df = df.sort_values(['Pat_ID', 'image_path']).reset_index(drop=True)
    return df


def slice_holdout_patients(df: pd.DataFrame, offset: int, limit: int):
    uniq = sorted(df['Pat_ID'].astype(str).unique().tolist())
    if offset < 0:
        raise ValueError('--holdout-patient-offset must be >= 0')
    if offset >= len(uniq):
        return df.iloc[0:0].copy(), uniq
    if limit <= 0:
        keep = uniq[offset:]
    else:
        keep = uniq[offset : offset + limit]
    out = df[df['Pat_ID'].isin(set(keep))].copy().reset_index(drop=True)
    return out, uniq


def infer_dataframe(df: pd.DataFrame, model, transform, device, batch_size: int, num_workers: int):
    ds = InferenceDataset(df, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    out_pat_id = []
    out_path = []
    out_prob = []

    with torch.no_grad():
        for x, pat_id, image_path in dl:
            x = x.to(device)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            out_pat_id.extend(list(pat_id))
            out_path.extend(list(image_path))
            out_prob.extend([float(p) for p in probs])

    out = pd.DataFrame({'Pat_ID': out_pat_id, 'image_path': out_path, 'p_pos': out_prob})
    return out


def infer_annotated_splits_if_available(project_root: Path, output_dir: Path, model, transform, device, batch_size: int, num_workers: int):
    for split in ['train', 'val', 'test']:
        csv_path = project_root / 'manifests' / f'patch_{split}.csv'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        required = {'Pat_ID', 'image_path'}
        if not required.issubset(set(df.columns)):
            print(f'[skip] {csv_path} missing columns: {required - set(df.columns)}')
            continue

        inp = df[['Pat_ID', 'image_path']].copy()
        inp['Pat_ID'] = inp['Pat_ID'].astype(str).str.strip()
        inp['image_path'] = inp['image_path'].astype(str)

        pred = infer_dataframe(inp, model, transform, device, batch_size=batch_size, num_workers=num_workers)
        out_path = output_dir / f'patch_{split}_pred.csv'
        pred.to_csv(out_path, index=False)
        print(f'annotated {split}: patches={len(pred)}, patients={pred["Pat_ID"].nunique()}, file={out_path}')


def main():
    parser = argparse.ArgumentParser(description='Patch-level Presence inference for HoldOut (and optional annotated splits).')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument('--holdout-root', type=Path, default=HOLDOUT_ROOT_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--skip-annotated-splits', action='store_true')
    parser.add_argument('--holdout-patient-offset', type=int, default=0)
    parser.add_argument('--holdout-patient-limit', type=int, default=0)
    parser.add_argument('--holdout-output-name', type=str, default='holdout.csv')
    args = parser.parse_args()

    _ = args.data_root  # kept for interface parity with project paths.

    set_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ckpt_path = resolve_checkpoint(PROJECT_ROOT, args.checkpoint)
    model, transform, image_size = load_model_and_transform(ckpt_path, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    holdout_df = scan_holdout(args.holdout_root)
    holdout_df, all_holdout_patients = slice_holdout_patients(
        holdout_df,
        offset=int(args.holdout_patient_offset),
        limit=int(args.holdout_patient_limit),
    )
    if len(holdout_df) == 0:
        raise RuntimeError(
            f'No holdout patches selected with offset={args.holdout_patient_offset}, '
            f'limit={args.holdout_patient_limit}, total_patients={len(all_holdout_patients)}'
        )
    holdout_pred = infer_dataframe(
        holdout_df,
        model,
        transform,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    holdout_csv = args.output_dir / args.holdout_output_name
    holdout_pred.to_csv(holdout_csv, index=False)

    if len(holdout_pred) == 0:
        raise RuntimeError('holdout.csv is empty after inference.')

    prob_ok = bool(((holdout_pred['p_pos'] >= 0.0) & (holdout_pred['p_pos'] <= 1.0)).all())
    if not prob_ok:
        raise RuntimeError('p_pos contains values outside [0,1].')

    print(f'checkpoint: {ckpt_path}')
    print(f'device: {device}')
    print(f'image_size: {image_size}')
    print(f'holdout patches: {len(holdout_pred)}')
    print(f'holdout patients: {holdout_pred["Pat_ID"].nunique()}')
    print(
        f'holdout slice: offset={args.holdout_patient_offset}, '
        f'limit={args.holdout_patient_limit}, total_patients={len(all_holdout_patients)}'
    )
    print(f'holdout csv: {holdout_csv}')
    print(f'p_pos in [0,1]: {prob_ok}')

    if not args.skip_annotated_splits:
        infer_annotated_splits_if_available(
            PROJECT_ROOT,
            args.output_dir,
            model,
            transform,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
