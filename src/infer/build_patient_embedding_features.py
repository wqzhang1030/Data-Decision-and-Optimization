#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.resnet_presence import build_resnet_presence  # noqa: E402

HOLDOUT_ROOT_DEFAULT = Path('/hhome/ricse03/HelicoData/HoldOut')
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / 'outputs' / 'patient_features_embed'
PATIENT_LABELS_DEFAULT = PROJECT_ROOT / 'manifests' / 'patient_labels.csv'


class PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        pid = row['Pat_ID']
        with Image.open(path) as img:
            img = img.convert('RGB')
            x = self.transform(img)
        return x, pid


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


def load_model_feature_extractor(checkpoint_path: Path, device):
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

    # ResNet penultimate feature extractor.
    feat_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    classifier = model.fc

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    emb_dim = model.fc.in_features if hasattr(model.fc, 'in_features') else list(model.fc.children())[-1].in_features
    return feat_extractor, classifier, tfm, image_size, emb_dim


def scan_holdout(holdout_root: Path):
    rows = []
    for p in holdout_root.rglob('*.png'):
        rel = p.relative_to(holdout_root)
        patient_folder = rel.parts[0] if len(rel.parts) > 0 else p.parent.name
        pid = patient_folder.split('_')[0]
        rows.append({'Pat_ID': pid, 'image_path': str(p.resolve())})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f'No holdout PNG found in {holdout_root}')
    return df.sort_values(['Pat_ID', 'image_path']).reset_index(drop=True)


def load_patch_split(project_root: Path, split: str):
    p = project_root / 'manifests' / f'patch_{split}.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}')
    df = pd.read_csv(p)
    req = {'Pat_ID', 'image_path'}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f'{p} missing columns: {miss}')
    out = df[['Pat_ID', 'image_path']].copy()
    out['Pat_ID'] = out['Pat_ID'].astype(str).str.strip()
    out['image_path'] = out['image_path'].astype(str)
    out = out[out['Pat_ID'] != ''].reset_index(drop=True)
    return out


def aggregate_patient_features(
    df: pd.DataFrame,
    feat_extractor,
    classifier,
    transform,
    device,
    emb_dim: int,
    batch_size: int,
    num_workers: int,
    split_name: str,
    log_every: int,
):
    ds = PatchDataset(df, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    total_batches = len(dl)
    total_rows = len(df)

    print(f'[{split_name}] start: patches={total_rows}, batch_size={batch_size}, batches={total_batches}', flush=True)

    emb_sum = defaultdict(lambda: np.zeros(emb_dim, dtype=np.float64))
    emb_sumsq = defaultdict(lambda: np.zeros(emb_dim, dtype=np.float64))
    counts = defaultdict(int)
    p_values = defaultdict(list)

    with torch.no_grad():
        for bidx, (x, pid) in enumerate(dl, start=1):
            x = x.to(device)
            emb = feat_extractor(x)
            emb = torch.flatten(emb, 1)
            logits = classifier(emb).squeeze(1)
            p = torch.sigmoid(logits).cpu().numpy()
            e = emb.cpu().numpy()

            for i, patient_id in enumerate(pid):
                v = e[i].astype(np.float64)
                emb_sum[patient_id] += v
                emb_sumsq[patient_id] += v * v
                counts[patient_id] += 1
                p_values[patient_id].append(float(p[i]))

            if log_every > 0 and (bidx % log_every == 0 or bidx == total_batches):
                done = min(bidx * batch_size, total_rows)
                pct = 100.0 * done / max(1, total_rows)
                print(f'[{split_name}] progress: {done}/{total_rows} ({pct:.1f}%)', flush=True)

    rows = []
    for pid in sorted(counts.keys()):
        n = counts[pid]
        s = emb_sum[pid]
        ss = emb_sumsq[pid]
        mean = s / max(1, n)
        var = np.maximum(ss / max(1, n) - mean * mean, 0.0)
        std = np.sqrt(var)

        p = np.array(p_values[pid], dtype=float)
        ps = np.sort(p)[::-1]
        k = min(5, len(ps))

        row = {
            'Pat_ID': pid,
            'num_patches': int(n),
            'mean_p': float(np.mean(p)),
            'std_p': float(np.std(p, ddof=0)),
            'max_p': float(np.max(p)),
            'topk_mean_p': float(np.mean(ps[:k])),
            'pos_ratio_05': float(np.mean(p > 0.5)),
            'pos_ratio_07': float(np.mean(p > 0.7)),
            'q90': float(np.quantile(p, 0.90)),
            'q50': float(np.quantile(p, 0.50)),
        }

        for j in range(emb_dim):
            row[f'emb_mean_{j}'] = float(mean[j])
            row[f'emb_std_{j}'] = float(std[j])

        rows.append(row)

    out = pd.DataFrame(rows)
    print(f'[{split_name}] done: patients={len(out)}', flush=True)
    return out


def attach_labels(feat_df: pd.DataFrame, labels_path: Path):
    labels = pd.read_csv(labels_path)
    req = {'Pat_ID', 'DENSITAT'}
    if not req.issubset(labels.columns):
        raise RuntimeError(f'{labels_path} missing columns: {req - set(labels.columns)}')

    labels = labels[['Pat_ID', 'DENSITAT']].copy()
    labels['Pat_ID'] = labels['Pat_ID'].astype(str).str.strip()
    labels['DENSITAT'] = labels['DENSITAT'].astype(str).str.strip()

    out = feat_df.merge(labels, on='Pat_ID', how='left')
    out['DENSITAT'] = out['DENSITAT'].fillna('')
    return out


def main():
    parser = argparse.ArgumentParser(description='Build patient-level embedding features from Task1 checkpoint.')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--holdout-root', type=Path, default=HOLDOUT_ROOT_DEFAULT)
    parser.add_argument('--patient-labels', type=Path, default=PATIENT_LABELS_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ckpt = resolve_checkpoint(PROJECT_ROOT, args.checkpoint)
    feat_extractor, classifier, transform, image_size, emb_dim = load_model_feature_extractor(ckpt, device)

    train_df = load_patch_split(PROJECT_ROOT, 'train')
    val_df = load_patch_split(PROJECT_ROOT, 'val')
    test_df = load_patch_split(PROJECT_ROOT, 'test')
    holdout_df = scan_holdout(args.holdout_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_feat = aggregate_patient_features(
        train_df, feat_extractor, classifier, transform, device, emb_dim, args.batch_size, args.num_workers, 'train', args.log_every
    )
    val_feat = aggregate_patient_features(
        val_df, feat_extractor, classifier, transform, device, emb_dim, args.batch_size, args.num_workers, 'val', args.log_every
    )
    test_feat = aggregate_patient_features(
        test_df, feat_extractor, classifier, transform, device, emb_dim, args.batch_size, args.num_workers, 'test', args.log_every
    )
    hold_feat = aggregate_patient_features(
        holdout_df, feat_extractor, classifier, transform, device, emb_dim, args.batch_size, args.num_workers, 'holdout', args.log_every
    )

    train_feat = attach_labels(train_feat, args.patient_labels)
    val_feat = attach_labels(val_feat, args.patient_labels)
    test_feat = attach_labels(test_feat, args.patient_labels)
    hold_feat = attach_labels(hold_feat, args.patient_labels)

    train_path = args.output_dir / 'train_features.csv'
    val_path = args.output_dir / 'val_features.csv'
    test_path = args.output_dir / 'test_features.csv'
    hold_path = args.output_dir / 'holdout_features.csv'

    train_feat.to_csv(train_path, index=False)
    val_feat.to_csv(val_path, index=False)
    test_feat.to_csv(test_path, index=False)
    hold_feat.to_csv(hold_path, index=False)

    print(f'checkpoint: {ckpt}')
    print(f'device: {device}')
    print(f'image_size: {image_size}, emb_dim: {emb_dim}')
    print(f'train patients: {len(train_feat)}, val patients: {len(val_feat)}, test patients: {len(test_feat)}, holdout patients: {len(hold_feat)}')
    print(f'outputs: {train_path}, {val_path}, {test_path}, {hold_path}')

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
