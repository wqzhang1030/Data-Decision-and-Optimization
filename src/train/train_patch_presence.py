#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.resnet_presence import build_resnet_presence  # noqa: E402
from utils.metrics import binary_classification_metrics, pr_points, roc_points  # noqa: E402


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        targets = targets.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t).clamp(min=1e-8) ** self.gamma) * bce
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


class PatchPresenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        y = int(row['y'])

        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.float32), path


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def maybe_limit_patients_in_split(df: pd.DataFrame, limit):
    if limit is None:
        return df
    uniq = sorted(df['Pat_ID'].astype(str).str.strip().unique().tolist())
    keep = set(uniq[: max(0, int(limit))])
    return df[df['Pat_ID'].isin(keep)].copy()


def prepare_split_df(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Split CSV not found: {path}')
    df = pd.read_csv(path)
    req = {'image_path', 'Pat_ID', 'Presence'}
    if not req.issubset(set(df.columns)):
        raise RuntimeError(f'{path} missing columns: {req - set(df.columns)}')

    df = df.copy()
    df['Pat_ID'] = df['Pat_ID'].astype(str).str.strip()
    df['Presence'] = pd.to_numeric(df['Presence'], errors='raise').astype(int)
    df = df[df['Presence'].isin([-1, 1])].copy()
    df['y'] = (df['Presence'] == 1).astype(int)

    missing_paths = [p for p in df['image_path'].tolist() if not Path(p).exists()]
    if missing_paths:
        raise RuntimeError(f'{path} has missing image_path. Sample: {missing_paths[:3]}')
    return df.reset_index(drop=True)


def run_dir(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in output_root.glob('run_*'):
        m = re.match(r'^run_(\d+)$', p.name)
        if m:
            runs.append(int(m.group(1)))
    nxt = 1 if not runs else max(runs) + 1
    d = output_root / f'run_{nxt:03d}'
    d.mkdir(parents=True, exist_ok=False)
    return d


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            prob = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            y_true.extend(y.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return {
        'loss': float(np.mean(losses)) if losses else float('nan'),
        'n_samples': int(len(y_true)),
        'y_true': y_true,
        'y_prob': y_prob,
    }


def select_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5, float('nan')

    candidates = np.unique(y_prob)
    if len(candidates) > 200:
        q = np.linspace(0.0, 1.0, num=200)
        candidates = np.quantile(y_prob, q=q)
        candidates = np.unique(candidates)

    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        m = binary_classification_metrics(y_true, y_prob, threshold=float(t))
        f1 = float(m['f1'])
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def plot_confusion(cm, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_roc(points, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(points['fpr'], points['tpr'], label='ROC')
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pr(points, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(points['recall'], points['precision'], label='PR')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train patch-level Presence classifier.')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'patch.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--limit_patients', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))

    seed = int(cfg.get('seed', 42) if args.seed is None else args.seed)
    set_seed(seed)

    paths = cfg['paths']
    train_cfg = cfg['train']
    metric_cfg = cfg.get('metrics', {})
    model_cfg = cfg['model']

    train_csv = PROJECT_ROOT / paths['patch_train_csv']
    val_csv = PROJECT_ROOT / paths['patch_val_csv']
    test_csv = PROJECT_ROOT / paths['patch_test_csv']
    output_root = PROJECT_ROOT / paths['output_root']

    train_df = prepare_split_df(train_csv)
    val_df = prepare_split_df(val_csv)
    test_df = prepare_split_df(test_csv)

    train_df = maybe_limit_patients_in_split(train_df, args.limit_patients)
    val_df = maybe_limit_patients_in_split(val_df, args.limit_patients)
    test_df = maybe_limit_patients_in_split(test_df, args.limit_patients)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f'Empty split after filtering. train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    image_size = int(train_cfg.get('image_size', 224))
    train_tf = [transforms.Resize((image_size, image_size))]
    if bool(train_cfg.get('train_flip', True)):
        train_tf.append(transforms.RandomHorizontalFlip(p=0.5))
    train_tf.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = PatchPresenceDataset(train_df, transform=transforms.Compose(train_tf))
    val_ds = PatchPresenceDataset(val_df, transform=eval_tf)
    test_ds = PatchPresenceDataset(test_df, transform=eval_tf)

    batch_size = int(train_cfg.get('batch_size', 32))
    num_workers = int(train_cfg.get('num_workers', 0))
    pin_memory = bool(train_cfg.get('pin_memory', True))
    g = torch.Generator()
    g.manual_seed(seed)

    train_sampler = None
    if bool(train_cfg.get('use_weighted_sampler', False)):
        y = train_df['y'].to_numpy(dtype=int)
        class_counts = np.bincount(y, minlength=2).astype(float)
        class_counts[class_counts == 0.0] = 1.0
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=g,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device if args.device != 'auto' else 'cpu')

    model = build_resnet_presence(
        backbone=model_cfg.get('backbone', 'resnet18'),
        pretrained=bool(model_cfg.get('pretrained', False)),
        dropout=float(model_cfg.get('dropout', 0.0)),
    ).to(device)

    loss_type = str(train_cfg.get('loss', 'bce')).lower()
    pos_weight_tensor = None
    if bool(train_cfg.get('use_pos_weight', True)):
        n_pos = int((train_df['y'] == 1).sum())
        n_neg = int((train_df['y'] == 0).sum())
        if n_pos > 0:
            pos_weight_tensor = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)

    if loss_type == 'focal':
        criterion = BinaryFocalLoss(
            gamma=float(train_cfg.get('focal_gamma', 2.0)),
            alpha=float(train_cfg.get('focal_alpha', 0.25)),
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get('lr', 3e-4)),
        weight_decay=float(train_cfg.get('weight_decay', 1e-4)),
    )
    scheduler = None
    scheduler_type = str(train_cfg.get('scheduler', 'none')).lower()
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(train_cfg.get('epochs', 8))),
            eta_min=float(train_cfg.get('lr_min', 1e-6)),
        )

    epochs = int(train_cfg.get('epochs', 8))
    default_threshold = float(metric_cfg.get('threshold', 0.5))

    out_dir = run_dir(output_root)
    best_ckpt = out_dir / 'best.ckpt'
    best_auc = -math.inf
    best_epoch = -1
    history = []
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_eval = evaluate(model, val_loader, criterion, device)
        val_metrics = binary_classification_metrics(val_eval['y_true'], val_eval['y_prob'], threshold=default_threshold)

        val_auc = val_metrics.get('roc_auc', float('nan'))
        score = -math.inf if (val_auc is None or np.isnan(val_auc)) else float(val_auc)
        if (score > best_auc) or (best_state is None):
            best_auc = score
            best_epoch = epoch
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'seed': seed,
                'val_metrics': val_metrics,
            }
            torch.save(best_state, best_ckpt)

        item = {
            'epoch': epoch,
            'train_loss': float(np.mean(train_losses)) if train_losses else float('nan'),
            'val_loss': float(val_eval['loss']),
            'val_roc_auc': float(val_metrics['roc_auc']) if not np.isnan(val_metrics['roc_auc']) else None,
            'val_pr_auc': float(val_metrics['pr_auc']) if not np.isnan(val_metrics['pr_auc']) else None,
            'val_acc': float(val_metrics['acc']),
            'val_f1': float(val_metrics['f1']),
        }
        history.append(item)
        print(
            f"epoch={epoch} train_loss={item['train_loss']:.4f} "
            f"val_loss={item['val_loss']:.4f} val_roc_auc={item['val_roc_auc']} val_pr_auc={item['val_pr_auc']}"
        )
        if scheduler is not None:
            scheduler.step()

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    train_eval = evaluate(model, train_loader, criterion, device)
    val_eval = evaluate(model, val_loader, criterion, device)
    test_eval = evaluate(model, test_loader, criterion, device)

    selected_threshold, selected_val_f1 = select_threshold_by_f1(val_eval['y_true'], val_eval['y_prob'])
    threshold = selected_threshold if not np.isnan(selected_threshold) else default_threshold

    train_metrics = binary_classification_metrics(train_eval['y_true'], train_eval['y_prob'], threshold=threshold)
    val_metrics = binary_classification_metrics(val_eval['y_true'], val_eval['y_prob'], threshold=threshold)
    test_metrics = binary_classification_metrics(test_eval['y_true'], test_eval['y_prob'], threshold=threshold)

    roc = roc_points(test_eval['y_true'], test_eval['y_prob'])
    pr = pr_points(test_eval['y_true'], test_eval['y_prob'])

    plot_confusion(test_metrics['confusion_matrix'], out_dir / 'confusion_matrix.png')
    plot_roc(roc, out_dir / 'roc_curve.png')
    plot_pr(pr, out_dir / 'pr_curve.png')

    payload = {
        'timestamp': int(time.time()),
        'seed': seed,
        'device': str(device),
        'output_dir': str(out_dir),
        'limit_patients': args.limit_patients,
        'best_epoch': best_epoch,
        'selection_metric': 'val_roc_auc',
        'threshold': {
            'default': default_threshold,
            'selected_from_val_f1': float(threshold),
            'selected_val_f1': float(selected_val_f1) if not np.isnan(selected_val_f1) else None,
        },
        'history': history,
        'split_sizes': {
            'train_patients': int(train_df['Pat_ID'].nunique()),
            'val_patients': int(val_df['Pat_ID'].nunique()),
            'test_patients': int(test_df['Pat_ID'].nunique()),
            'train_patches': len(train_df),
            'val_patches': len(val_df),
            'test_patches': len(test_df),
        },
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        },
    }

    with (out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'best.ckpt: {best_ckpt}')
    print(f'selected_threshold: {threshold:.6f}')
    print(f'metrics.json: {out_dir / "metrics.json"}')
    print(f'confusion_matrix.png: {out_dir / "confusion_matrix.png"}')
    print(f'roc_curve.png: {out_dir / "roc_curve.png"}')
    print(f'pr_curve.png: {out_dir / "pr_curve.png"}')

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
