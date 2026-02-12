#!/usr/bin/env python3
import argparse
import json
import os
import random
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.resnet_presence import build_resnet_presence  # noqa: E402

CLASS_ORDER = ['NEGATIVA', 'BAIXA', 'ALTA']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


class PathImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            x = self.transform(img)
        return x, path


class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25, n_classes: int = 3):
        super().__init__()
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.attn_u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, bag_x: torch.Tensor):
        h = self.patch_encoder(bag_x)
        a = torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))
        a = self.attn_w(a).squeeze(1)
        w = torch.softmax(a, dim=0).unsqueeze(1)
        pooled_attn = torch.sum(w * h, dim=0)
        pooled_mean = torch.mean(h, dim=0)
        pooled_max = torch.max(h, dim=0).values
        pooled = torch.cat([pooled_attn, pooled_mean, pooled_max], dim=0)
        logits = self.classifier(pooled)
        return logits


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_run_dir(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in output_root.glob('run_*'):
        m = re.match(r'^run_(\d+)$', p.name)
        if m:
            runs.append(int(m.group(1)))
    nxt = 1 if not runs else max(runs) + 1
    out_dir = output_root / f'run_{nxt:03d}'
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def resolve_patch_checkpoint(project_root: Path, configured_path: str):
    if configured_path:
        p = project_root / configured_path
        if p.exists():
            return p
        abs_p = Path(configured_path)
        if abs_p.exists():
            return abs_p
        raise FileNotFoundError(f'Configured patch checkpoint not found: {configured_path}')

    report = project_root / 'outputs' / 'patch' / 'final_training_report.json'
    if report.exists():
        obj = json.loads(report.read_text(encoding='utf-8'))
        best_run = obj.get('best_run_recommendation')
        if best_run:
            p = project_root / 'outputs' / 'patch' / best_run / 'best.ckpt'
            if p.exists():
                return p

    candidates = []
    for p in (project_root / 'outputs' / 'patch').glob('run_*'):
        m = re.match(r'^run_(\d+)$', p.name)
        ckpt = p / 'best.ckpt'
        if m and ckpt.exists():
            candidates.append((int(m.group(1)), ckpt))
    if not candidates:
        raise FileNotFoundError('No patch checkpoint found.')
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_patch_feature_extractor(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('train', {})

    backbone = str(model_cfg.get('backbone', 'resnet18'))
    dropout = float(model_cfg.get('dropout', 0.0))
    image_size = int(train_cfg.get('image_size', 224))

    model = build_resnet_presence(backbone=backbone, pretrained=False, dropout=dropout)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    if isinstance(model.fc, nn.Sequential):
        emb_dim = int(model.fc[-1].in_features)
    else:
        emb_dim = int(model.fc.in_features)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return extractor, transform, emb_dim


def read_patch_pred_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Missing patch prediction CSV: {path}')
    df = pd.read_csv(path)
    req = {'Pat_ID', 'image_path', 'p_pos'}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f'{path} missing columns: {miss}')
    df = df[['Pat_ID', 'image_path', 'p_pos']].copy()
    df['Pat_ID'] = df['Pat_ID'].astype(str).str.strip()
    df['image_path'] = df['image_path'].astype(str)
    df['p_pos'] = pd.to_numeric(df['p_pos'], errors='coerce')
    df = df.dropna(subset=['Pat_ID', 'image_path', 'p_pos'])
    df = df[df['Pat_ID'] != '']
    return df.reset_index(drop=True)


def read_patient_labels(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Missing patient labels CSV: {path}')
    df = pd.read_csv(path)
    req = {'Pat_ID', 'DENSITAT'}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f'{path} missing columns: {miss}')
    df = df[['Pat_ID', 'DENSITAT']].copy()
    df['Pat_ID'] = df['Pat_ID'].astype(str).str.strip()
    df['DENSITAT'] = df['DENSITAT'].astype(str).str.strip()
    df = df[df['DENSITAT'].isin(CLASS_ORDER)].drop_duplicates(subset=['Pat_ID'])
    return df.reset_index(drop=True)


def build_patient_groups(df: pd.DataFrame, label_map=None, require_label=True):
    groups = {}
    for pid, g in df.groupby('Pat_ID', sort=True):
        if require_label:
            if label_map is None or pid not in label_map:
                continue
            y = int(label_map[pid])
        else:
            y = None if label_map is None or pid not in label_map else int(label_map[pid])

        g2 = g.sort_values('image_path').reset_index(drop=True)
        groups[pid] = {
            'paths': g2['image_path'].tolist(),
            'p_pos': g2['p_pos'].to_numpy(dtype=np.float32),
            'y': y,
        }
    return groups


def extract_embeddings_for_paths(paths, feature_extractor, transform, device, batch_size: int, num_workers: int, seed: int):
    if len(paths) == 0:
        raise RuntimeError('Cannot extract embeddings from empty path list.')
    ds = PathImageDataset(paths, transform)
    g = torch.Generator()
    g.manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    out = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            e = feature_extractor(x)
            e = torch.flatten(e, 1).cpu().numpy().astype(np.float32)
            out.append(e)
    return np.concatenate(out, axis=0)


def build_bags_for_groups(
    groups,
    feature_extractor,
    transform,
    device,
    feature_batch_size: int,
    feature_num_workers: int,
    seed: int,
    log_prefix: str,
):
    bags = {}
    patient_ids = sorted(groups.keys())
    n_patients = len(patient_ids)
    for i, pid in enumerate(patient_ids, start=1):
        g = groups[pid]
        emb = extract_embeddings_for_paths(
            g['paths'],
            feature_extractor=feature_extractor,
            transform=transform,
            device=device,
            batch_size=feature_batch_size,
            num_workers=feature_num_workers,
            seed=seed,
        )
        p = g['p_pos'].reshape(-1, 1).astype(np.float32)
        bag_x = np.concatenate([emb, p], axis=1).astype(np.float32)
        bags[pid] = {'x': bag_x, 'y': g['y'], 'n_patches': int(len(g['paths']))}
        if i % 20 == 0 or i == n_patients:
            print(f'{log_prefix} bags: {i}/{n_patients}', flush=True)
    return bags


def class_weights_from_train(train_bags):
    y = np.array([v['y'] for v in train_bags.values()], dtype=int)
    counts = np.bincount(y, minlength=len(CLASS_ORDER)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    w = float(len(y)) / (len(CLASS_ORDER) * counts)
    return w.astype(np.float32), counts.astype(int)


def sample_probs_from_train(train_bags, oversample_power: float):
    items = sorted(train_bags.items(), key=lambda x: x[0])
    pids = [k for k, _ in items]
    y = np.array([v['y'] for _, v in items], dtype=int)
    counts = np.bincount(y, minlength=len(CLASS_ORDER)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = (1.0 / counts[y]) ** float(oversample_power)
    probs = inv / np.sum(inv)
    return pids, probs


def maybe_subsample_bag(bag_x: np.ndarray, max_patches: int, rng: np.random.Generator):
    if max_patches <= 0 or len(bag_x) <= max_patches:
        return bag_x
    idx = rng.choice(len(bag_x), size=max_patches, replace=False)
    idx = np.sort(idx)
    return bag_x[idx]


def evaluate_split(model, bags, patient_ids, device, max_patches_eval: int = 0):
    model.eval()
    y_true, y_pred, y_prob, out_rows = [], [], [], []
    with torch.no_grad():
        for pid in sorted(patient_ids):
            bag = bags[pid]
            x_np = bag['x']
            if max_patches_eval > 0 and len(x_np) > max_patches_eval:
                x_np = x_np[:max_patches_eval]
            x = torch.from_numpy(x_np).to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=0).cpu().numpy().astype(float)
            pred = int(np.argmax(prob))
            y = int(bag['y'])
            y_true.append(y)
            y_pred.append(pred)
            y_prob.append(prob.tolist())
            out_rows.append(
                {
                    'Pat_ID': pid,
                    'true_label': CLASS_ORDER[y],
                    'pred_label': CLASS_ORDER[pred],
                    'p_NEGATIVA': prob[0],
                    'p_BAIXA': prob[1],
                    'p_ALTA': prob[2],
                }
            )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'confusion_matrix': cm.astype(int).tolist(),
        'classification_report': report,
        'n_patients': int(len(patient_ids)),
        'pred_rows': out_rows,
    }


def infer_holdout(
    model,
    holdout_groups,
    feature_extractor,
    transform,
    device,
    feature_batch_size: int,
    feature_num_workers: int,
    seed: int,
    label_map,
    max_patches_eval: int = 0,
):
    model.eval()
    rows = []
    y_true, y_pred = [], []
    patient_ids = sorted(holdout_groups.keys())
    n_patients = len(patient_ids)
    with torch.no_grad():
        for i, pid in enumerate(patient_ids, start=1):
            g = holdout_groups[pid]
            emb = extract_embeddings_for_paths(
                g['paths'],
                feature_extractor=feature_extractor,
                transform=transform,
                device=device,
                batch_size=feature_batch_size,
                num_workers=feature_num_workers,
                seed=seed,
            )
            p = g['p_pos'].reshape(-1, 1).astype(np.float32)
            x_np = np.concatenate([emb, p], axis=1).astype(np.float32)
            if max_patches_eval > 0 and len(x_np) > max_patches_eval:
                x_np = x_np[:max_patches_eval]
            x = torch.from_numpy(x_np).to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=0).cpu().numpy().astype(float)
            pred = int(np.argmax(prob))

            rec = {
                'Pat_ID': pid,
                'pred_class': CLASS_ORDER[pred],
                'p_NEGATIVA': prob[0],
                'p_BAIXA': prob[1],
                'p_ALTA': prob[2],
                'num_patches': int(len(g['paths'])),
            }
            if label_map is not None and pid in label_map:
                true_y = int(label_map[pid])
                rec['true_class'] = CLASS_ORDER[true_y]
                y_true.append(true_y)
                y_pred.append(pred)
            rows.append(rec)

            if i % 10 == 0 or i == n_patients:
                print(f'holdout inference: {i}/{n_patients}', flush=True)

    eval_metrics = None
    if len(y_true) > 0:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
        eval_metrics = {
            'accuracy': float(accuracy_score(yt, yp)),
            'macro_f1': float(f1_score(yt, yp, average='macro')),
            'confusion_matrix': cm.astype(int).tolist(),
            'classification_report': classification_report(
                yt,
                yp,
                labels=[0, 1, 2],
                target_names=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            ),
            'n_patients': int(len(yt)),
        }
    return rows, eval_metrics


def plot_confusion(cm: np.ndarray, out_path: Path, title: str):
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
        title=title,
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
    parser = argparse.ArgumentParser(description='Train patient-level MIL classifier for DENSITAT.')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'patient_mil.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--skip-holdout', action='store_true')
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    seed = int(cfg.get('seed', 42) if args.seed is None else args.seed)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    paths_cfg = cfg['paths']
    train_cfg = cfg['train']
    model_cfg = cfg['model']

    train_pred_path = PROJECT_ROOT / paths_cfg['patch_train_pred_csv']
    val_pred_path = PROJECT_ROOT / paths_cfg['patch_val_pred_csv']
    test_pred_path = PROJECT_ROOT / paths_cfg['patch_test_pred_csv']
    holdout_pred_path = PROJECT_ROOT / paths_cfg['holdout_pred_csv']
    labels_path = PROJECT_ROOT / paths_cfg['patient_labels_csv']
    out_root = PROJECT_ROOT / paths_cfg['output_root']

    out_dir = make_run_dir(out_root)
    print(f'output_dir: {out_dir}')

    patch_ckpt = resolve_patch_checkpoint(PROJECT_ROOT, paths_cfg.get('patch_checkpoint', ''))

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    feature_extractor, transform, emb_dim = load_patch_feature_extractor(patch_ckpt, device)

    train_df = read_patch_pred_csv(train_pred_path)
    val_df = read_patch_pred_csv(val_pred_path)
    test_df = read_patch_pred_csv(test_pred_path)
    holdout_df = read_patch_pred_csv(holdout_pred_path)
    labels_df = read_patient_labels(labels_path)

    label_map = {r['Pat_ID']: CLASS_TO_IDX[r['DENSITAT']] for _, r in labels_df.iterrows()}

    train_groups = build_patient_groups(train_df, label_map=label_map, require_label=True)
    val_groups = build_patient_groups(val_df, label_map=label_map, require_label=True)
    test_groups = build_patient_groups(test_df, label_map=label_map, require_label=True)
    holdout_groups = build_patient_groups(holdout_df, label_map=label_map, require_label=False)

    feature_batch_size = int(train_cfg.get('feature_batch_size', 256))
    feature_num_workers = int(train_cfg.get('feature_num_workers', 0))

    print('building train bags...', flush=True)
    train_bags = build_bags_for_groups(
        train_groups,
        feature_extractor=feature_extractor,
        transform=transform,
        device=device,
        feature_batch_size=feature_batch_size,
        feature_num_workers=feature_num_workers,
        seed=seed,
        log_prefix='train',
    )
    print('building val bags...', flush=True)
    val_bags = build_bags_for_groups(
        val_groups,
        feature_extractor=feature_extractor,
        transform=transform,
        device=device,
        feature_batch_size=feature_batch_size,
        feature_num_workers=feature_num_workers,
        seed=seed,
        log_prefix='val',
    )
    print('building test bags...', flush=True)
    test_bags = build_bags_for_groups(
        test_groups,
        feature_extractor=feature_extractor,
        transform=transform,
        device=device,
        feature_batch_size=feature_batch_size,
        feature_num_workers=feature_num_workers,
        seed=seed,
        log_prefix='test',
    )

    input_dim = emb_dim + 1
    model = GatedAttentionMIL(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get('hidden_dim', 256)),
        dropout=float(model_cfg.get('dropout', 0.25)),
        n_classes=len(CLASS_ORDER),
    ).to(device)

    class_weights, class_counts = class_weights_from_train(train_bags)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get('lr', 8e-4)),
        weight_decay=float(train_cfg.get('weight_decay', 1e-4)),
    )

    train_patient_ids, train_probs = sample_probs_from_train(train_bags, oversample_power=float(train_cfg.get('oversample_power', 1.0)))
    val_patient_ids = sorted(val_bags.keys())
    test_patient_ids = sorted(test_bags.keys())

    epochs = int(train_cfg.get('epochs', 120) if args.epochs is None else args.epochs)
    patience = int(train_cfg.get('patience', 25) if args.patience is None else args.patience)
    steps_per_epoch = int(train_cfg.get('steps_per_epoch', 500) if args.steps_per_epoch is None else args.steps_per_epoch)
    max_patches_train = int(train_cfg.get('max_patches_train', 256))
    max_patches_eval = int(train_cfg.get('max_patches_eval', 0))
    grad_clip = float(train_cfg.get('grad_clip_norm', 5.0))

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = out_dir / 'best.ckpt'
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for _ in range(steps_per_epoch):
            pid = rng.choice(train_patient_ids, p=train_probs)
            bag = train_bags[pid]
            x_np = maybe_subsample_bag(bag['x'], max_patches=max_patches_train, rng=rng)
            y = int(bag['y'])

            x = torch.from_numpy(x_np).to(device)
            yt = torch.tensor([y], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).unsqueeze(0)
            loss = criterion(logits, yt)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate_split(model, val_bags, val_patient_ids, device=device, max_patches_eval=max_patches_eval)
        train_metrics = evaluate_split(model, train_bags, sorted(train_bags.keys()), device=device, max_patches_eval=max_patches_eval)

        row = {
            'epoch': epoch,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else None,
            'train_acc': train_metrics['accuracy'],
            'train_macro_f1': train_metrics['macro_f1'],
            'val_acc': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | loss={row['train_loss']:.4f} | "
            f"train_f1={row['train_macro_f1']:.4f} | val_f1={row['val_macro_f1']:.4f} | val_acc={row['val_acc']:.4f}",
            flush=True,
        )

        if row['val_macro_f1'] > best_val_f1 + 1e-8:
            best_val_f1 = row['val_macro_f1']
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    'epoch': epoch,
                    'seed': seed,
                    'model_state_dict': model.state_dict(),
                    'config': cfg,
                    'class_order': CLASS_ORDER,
                    'input_dim': input_dim,
                    'hidden_dim': int(model_cfg.get('hidden_dim', 256)),
                    'dropout': float(model_cfg.get('dropout', 0.25)),
                    'patch_checkpoint': str(patch_ckpt),
                },
                best_ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'early stopping at epoch {epoch} (patience={patience})', flush=True)
                break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    final_train = evaluate_split(model, train_bags, sorted(train_bags.keys()), device=device, max_patches_eval=max_patches_eval)
    final_val = evaluate_split(model, val_bags, val_patient_ids, device=device, max_patches_eval=max_patches_eval)
    final_test = evaluate_split(model, test_bags, test_patient_ids, device=device, max_patches_eval=max_patches_eval)

    if args.skip_holdout:
        holdout_rows, holdout_eval = [], None
    else:
        holdout_rows, holdout_eval = infer_holdout(
            model=model,
            holdout_groups=holdout_groups,
            feature_extractor=feature_extractor,
            transform=transform,
            device=device,
            feature_batch_size=feature_batch_size,
            feature_num_workers=feature_num_workers,
            seed=seed,
            label_map=label_map,
            max_patches_eval=max_patches_eval,
        )

    holdout_pred_path = out_dir / 'preds_holdout.csv'
    pd.DataFrame(holdout_rows).to_csv(holdout_pred_path, index=False)

    train_pred_path = out_dir / 'preds_train.csv'
    val_pred_path = out_dir / 'preds_val.csv'
    test_pred_path = out_dir / 'preds_test.csv'
    pd.DataFrame(final_train['pred_rows']).to_csv(train_pred_path, index=False)
    pd.DataFrame(final_val['pred_rows']).to_csv(val_pred_path, index=False)
    pd.DataFrame(final_test['pred_rows']).to_csv(test_pred_path, index=False)

    plot_confusion(np.array(final_val['confusion_matrix'], dtype=int), out_dir / 'confusion_matrix_val.png', title='Val Confusion Matrix')
    plot_confusion(np.array(final_test['confusion_matrix'], dtype=int), out_dir / 'confusion_matrix_test.png', title='Test Confusion Matrix')
    if holdout_eval is not None:
        plot_confusion(np.array(holdout_eval['confusion_matrix'], dtype=int), out_dir / 'confusion_matrix_holdout.png', title='HoldOut Confusion Matrix')

    metrics = {
        'timestamp': int(time.time()),
        'seed': seed,
        'device': str(device),
        'patch_checkpoint': str(patch_ckpt),
        'class_order': CLASS_ORDER,
        'best_epoch': int(best_epoch),
        'best_val_macro_f1': float(best_val_f1),
        'class_weights': {CLASS_ORDER[i]: float(class_weights[i]) for i in range(len(CLASS_ORDER))},
        'train_class_counts': {CLASS_ORDER[i]: int(class_counts[i]) for i in range(len(CLASS_ORDER))},
        'split_sizes': {
            'train_patients': int(len(train_bags)),
            'val_patients': int(len(val_bags)),
            'test_patients': int(len(test_bags)),
            'holdout_patients': int(len(holdout_groups)),
            'train_patches': int(sum(v['n_patches'] for v in train_bags.values())),
            'val_patches': int(sum(v['n_patches'] for v in val_bags.values())),
            'test_patches': int(sum(v['n_patches'] for v in test_bags.values())),
            'holdout_patches': int(sum(len(v['paths']) for v in holdout_groups.values())),
        },
        'history': history,
        'metrics': {
            'train': {k: v for k, v in final_train.items() if k != 'pred_rows'},
            'val': {k: v for k, v in final_val.items() if k != 'pred_rows'},
            'test': {k: v for k, v in final_test.items() if k != 'pred_rows'},
            'holdout': holdout_eval,
        },
        'artifacts': {
            'best_ckpt': str(best_ckpt_path),
            'preds_train_csv': str(train_pred_path),
            'preds_val_csv': str(val_pred_path),
            'preds_test_csv': str(test_pred_path),
            'preds_holdout_csv': str(holdout_pred_path),
            'confusion_matrix_val_png': str(out_dir / 'confusion_matrix_val.png'),
            'confusion_matrix_test_png': str(out_dir / 'confusion_matrix_test.png'),
            'confusion_matrix_holdout_png': str(out_dir / 'confusion_matrix_holdout.png') if holdout_eval is not None else None,
        },
    }
    metrics_path = out_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print(f'best_epoch: {best_epoch}, best_val_macro_f1: {best_val_f1:.4f}')
    print(f"test_acc: {final_test['accuracy']:.4f}, test_macro_f1: {final_test['macro_f1']:.4f}")
    if holdout_eval is not None:
        print(f"holdout_acc: {holdout_eval['accuracy']:.4f}, holdout_macro_f1: {holdout_eval['macro_f1']:.4f}")
    print(f'metrics: {metrics_path}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
