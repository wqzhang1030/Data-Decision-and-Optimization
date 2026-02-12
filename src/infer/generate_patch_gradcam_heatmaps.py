#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path('/hhome/ricse03/Deep_Learning_Group 3')
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.resnet_presence import build_resnet_presence  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / 'outputs' / 'patch_matrix' / 'resnet18_s42' / 'run_001' / 'best.ckpt'
DEFAULT_SPLIT_CSV = PROJECT_ROOT / 'manifests' / 'patch_test.csv'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'patch_heatmaps' / 'run_001'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tfm(checkpoint: Path, device: torch.device):
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    ckpt = torch.load(checkpoint, map_location=device)
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
    return model, tfm, image_size, backbone


def normalize_name(text: str):
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text))
    text = re.sub(r'_+', '_', text).strip('_')
    return text[:120] if text else 'item'


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, args, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)
        score = logits[0]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError('GradCAM hooks did not capture activations/gradients.')

        a = self.activations[0]
        g = self.gradients[0]
        w = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(w * a, dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        p_pos = float(torch.sigmoid(score).item())
        return cam.detach().cpu().numpy(), p_pos


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray):
    h, w = rgb.shape[:2]
    cam_img = Image.fromarray(np.uint8(np.clip(cam, 0.0, 1.0) * 255.0), mode='L').resize((w, h), resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0

    cmap = plt.get_cmap('jet')(cam_resized)[..., :3]
    overlay = np.clip(0.45 * rgb + 0.55 * cmap, 0.0, 1.0)
    return cam_resized, overlay


def sample_rows(df: pd.DataFrame, max_per_class: int, seed: int):
    pos = df[df['Presence'] == 1].copy()
    neg = df[df['Presence'] == -1].copy()
    n_pos = min(max_per_class, len(pos))
    n_neg = min(max_per_class, len(neg))
    pos = pos.sample(n=n_pos, random_state=seed) if n_pos > 0 else pos.iloc[:0]
    neg = neg.sample(n=n_neg, random_state=seed + 1) if n_neg > 0 else neg.iloc[:0]
    out = pd.concat([pos, neg], ignore_index=True)
    out = out.sort_values(['Presence', 'Pat_ID', 'image_path']).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description='Generate patch-level Grad-CAM heatmaps for Presence model.')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--split-csv', type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--max-per-class', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.max_per_class <= 0:
        raise ValueError('--max-per-class must be > 0')
    if not args.split_csv.exists():
        raise FileNotFoundError(f'Split CSV not found: {args.split_csv}')

    set_seed(args.seed)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    split_df = pd.read_csv(args.split_csv)
    required = {'image_path', 'Pat_ID', 'Presence'}
    missing = required - set(split_df.columns)
    if missing:
        raise RuntimeError(f'{args.split_csv} missing columns: {missing}')

    split_df = split_df[['image_path', 'Pat_ID', 'Presence']].copy()
    split_df['image_path'] = split_df['image_path'].astype(str)
    split_df['Pat_ID'] = split_df['Pat_ID'].astype(str).str.strip()
    split_df['Presence'] = pd.to_numeric(split_df['Presence'], errors='coerce')
    split_df = split_df.dropna(subset=['Presence'])
    split_df['Presence'] = split_df['Presence'].astype(int)
    split_df = split_df[split_df['Presence'].isin([-1, 1])].copy()
    split_df = split_df[split_df['image_path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    if len(split_df) == 0:
        raise RuntimeError('No valid rows found in split CSV after filtering.')

    sampled = sample_rows(split_df, max_per_class=args.max_per_class, seed=args.seed)
    if len(sampled) == 0:
        raise RuntimeError('No rows sampled for heatmaps.')

    model, tfm, image_size, backbone = load_model_and_tfm(args.checkpoint, device=device)
    target_layer = model.layer4[-1] if hasattr(model, 'layer4') else list(model.children())[-2]
    cam_engine = GradCAM(model=model, target_layer=target_layer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_pos = args.output_dir / 'presence_pos'
    out_neg = args.output_dir / 'presence_neg'
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for idx, row in sampled.iterrows():
        img_path = Path(row['image_path'])
        true_presence = int(row['Presence'])
        pat_id = str(row['Pat_ID'])

        with Image.open(img_path) as im:
            im = im.convert('RGB')
            rgb = np.asarray(im).astype(np.float32) / 255.0
            x = tfm(im).unsqueeze(0).to(device)

        cam, p_pos = cam_engine(x)
        _, overlay = overlay_heatmap(rgb, cam)
        pred_presence = 1 if p_pos >= 0.5 else -1

        stem = normalize_name(f'{idx:04d}_{pat_id}_{img_path.stem}')
        out_dir = out_pos if true_presence == 1 else out_neg
        out_png = out_dir / f'{stem}.png'

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        fig.suptitle(
            f'Pat_ID={pat_id} | true={true_presence} | pred={pred_presence} | p_pos={p_pos:.4f}',
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)

        meta_rows.append(
            {
                'Pat_ID': pat_id,
                'image_path': str(img_path),
                'true_presence': true_presence,
                'pred_presence': pred_presence,
                'p_pos': p_pos,
                'heatmap_path': str(out_png),
            }
        )

    cam_engine.remove()

    meta_df = pd.DataFrame(meta_rows)
    meta_path = args.output_dir / 'heatmaps_index.csv'
    meta_df.to_csv(meta_path, index=False)

    summary = {
        'checkpoint': str(args.checkpoint),
        'split_csv': str(args.split_csv),
        'device': str(device),
        'backbone': backbone,
        'image_size': image_size,
        'n_total_rows_in_split': int(len(split_df)),
        'n_heatmaps': int(len(meta_df)),
        'n_true_pos_sampled': int((meta_df['true_presence'] == 1).sum()),
        'n_true_neg_sampled': int((meta_df['true_presence'] == -1).sum()),
        'output_dir': str(args.output_dir),
        'index_csv': str(meta_path),
    }
    summary_path = args.output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'device: {device}')
    print(f'backbone: {backbone}, image_size: {image_size}')
    print(f'heatmaps saved: {len(meta_df)}')
    print(f'index: {meta_path}')
    print(f'summary: {summary_path}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
