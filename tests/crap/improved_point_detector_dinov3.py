"""
Improved Point Detector v2

Key improvements over v1:
1. Higher resolution output (128x128 instead of 32x32) via upsampling
2. Hard negative mining - only backprop on top-k hardest negatives
3. Quality focal loss - penalize confident false positives more heavily
4. Multi-scale feature fusion from DINOv3 intermediate layers

Usage:
    python train_point_v2.py \
        --train_csv /path/to/train.csv \
        --train_image_dir /path/to/images \
        --val_csv /path/to/val.csv \
        --val_image_dir /path/to/val_images \
        --output_dir ./outputs_v2
"""

import os
import argparse
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

from models.visualise import evaluate_with_visualization

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# =============================================================================
# DATASET
# =============================================================================

class PointDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, image_size: int = 512,
                 augment: bool = False, max_points: int = 500):

        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.max_points = max_points

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        self.annotations = {}
        for name, grp in self.df.groupby('images'):
            self.annotations[name] = grp[['x', 'y']].values.astype(np.float32)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.transform = self._build_transform()

        total_pts = sum(len(pts) for pts in self.annotations.values())
        print(f"Loaded {len(self.image_names)} images, {total_pts} points")

    def _build_transform(self):
        if not HAS_ALBUMENTATIONS:
            return None

        if not self.augment:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

        return A.Compose([
            A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                               border_mode=0, p=0.3),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, img_name)).convert('RGB'))
        points = self.annotations[img_name].copy()

        if self.transform:
            result = self.transform(image=image, keypoints=[(p[0], p[1]) for p in points])
            image = result['image']
            points = np.array(result['keypoints'], dtype=np.float32) if result['keypoints'] else np.zeros((0, 2))

        if len(points) > 0:
            valid = (points[:, 0] >= 0) & (points[:, 0] < self.image_size) & \
                    (points[:, 1] >= 0) & (points[:, 1] < self.image_size)
            points = points[valid]

        n_points = len(points)
        padded_points = np.zeros((self.max_points, 2), dtype=np.float32)
        if n_points > 0:
            n_use = min(n_points, self.max_points)
            padded_points[:n_use] = points[:n_use]

        return image, {
            'points': torch.from_numpy(padded_points),
            'n_points': torch.tensor(n_points, dtype=torch.long),
            'image_name': img_name
        }


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# =============================================================================
# DINOV3 MULTI-SCALE BACKBONE
# =============================================================================

class DINOv3MultiScale(nn.Module):
    """Extract features from multiple DINOv3 layers for better multi-scale representation."""

    def __init__(self, model_name: str = 'vit_large_patch16_dinov3.sat493m',
                 pretrained: bool = True, freeze: bool = True,
                 image_size: int = 512, extract_layers: List[int] = [8, 16, 24]):
        super().__init__()

        self.extract_layers = extract_layers
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, img_size=image_size)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)

        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            feat = self.backbone.forward_features(dummy)
            self.feat_dim = feat.shape[-1]
            self.n_tokens = feat.shape[1] - self.num_prefix_tokens
            self.spatial_size = int(np.sqrt(self.n_tokens))

        print(f"DINOv3 Multi-Scale: {model_name}")
        print(f"  Spatial: {self.spatial_size}x{self.spatial_size}, Dim: {self.feat_dim}")
        print(f"  Extract layers: {extract_layers}")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen")

        # Register hooks for intermediate layers
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        for layer_idx in self.extract_layers:
            if layer_idx <= len(self.backbone.blocks):
                self.backbone.blocks[layer_idx - 1].register_forward_hook(
                    hook_fn(f'layer_{layer_idx}')
                )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        self._features = {}

        _ = self.backbone.forward_features(x)

        outputs = {}
        for layer_idx in self.extract_layers:
            key = f'layer_{layer_idx}'
            if key in self._features:
                feat = self._features[key]
                feat = feat[:, self.num_prefix_tokens:, :]
                feat = feat.view(B, self.spatial_size, self.spatial_size, self.feat_dim)
                feat = feat.permute(0, 3, 1, 2)
                outputs[key] = feat

        return outputs

    def unfreeze_last_n(self, n: int):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in self.backbone.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"Unfroze last {n} blocks")


# =============================================================================
# FEATURE PYRAMID WITH UPSAMPLING
# =============================================================================

class FeaturePyramidHead(nn.Module):
    """
    Fuse multi-scale features and upsample to higher resolution.
    32x32 → 128x128 (4x upsampling)
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256,
                 num_scales: int = 3, output_stride: int = 4):
        super().__init__()

        self.output_stride = output_stride

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, 1) for _ in range(num_scales)
        ])

        # Fusion convs (3x3 after adding)
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_scales)
        ])

        # Final fusion of all scales
        self.final_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * num_scales, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Upsampling: 32x32 → 128x128 (4x)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Detection head
        self.conf_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1)
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize conf to predict low probability initially
        # sigmoid(-2) ≈ 0.12, sigmoid(-4) ≈ 0.018
        nn.init.zeros_(self.conf_head[-1].weight)
        nn.init.constant_(self.conf_head[-1].bias, -2.0)  # Start at ~12% not 2%

        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def forward(self, features: Dict[str, torch.Tensor], target_size: int) -> Dict[str, torch.Tensor]:
        # Get features in order (assuming keys are layer_8, layer_16, layer_24)
        feat_list = list(features.values())

        # Apply lateral convs
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feat_list)]

        # Top-down pathway with addition
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode='bilinear', align_corners=False
            )

        # Apply fusion convs
        fused = [conv(lat) for conv, lat in zip(self.fusion_convs, laterals)]

        # Resize all to same size and concatenate
        target_spatial = fused[0].shape[-2:]
        fused = [F.interpolate(f, size=target_spatial, mode='bilinear', align_corners=False)
                 if f.shape[-2:] != target_spatial else f for f in fused]

        combined = self.final_fusion(torch.cat(fused, dim=1))

        # Upsample to higher resolution
        upsampled = self.upsample(combined)

        # Ensure correct output size
        output_size = target_size // self.output_stride
        if upsampled.shape[-1] != output_size:
            upsampled = F.interpolate(upsampled, size=(output_size, output_size),
                                      mode='bilinear', align_corners=False)

        # Predict confidence and offset
        conf = self.conf_head(upsampled).squeeze(1)
        offset = self.offset_head(upsampled)

        return {'conf': conf, 'offset': offset}


# =============================================================================
# FULL MODEL
# =============================================================================

class PointDetectorV2(nn.Module):
    def __init__(self, backbone: str = 'vit_large_patch16_dinov3.sat493m',
                 hidden_dim: int = 256, freeze_backbone: bool = True,
                 image_size: int = 512, output_stride: int = 4,
                 extract_layers: List[int] = [8, 16, 24]):
        super().__init__()

        self.image_size = image_size
        self.output_stride = output_stride
        self.output_size = image_size // output_stride

        self.backbone = DINOv3MultiScale(
            backbone, freeze=freeze_backbone,
            image_size=image_size, extract_layers=extract_layers
        )

        self.head = FeaturePyramidHead(
            self.backbone.feat_dim, hidden_dim,
            num_scales=len(extract_layers), output_stride=output_stride
        )

        print(f"  Output: {self.output_size}x{self.output_size} (stride={output_stride})")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return self.head(features, self.image_size)

    def unfreeze_backbone(self, n_blocks: int = 6):
        self.backbone.unfreeze_last_n(n_blocks)


# =============================================================================
# IMPROVED LOSS WITH HARD NEGATIVE MINING
# =============================================================================

class ImprovedPointLoss(nn.Module):
    """
    Improved loss with:
    1. Hard negative mining - only backprop on top-k hardest negatives
    2. Quality focal weighting - penalize confident false positives more
    3. Warmup period where hard mining is disabled
    """

    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0,
                 offset_weight: float = 1.0, hard_neg_ratio: float = 3.0,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 warmup_steps: int = 500):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.offset_weight = offset_weight
        self.hard_neg_ratio = hard_neg_ratio
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.warmup_steps = warmup_steps
        self._step = 0

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: List[Dict],
                image_size: int, stride: int) -> Tuple[torch.Tensor, Dict]:

        pred_conf = outputs['conf']  # B, H, W
        pred_offset = outputs['offset']  # B, 2, H, W

        B, H, W = pred_conf.shape
        device = pred_conf.device

        # Build targets
        target_conf = torch.zeros_like(pred_conf)
        target_offset = torch.zeros_like(pred_offset)
        pos_mask = torch.zeros_like(pred_conf, dtype=torch.bool)

        total_n_pos = 0

        for b in range(B):
            points = targets[b]['points'].to(device)
            n_points = targets[b]['n_points'].item()

            if n_points == 0:
                continue

            points = points[:n_points]
            total_n_pos += n_points

            # Convert to grid coordinates
            grid_x = (points[:, 0] / stride).long().clamp(0, W - 1)
            grid_y = (points[:, 1] / stride).long().clamp(0, H - 1)

            # Compute offsets
            center_x = (grid_x.float() + 0.5) * stride
            center_y = (grid_y.float() + 0.5) * stride
            offset_x = (points[:, 0] - center_x) / stride
            offset_y = (points[:, 1] - center_y) / stride

            for i in range(n_points):
                gx, gy = grid_x[i], grid_y[i]
                target_conf[b, gy, gx] = 1.0
                target_offset[b, 0, gy, gx] = offset_x[i]
                target_offset[b, 1, gy, gx] = offset_y[i]
                pos_mask[b, gy, gx] = True

        # Compute sigmoid predictions
        pred_sigmoid = torch.sigmoid(pred_conf)

        # === POSITIVE LOSS (simple BCE, no focal for positives) ===
        if pos_mask.any():
            pos_pred = pred_sigmoid[pos_mask]
            pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred), reduction='mean')
        else:
            pos_loss = torch.tensor(0.0, device=device)

        # === NEGATIVE LOSS ===
        neg_mask = ~pos_mask
        neg_pred = pred_sigmoid[neg_mask]

        if neg_pred.numel() > 0:
            # Warmup: use all negatives initially, then hard mining
            warmup_ratio = min(1.0, self._step / self.warmup_steps)

            if warmup_ratio < 1.0 or self.hard_neg_ratio <= 0:
                # During warmup: simple BCE on all negatives
                neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred), reduction='mean')
            else:
                # After warmup: hard negative mining
                n_pos = max(total_n_pos, 1)
                n_hard_neg = int(n_pos * self.hard_neg_ratio)
                n_hard_neg = max(n_hard_neg, 100)  # At least 100 negatives
                n_hard_neg = min(n_hard_neg, neg_pred.numel())

                # Hardest negatives = highest predicted confidence
                hard_neg_pred, _ = neg_pred.topk(n_hard_neg)
                neg_loss = F.binary_cross_entropy(hard_neg_pred, torch.zeros_like(hard_neg_pred), reduction='mean')
        else:
            neg_loss = torch.tensor(0.0, device=device)

        conf_loss = self.pos_weight * pos_loss + self.neg_weight * neg_loss

        # === OFFSET LOSS ===
        if pos_mask.any():
            pos_mask_expanded = pos_mask.unsqueeze(1).expand_as(pred_offset)
            pred_offset_pos = pred_offset[pos_mask_expanded].view(2, -1)
            target_offset_pos = target_offset[pos_mask_expanded].view(2, -1)
            offset_loss = F.smooth_l1_loss(pred_offset_pos, target_offset_pos)
        else:
            offset_loss = torch.tensor(0.0, device=device)

        total = conf_loss + self.offset_weight * offset_loss

        # Logging
        self._step += 1
        if self._step % 50 == 0:
            with torch.no_grad():
                pos_pred_mean = pred_sigmoid[pos_mask].mean().item() if pos_mask.any() else 0
                neg_pred_mean = pred_sigmoid[neg_mask].mean().item() if neg_mask.any() else 0
                neg_pred_max = pred_sigmoid[neg_mask].max().item() if neg_mask.any() else 0
                max_pred = pred_sigmoid.max().item()

                mining_status = "warmup" if self._step < self.warmup_steps else "hard_neg"

                print(f"  [Step {self._step}] loss={total.item():.4f} "
                      f"pos_l={pos_loss.item():.4f} neg_l={neg_loss.item():.4f} | "
                      f"pos={pos_pred_mean:.3f} neg={neg_pred_mean:.3f} neg_max={neg_pred_max:.3f} "
                      f"max={max_pred:.3f} [{mining_status}]")

        return total, {
            'conf_loss': conf_loss.item(),
            'pos_loss': pos_loss.item(),
            'neg_loss': neg_loss.item(),
            'offset_loss': offset_loss.item(),
            'n_pos': total_n_pos
        }


# =============================================================================
# INFERENCE WITH NMS
# =============================================================================

def extract_points(pred_conf: torch.Tensor, pred_offset: torch.Tensor,
                   threshold: float, stride: int, nms_kernel: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract points with local NMS."""
    conf = torch.sigmoid(pred_conf)

    # Local maximum suppression
    conf_pad = F.pad(conf.unsqueeze(0).unsqueeze(0), [nms_kernel//2]*4, mode='constant', value=0)
    conf_max = F.max_pool2d(conf_pad, nms_kernel, stride=1).squeeze()

    keep = (conf == conf_max) & (conf >= threshold)

    if not keep.any():
        return torch.zeros((0, 2), device=conf.device), torch.zeros((0,), device=conf.device)

    y_idx, x_idx = torch.where(keep)
    scores = conf[keep]

    # Apply offset
    offset_x = pred_offset[0, y_idx, x_idx]
    offset_y = pred_offset[1, y_idx, x_idx]

    x = (x_idx.float() + 0.5 + offset_x) * stride
    y = (y_idx.float() + 0.5 + offset_y) * stride

    return torch.stack([x, y], dim=1), scores


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, dataloader, device, image_size, stride,
             threshold=0.3, match_radius=25):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    all_max_conf = []
    all_neg_mean = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            for b in range(len(targets)):
                gt_points = targets[b]['points'].to(device)
                n_gt = targets[b]['n_points'].item()
                gt_points = gt_points[:n_gt]

                pred_conf = outputs['conf'][b]
                pred_offset = outputs['offset'][b]

                conf_sigmoid = torch.sigmoid(pred_conf)
                all_max_conf.append(conf_sigmoid.max().item())
                all_neg_mean.append(conf_sigmoid.mean().item())

                pred_points, _ = extract_points(pred_conf, pred_offset, threshold, stride)

                n_pred = len(pred_points)
                matched_gt, matched_pred = set(), set()

                if n_pred > 0 and n_gt > 0:
                    dists = torch.cdist(pred_points, gt_points)
                    for idx in dists.flatten().argsort():
                        if dists.flatten()[idx] > match_radius:
                            break
                        pi, gi = (idx // n_gt).item(), (idx % n_gt).item()
                        if pi not in matched_pred and gi not in matched_gt:
                            matched_pred.add(pi)
                            matched_gt.add(gi)

                total_tp += len(matched_pred)
                total_fp += n_pred - len(matched_pred)
                total_fn += n_gt - len(matched_gt)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)




    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'avg_max_conf': np.mean(all_max_conf),
        'avg_conf': np.mean(all_neg_mean)
    }


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                 train_loader, val_loader, device, output_dir, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch = 0
        self.best_f1 = 0.0
        self.patience_counter = 0

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self):
        self.model.train()
        total_loss, n = 0, 0
        loss_components = {}

        for images, targets in self.train_loader:
            images = images.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss, loss_dict = self.criterion(
                outputs, targets,
                self.config['image_size'],
                self.model.output_stride
            )

            if torch.isnan(loss):
                print("WARNING: NaN loss")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
            n += 1

        return {'loss': total_loss / max(n, 1),
                **{k: v / n for k, v in loss_components.items()}}

    def save_checkpoint(self, name):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config,
        }, self.output_dir / f'{name}.pth')

    def train(self, epochs, patience=25, unfreeze_epoch=None, unfreeze_blocks=6):
        print(f"\n{'='*60}")
        print(f"Training for {epochs} epochs")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Output stride: {self.model.output_stride}")
        print(f"  Output size: {self.model.output_size}x{self.model.output_size}")
        print(f"  Hard negative ratio: {self.criterion.hard_neg_ratio}")
        print(f"  Hard mining warmup: {self.criterion.warmup_steps} steps")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            if unfreeze_epoch and epoch == unfreeze_epoch:
                print(f"\n⚡ Unfreezing backbone...")
                self.model.unfreeze_backbone(unfreeze_blocks)
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= 0.1
                print(f"  LR reduced to {self.optimizer.param_groups[0]['lr']:.2e}\n")

            train_metrics = self.train_epoch()

            if self.scheduler:
                self.scheduler.step()

            val_metrics = {}
            if self.val_loader:
                val_metrics = evaluate(
                    self.model, self.val_loader, self.device,
                    self.config['image_size'], self.model.output_stride,
                    self.config['threshold'], self.config['match_radius']
                )

                if epoch > 10:
                    # After training, run evaluation with visualization:
                    val_metrics = evaluate_with_visualization(
                        model=self.model,
                        dataloader=self.val_loader,
                        device=self.device,
                        stride=self.model.output_stride,
                        threshold=self.config['threshold'],
                        match_radius=self.config['match_radius'],
                        visualize=True,
                        vis_dir='./visualizations',
                        num_vis=30,
                        extract_points_fn=extract_points
                    )

                if val_metrics['f1'] > self.best_f1:
                    self.best_f1 = val_metrics['f1']
                    self.patience_counter = 0
                    self.save_checkpoint('best')
                else:
                    self.patience_counter += 1

            lr = self.optimizer.param_groups[0]['lr']
            log = f"Epoch {epoch:3d} ({time.time()-t0:.1f}s) | loss={train_metrics['loss']:.4f} | lr={lr:.2e}"

            if val_metrics:
                log += f" | P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} F1={val_metrics['f1']:.3f}"
                log += f" | TP={val_metrics['tp']} FP={val_metrics['fp']} FN={val_metrics['fn']}"
                if val_metrics['f1'] >= self.best_f1:
                    log += " ★"

            print(log)

            if epoch % 10 == 0:
                self.save_checkpoint('latest')

            if patience > 0 and self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.save_checkpoint('final')
        print(f"\n{'='*60}")
        print(f"Done. Best F1: {self.best_f1:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--train_image_dir', type=str, required=True)
    parser.add_argument('--val_csv', type=str)
    parser.add_argument('--val_image_dir', type=str)

    parser.add_argument('--backbone', type=str, default='vit_large_patch16_dinov3.sat493m')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--output_stride', type=int, default=4,
                        help='Output stride (4 = 128x128 output, 8 = 64x64)')
    parser.add_argument('--extract_layers', type=int, nargs='+', default=[8, 16, 24])

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--unfreeze_epoch', type=int, default=None)
    parser.add_argument('--unfreeze_blocks', type=int, default=6)

    parser.add_argument('--hard_neg_ratio', type=float, default=3.0,
                        help='Ratio of hard negatives to positives')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Steps before enabling hard negative mining')
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--match_radius', type=float, default=25)

    parser.add_argument('--output_dir', type=str, default='./outputs_v2')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = vars(args)

    # Data
    print("\n" + "="*60)
    print("DATA")
    print("="*60)

    train_dataset = PointDataset(args.train_csv, args.train_image_dir,
                                  args.image_size, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True)

    val_loader = None
    if args.val_csv and args.val_image_dir:
        val_dataset = PointDataset(args.val_csv, args.val_image_dir,
                                    args.image_size, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate_fn, pin_memory=True)

    # Model
    print("\n" + "="*60)
    print("MODEL")
    print("="*60)

    model = PointDetectorV2(
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        freeze_backbone=True,
        image_size=args.image_size,
        output_stride=args.output_stride,
        extract_layers=args.extract_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}, Trainable: {n_trainable:,}")

    # Loss
    criterion = ImprovedPointLoss(
        hard_neg_ratio=args.hard_neg_ratio,
        focal_gamma=args.focal_gamma,
        warmup_steps=args.warmup_steps
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    # Train
    trainer = Trainer(model, criterion, optimizer, scheduler,
                      train_loader, val_loader, device, args.output_dir, config)
    trainer.train(args.epochs, args.patience, args.unfreeze_epoch, args.unfreeze_blocks)

    # Threshold sweep
    if val_loader:
        print("\n" + "="*60)
        print("THRESHOLD SWEEP")
        print("="*60)

        best_path = Path(args.output_dir) / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])

        print(f"\n{'Thresh':>7} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-" * 55)

        best_f1, best_t = 0, 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            m = evaluate(model, val_loader, device, args.image_size,
                        model.output_stride, t, args.match_radius)
            marker = " ★" if m['f1'] > best_f1 else ""
            print(f"{t:>7.2f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
                  f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}{marker}")
            if m['f1'] > best_f1:
                best_f1, best_t = m['f1'], t

        print("-" * 55)
        print(f"Best: threshold={best_t:.2f} → F1={best_f1:.4f}")


if __name__ == '__main__':
    main()