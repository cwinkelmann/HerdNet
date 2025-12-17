"""
Multi-Backbone Point Detector

Supports multiple backbone architectures:
- Swin Transformer (hierarchical, good for detection)
- ConvNeXt (modern CNN, efficient)
- ResNet (classic, proven)
- DLA (what HerdNet uses - proven for point detection)
- DINOv2/v3 (self-supervised, needs fine-tuning)

Key insight: DINOv2 produces flat feature maps (no hierarchy) while
Swin/ConvNeXt/ResNet/DLA produce multi-scale pyramids naturally.
This makes them better suited for dense prediction tasks.
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

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except ImportError:
    HAS_ALB = False


# =============================================================================
# DATASET
# =============================================================================

class PointDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, image_size: int = 512, augment: bool = False):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()
        self.annotations = {n: g[['x', 'y']].values.astype(np.float32)
                          for n, g in self.df.groupby('images')}

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.transform = self._build_transform()

        print(f"Dataset: {len(self.image_names)} images, {sum(len(v) for v in self.annotations.values())} points")

    def _build_transform(self):
        if not HAS_ALB:
            return None

        if self.augment:
            return A.Compose([
                A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img = np.array(Image.open(os.path.join(self.image_dir, name)).convert('RGB'))
        pts = self.annotations[name].copy()

        h, w = img.shape[:2]
        pts[:, 0] *= self.image_size / w
        pts[:, 1] *= self.image_size / h

        if self.transform:
            r = self.transform(image=img, keypoints=[(p[0], p[1]) for p in pts])
            img = r['image']
            pts = np.array(r['keypoints'], dtype=np.float32) if r['keypoints'] else np.zeros((0, 2))

        if len(pts) > 0:
            valid = (pts[:, 0] >= 0) & (pts[:, 0] < self.image_size) & \
                    (pts[:, 1] >= 0) & (pts[:, 1] < self.image_size)
            pts = pts[valid]

        return img, {
            'points': torch.from_numpy(pts).float(),
            'count': torch.tensor(len(pts), dtype=torch.float32),
            'name': name
        }


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# =============================================================================
# BACKBONE FACTORY
# =============================================================================

class BackboneWithFPN(nn.Module):
    """
    Wrapper that extracts multi-scale features from various backbones.

    Output: Dict with 'p2', 'p3', 'p4', 'p5' feature maps at different scales.
    For input size S:
        p2: S/4 (stride 4)
        p3: S/8 (stride 8)
        p4: S/16 (stride 16)
        p5: S/32 (stride 32)
    """

    def __init__(self, backbone_name: str, pretrained: bool = True,
                 freeze: bool = False, out_channels: int = 256,
                 image_size: int = 512):
        super().__init__()

        self.backbone_name = backbone_name
        self.out_channels = out_channels
        self.image_size = image_size

        # Detect backbone type and create appropriately
        if 'swin' in backbone_name.lower():
            self._init_swin(backbone_name, pretrained)
        elif 'convnext' in backbone_name.lower():
            self._init_convnext(backbone_name, pretrained)
        elif 'resnet' in backbone_name.lower() or 'resnext' in backbone_name.lower():
            self._init_resnet(backbone_name, pretrained)
        elif 'dla' in backbone_name.lower():
            self._init_dla(backbone_name, pretrained)
        elif 'dino' in backbone_name.lower() or 'vit' in backbone_name.lower():
            self._init_vit(backbone_name, pretrained)
        elif 'efficientnet' in backbone_name.lower():
            self._init_efficientnet(backbone_name, pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # FPN lateral connections
        self._build_fpn()

        if freeze:
            self.freeze_backbone()

        print(f"Backbone: {backbone_name}")
        print(f"  Input size: {image_size}x{image_size}")
        print(f"  Feature channels: {self.feat_channels}")
        print(f"  FPN output: {out_channels} channels")

    def _init_swin(self, name: str, pretrained: bool):
        """Initialize Swin Transformer backbone."""
        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          features_only=True, out_indices=(0, 1, 2, 3))

        # Get feature info
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            feats = self.backbone(dummy)
            self.feat_channels = [f.shape[1] for f in feats]
            self.feat_sizes = [f.shape[-1] for f in feats]

        self.backbone_type = 'hierarchical'

    def _init_convnext(self, name: str, pretrained: bool):
        """Initialize ConvNeXt backbone."""
        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          features_only=True, out_indices=(0, 1, 2, 3))

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            feats = self.backbone(dummy)
            self.feat_channels = [f.shape[1] for f in feats]
            self.feat_sizes = [f.shape[-1] for f in feats]

        self.backbone_type = 'hierarchical'

    def _init_resnet(self, name: str, pretrained: bool):
        """Initialize ResNet backbone."""
        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          features_only=True, out_indices=(1, 2, 3, 4))

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            feats = self.backbone(dummy)
            self.feat_channels = [f.shape[1] for f in feats]
            self.feat_sizes = [f.shape[-1] for f in feats]

        self.backbone_type = 'hierarchical'

    def _init_dla(self, name: str, pretrained: bool):
        """Initialize DLA backbone (what HerdNet uses!)."""
        # DLA in timm
        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          features_only=True, out_indices=(1, 2, 3, 4))

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            feats = self.backbone(dummy)
            self.feat_channels = [f.shape[1] for f in feats]
            self.feat_sizes = [f.shape[-1] for f in feats]

        self.backbone_type = 'hierarchical'

    def _init_efficientnet(self, name: str, pretrained: bool):
        """Initialize EfficientNet backbone."""
        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          features_only=True, out_indices=(1, 2, 3, 4))

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            feats = self.backbone(dummy)
            self.feat_channels = [f.shape[1] for f in feats]
            self.feat_sizes = [f.shape[-1] for f in feats]

        self.backbone_type = 'hierarchical'

    def _init_vit(self, name: str, pretrained: bool):
        """Initialize ViT/DINOv2 backbone (flat features, needs special handling)."""
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                          img_size=self.image_size)

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            feat = self.backbone.forward_features(dummy)
            self.vit_feat_dim = feat.shape[-1]
            self.vit_num_prefix = getattr(self.backbone, 'num_prefix_tokens', 1)
            n_tokens = feat.shape[1] - self.vit_num_prefix
            self.vit_spatial = int(np.sqrt(n_tokens))

        # For ViT, we create synthetic multi-scale via convolutions
        self.feat_channels = [self.vit_feat_dim // 4, self.vit_feat_dim // 2,
                              self.vit_feat_dim, self.vit_feat_dim]

        # Compute output sizes (p2=input/4, p3=input/8, p4=input/16, p5=input/32)
        self.feat_sizes = [self.image_size // 4, self.image_size // 8,
                          self.image_size // 16, self.image_size // 32]

        # Calculate upsampling factors from vit_spatial to target sizes
        # vit_spatial is typically image_size/patch_size (e.g., 512/14 ≈ 36 for DINOv2)
        p2_scale = self.feat_sizes[0] / self.vit_spatial
        p3_scale = self.feat_sizes[1] / self.vit_spatial
        p4_scale = self.feat_sizes[2] / self.vit_spatial
        p5_scale = self.feat_sizes[3] / self.vit_spatial

        # Adapters to create multi-scale from flat ViT features
        self.vit_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.vit_feat_dim, self.feat_channels[0], 1),
                nn.BatchNorm2d(self.feat_channels[0]),
                nn.ReLU(inplace=True),
                nn.Upsample(size=self.feat_sizes[0], mode='bilinear', align_corners=False),
            ),  # p2
            nn.Sequential(
                nn.Conv2d(self.vit_feat_dim, self.feat_channels[1], 1),
                nn.BatchNorm2d(self.feat_channels[1]),
                nn.ReLU(inplace=True),
                nn.Upsample(size=self.feat_sizes[1], mode='bilinear', align_corners=False),
            ),  # p3
            nn.Sequential(
                nn.Conv2d(self.vit_feat_dim, self.feat_channels[2], 1),
                nn.BatchNorm2d(self.feat_channels[2]),
                nn.ReLU(inplace=True),
                nn.Upsample(size=self.feat_sizes[2], mode='bilinear', align_corners=False),
            ),  # p4
            nn.Sequential(
                nn.Conv2d(self.vit_feat_dim, self.feat_channels[3], 1),
                nn.BatchNorm2d(self.feat_channels[3]),
                nn.ReLU(inplace=True),
                nn.Upsample(size=self.feat_sizes[3], mode='bilinear', align_corners=False),
            ),  # p5
        ])

        self.backbone_type = 'vit'
        print(f"  ViT spatial: {self.vit_spatial}x{self.vit_spatial} → FPN sizes: {self.feat_sizes}")

    def _build_fpn(self):
        """Build FPN lateral and output convolutions."""
        # Lateral convolutions (reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, self.out_channels, 1) for c in self.feat_channels
        ])

        # Output convolutions (smooth after addition)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            ) for _ in self.feat_channels
        ])

    def freeze_backbone(self):
        """Freeze backbone weights."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone: FROZEN")

    def unfreeze_backbone(self, n_layers: int = -1):
        """Unfreeze backbone (all or last n layers)."""
        if self.backbone_type == 'vit':
            if n_layers < 0:
                for p in self.backbone.parameters():
                    p.requires_grad = True
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for block in self.backbone.blocks[-n_layers:]:
                    for p in block.parameters():
                        p.requires_grad = True
        else:
            # For CNN backbones, unfreeze all or last stages
            for p in self.backbone.parameters():
                p.requires_grad = True

        print(f"  Backbone: UNFROZEN")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features."""

        if self.backbone_type == 'vit':
            # ViT: extract flat features and create multi-scale
            B = x.shape[0]
            feat = self.backbone.forward_features(x)
            feat = feat[:, self.vit_num_prefix:, :]
            feat = feat.view(B, self.vit_spatial, self.vit_spatial, self.vit_feat_dim)
            feat = feat.permute(0, 3, 1, 2)  # B, C, H, W

            feats = [adapter(feat) for adapter in self.vit_adapters]
        else:
            # Hierarchical backbone: direct multi-scale
            feats = self.backbone(x)

        # FPN: top-down pathway
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            size = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=size, mode='bilinear', align_corners=False
            )

        # Output
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return {
            'p2': outputs[0],  # 128x128
            'p3': outputs[1],  # 64x64
            'p4': outputs[2],  # 32x32
            'p5': outputs[3],  # 16x16
        }


# =============================================================================
# DETECTION HEAD
# =============================================================================

class PointDetectionHead(nn.Module):
    """
    Detection head that produces heatmap from FPN features.

    Uses highest resolution feature (p2) with context from lower levels.
    """

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128):
        super().__init__()

        # Fuse p2 with upsampled p3, p4, p5
        self.p3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.p4_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.p5_up = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Heatmap head
        self.heatmap = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

        # Initialize to predict low values initially
        nn.init.constant_(self.heatmap[-1].bias, -2.0)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Dict with 'p2', 'p3', 'p4', 'p5'

        Returns:
            heatmap: (B, H, W) logits at p2 resolution
        """
        p2 = features['p2']
        p3_up = self.p3_up(features['p3'])
        p4_up = self.p4_up(features['p4'])
        p5_up = self.p5_up(features['p5'])

        # Ensure same size
        target_size = p2.shape[-2:]
        if p3_up.shape[-2:] != target_size:
            p3_up = F.interpolate(p3_up, size=target_size, mode='bilinear', align_corners=False)
        if p4_up.shape[-2:] != target_size:
            p4_up = F.interpolate(p4_up, size=target_size, mode='bilinear', align_corners=False)
        if p5_up.shape[-2:] != target_size:
            p5_up = F.interpolate(p5_up, size=target_size, mode='bilinear', align_corners=False)

        fused = torch.cat([p2, p3_up, p4_up, p5_up], dim=1)
        fused = self.fusion(fused)

        return self.heatmap(fused).squeeze(1)


# =============================================================================
# FULL MODEL
# =============================================================================

class MultiBackbonePointDetector(nn.Module):
    """Point detector with configurable backbone."""

    def __init__(self, backbone: str = 'swin_base_patch4_window7_224',
                 freeze_backbone: bool = True, fpn_channels: int = 256,
                 image_size: int = 512):
        super().__init__()

        self.image_size = image_size

        self.backbone = BackboneWithFPN(backbone, pretrained=True,
                                        freeze=freeze_backbone, out_channels=fpn_channels,
                                        image_size=image_size)
        self.head = PointDetectionHead(fpn_channels, hidden_channels=128)

        # Stride for coordinate conversion (output at p2 = stride 4)
        self.stride = 4
        self.output_size = image_size // self.stride

        print(f"  Output heatmap: {self.output_size}x{self.output_size} (stride={self.stride})")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        heatmap = self.head(features)
        return {'heatmap': heatmap, 'features': features}

    def unfreeze_backbone(self, n_layers: int = -1):
        self.backbone.unfreeze_backbone(n_layers)


# =============================================================================
# LOSS
# =============================================================================

class FocalMSELoss(nn.Module):
    """
    MSE loss with focal-style weighting to emphasize hard examples.

    Also includes explicit peak supervision to prevent "stuck at 0.27" problem.
    """

    def __init__(self, sigma: float = 2.0, peak_weight: float = 10.0):
        super().__init__()
        self.sigma = sigma
        self.peak_weight = peak_weight
        self._step = 0

    def generate_target(self, points: torch.Tensor, H: int, W: int,
                        device: torch.device) -> torch.Tensor:
        """Generate Gaussian heatmap target."""
        heatmap = torch.zeros(H, W, device=device)
        if len(points) == 0:
            return heatmap

        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        for pt in points:
            px, py = pt[0].item(), pt[1].item()
            gaussian = torch.exp(-((xx - px)**2 + (yy - py)**2) / (2 * self.sigma**2))
            heatmap = torch.maximum(heatmap, gaussian)

        return heatmap

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: List[Dict], stride: int) -> Tuple[torch.Tensor, Dict]:

        pred = outputs['heatmap']
        B, H, W = pred.shape
        device = pred.device

        pred_prob = torch.sigmoid(pred)

        total_mse = 0.0
        total_peak = 0.0
        n_points = 0
        peak_vals = []
        bg_vals = []

        for b in range(B):
            gt_pts = targets[b]['points'].to(device)
            pts_scaled = gt_pts / stride if len(gt_pts) > 0 else torch.zeros((0, 2), device=device)
            n_points += len(gt_pts)

            target = self.generate_target(pts_scaled, H, W, device)

            # Weighted MSE
            # High weight on peaks, medium on slopes, low on background
            weight = torch.ones_like(target)
            weight[target > 0.5] = 50.0
            weight[(target > 0.1) & (target <= 0.5)] = 10.0

            mse = weight * (pred_prob[b] - target) ** 2
            total_mse += mse.mean()

            # Peak loss: predictions at GT locations should be HIGH
            if len(pts_scaled) > 0:
                px = pts_scaled[:, 0].long().clamp(0, W-1)
                py = pts_scaled[:, 1].long().clamp(0, H-1)
                pred_at_peaks = pred_prob[b, py, px]

                # MSE loss toward 1.0
                peak_loss = F.mse_loss(pred_at_peaks, torch.ones_like(pred_at_peaks))
                total_peak += peak_loss

                peak_vals.extend(pred_at_peaks.detach().cpu().tolist())

            # Sample background values
            bg_mask = target < 0.05
            if bg_mask.any():
                bg_vals.extend(pred_prob[b, bg_mask].detach().cpu().tolist()[:50])

        total_mse /= B
        total_peak = total_peak / B if n_points > 0 else torch.tensor(0.0, device=device)

        total = total_mse + self.peak_weight * total_peak

        # Logging
        self._step += 1
        if self._step % 50 == 0:
            avg_peak = np.mean(peak_vals) if peak_vals else 0
            avg_bg = np.mean(bg_vals) if bg_vals else 0
            print(f"  [Step {self._step}] mse={total_mse.item():.4f} peak_loss={total_peak.item():.4f} | "
                  f"peak_val={avg_peak:.3f} bg_val={avg_bg:.3f} gap={avg_peak-avg_bg:.3f}")

        return total, {'mse': total_mse.item(), 'peak': total_peak.item(), 'n_points': n_points}


# =============================================================================
# DETECTION & EVALUATION
# =============================================================================

def detect_points(heatmap: torch.Tensor, threshold: float = 0.3,
                  nms_kernel: int = 3, stride: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detect points via local maxima."""
    prob = torch.sigmoid(heatmap)

    pad = nms_kernel // 2
    prob_pad = F.pad(prob.unsqueeze(0).unsqueeze(0), [pad]*4, mode='replicate')
    local_max = F.max_pool2d(prob_pad, nms_kernel, stride=1).squeeze()

    keep = (prob == local_max) & (prob >= threshold)

    if not keep.any():
        return torch.zeros((0, 2), device=heatmap.device), torch.zeros((0,), device=heatmap.device)

    y_idx, x_idx = torch.where(keep)
    scores = prob[keep]

    x = (x_idx.float() + 0.5) * stride
    y = (y_idx.float() + 0.5) * stride

    return torch.stack([x, y], dim=1), scores


def evaluate(model, dataloader, device, threshold: float = 0.3, match_radius: float = 25) -> Dict:
    """Evaluate model."""
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    all_peak_scores = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            for b in range(len(targets)):
                gt_pts = targets[b]['points'].to(device)
                heatmap = outputs['heatmap'][b]
                prob = torch.sigmoid(heatmap)

                # Track peak scores
                if len(gt_pts) > 0:
                    px = (gt_pts[:, 0] / model.stride).long().clamp(0, prob.shape[1]-1)
                    py = (gt_pts[:, 1] / model.stride).long().clamp(0, prob.shape[0]-1)
                    all_peak_scores.extend(prob[py, px].cpu().tolist())

                # Detect
                pred_pts, pred_scores = detect_points(heatmap, threshold, 3, model.stride)

                # Match
                n_pred, n_gt = len(pred_pts), len(gt_pts)
                matched_gt, matched_pred = set(), set()

                if n_pred > 0 and n_gt > 0:
                    dists = torch.cdist(pred_pts, gt_pts)
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
        'avg_peak': np.mean(all_peak_scores) if all_peak_scores else 0,
    }


def threshold_sweep(model, dataloader, device, match_radius: float = 25):
    """Find optimal threshold."""
    print("\n" + "="*60)
    print("THRESHOLD SWEEP")
    print("="*60)
    print(f"{'Thresh':>7} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 55)

    best_f1, best_t = 0, 0.3
    for t in np.arange(0.05, 0.95, 0.05):
        m = evaluate(model, dataloader, device, t, match_radius)
        marker = " ★" if m['f1'] > best_f1 else ""
        print(f"{t:>7.2f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
              f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}{marker}")
        if m['f1'] > best_f1:
            best_f1, best_t = m['f1'], t

    print("-" * 55)
    print(f"Best: threshold={best_t:.2f} → F1={best_f1:.4f}")
    return best_t, best_f1


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
        self.best_f1 = 0.0
        self.patience_counter = 0

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self):
        self.model.train()
        total_loss, n = 0, 0

        for images, targets in self.train_loader:
            images = images.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss, _ = self.criterion(outputs, targets, self.model.stride)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def save(self, name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config,
        }, self.output_dir / f'{name}.pth')

    def train(self, epochs, patience=30, unfreeze_epoch=None):
        print(f"\n{'='*60}")
        print(f"Training with backbone: {self.config['backbone']}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            t0 = time.time()

            if unfreeze_epoch and epoch == unfreeze_epoch:
                print(f"\n*** Unfreezing backbone ***")
                self.model.unfreeze_backbone()

                # Add backbone params with lower LR
                backbone_params = [p for p in self.model.backbone.backbone.parameters() if p.requires_grad]
                if backbone_params:
                    self.optimizer.add_param_group({
                        'params': backbone_params,
                        'lr': self.optimizer.param_groups[0]['lr'] * 0.1
                    })

            loss = self.train_epoch()

            if self.scheduler:
                self.scheduler.step()

            val_m = {}
            if self.val_loader:
                val_m = evaluate(self.model, self.val_loader, self.device,
                                self.config['threshold'], self.config['match_radius'])

                if val_m['f1'] > self.best_f1:
                    self.best_f1 = val_m['f1']
                    self.patience_counter = 0
                    self.save('best')
                else:
                    self.patience_counter += 1

            lr = self.optimizer.param_groups[0]['lr']
            log = f"Epoch {epoch:3d} ({time.time()-t0:.1f}s) | loss={loss:.4f} | lr={lr:.2e}"

            if val_m:
                log += f" | P={val_m['precision']:.3f} R={val_m['recall']:.3f} F1={val_m['f1']:.3f}"
                log += f" | peak={val_m['avg_peak']:.3f}"
                if val_m['f1'] >= self.best_f1:
                    log += " ★"

            print(log)

            if epoch % 10 == 0:
                self.save('latest')

            if patience > 0 and self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.save('final')
        print(f"\nBest F1: {self.best_f1:.4f}")


# =============================================================================
# AVAILABLE BACKBONES
# =============================================================================

BACKBONE_OPTIONS = {
    # Swin Transformer (recommended for detection)
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base': 'swin_base_patch4_window7_224',
    'swin_large': 'swin_large_patch4_window7_224',
    'swinv2_tiny': 'swinv2_tiny_window8_256',
    'swinv2_small': 'swinv2_small_window8_256',
    'swinv2_base': 'swinv2_base_window8_256',

    # ConvNeXt (modern CNN)
    'convnext_tiny': 'convnext_tiny',
    'convnext_small': 'convnext_small',
    'convnext_base': 'convnext_base',
    'convnext_large': 'convnext_large',
    'convnextv2_tiny': 'convnextv2_tiny',
    'convnextv2_base': 'convnextv2_base',

    # ResNet (classic)
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
    'resnext50': 'resnext50_32x4d',
    'resnext101': 'resnext101_32x8d',

    # DLA (what HerdNet uses!)
    'dla34': 'dla34',
    'dla46_c': 'dla46_c',
    'dla60': 'dla60',
    'dla102': 'dla102',

    # EfficientNet
    'efficientnet_b0': 'efficientnet_b0',
    'efficientnet_b4': 'efficientnet_b4',
    'efficientnetv2_s': 'efficientnetv2_s',
    'efficientnetv2_m': 'efficientnetv2_m',

    # DINOv2 (needs fine-tuning)
    'dinov2_small': 'vit_small_patch14_dinov2.lvd142m',
    'dinov2_base': 'vit_base_patch14_dinov2.lvd142m',
    'dinov2_large': 'vit_large_patch14_dinov2.lvd142m',
}

# Recommended input sizes for each backbone type
BACKBONE_SIZES = {
    # Swin: window_size=7, patch_size=4 → works best with multiples of 28*n
    # Common: 224, 384, 512, 768
    'swin': [224, 384, 512, 768],
    'swinv2': [256, 384, 512, 768],

    # ConvNeXt/ResNet/DLA: Any multiple of 32 works
    'convnext': [224, 384, 512, 640, 768, 896, 1024],
    'resnet': [224, 384, 512, 640, 768, 896, 1024],
    'resnext': [224, 384, 512, 640, 768, 896, 1024],
    'dla': [224, 384, 512, 640, 768, 896, 1024],
    'efficientnet': [224, 384, 512, 640, 768],

    # DINOv2: patch_size=14 → best at 14*n (224, 518, etc.)
    # But timm interpolates pos embeddings, so 512 works too
    'dinov2': [224, 448, 518, 896],
    'vit': [224, 384, 512, 768],
}


def get_recommended_size(backbone_name: str, target_size: int) -> int:
    """
    Get the recommended input size for a backbone closest to target.

    Args:
        backbone_name: Full backbone name from timm
        target_size: Desired input size

    Returns:
        Recommended input size
    """
    # Determine backbone family
    name_lower = backbone_name.lower()

    if 'swinv2' in name_lower:
        family = 'swinv2'
    elif 'swin' in name_lower:
        family = 'swin'
    elif 'convnext' in name_lower:
        family = 'convnext'
    elif 'resnet' in name_lower or 'resnext' in name_lower:
        family = 'resnet'
    elif 'dla' in name_lower:
        family = 'dla'
    elif 'efficientnet' in name_lower:
        family = 'efficientnet'
    elif 'dinov2' in name_lower:
        family = 'dinov2'
    elif 'vit' in name_lower:
        family = 'vit'
    else:
        # Default: any multiple of 32
        return (target_size // 32) * 32

    sizes = BACKBONE_SIZES[family]

    # Find closest size
    closest = min(sizes, key=lambda x: abs(x - target_size))

    return closest


def validate_image_size(backbone_name: str, image_size: int) -> Tuple[int, str]:
    """
    Validate and potentially adjust image size for backbone.

    Returns:
        (adjusted_size, warning_message or "")
    """
    name_lower = backbone_name.lower()
    warning = ""

    # Check divisibility requirements
    if 'swin' in name_lower:
        # Swin needs size divisible by 32 (patch=4, stages reduce by 2^4)
        if image_size % 32 != 0:
            new_size = (image_size // 32) * 32
            warning = f"Swin requires size divisible by 32. Adjusting {image_size} → {new_size}"
            image_size = new_size

    elif 'dinov2' in name_lower or ('vit' in name_lower and 'patch14' in name_lower):
        # DINOv2 uses patch_size=14
        # While timm interpolates, it's more efficient at multiples of 14
        if image_size % 14 != 0:
            # Find nearest multiple of 14 that's also reasonable
            options = [14 * i for i in range(16, 80)]  # 224 to 1106
            new_size = min(options, key=lambda x: abs(x - image_size))
            warning = f"DINOv2 (patch=14) works best at multiples of 14. Consider {new_size} instead of {image_size}"

    elif 'vit' in name_lower and 'patch16' in name_lower:
        # Standard ViT uses patch_size=16
        if image_size % 16 != 0:
            new_size = (image_size // 16) * 16
            warning = f"ViT (patch=16) works best at multiples of 16. Consider {new_size} instead of {image_size}"

    else:
        # CNNs: just need divisible by 32 for most architectures
        if image_size % 32 != 0:
            new_size = (image_size // 32) * 32
            warning = f"CNN backbones work best with size divisible by 32. Adjusting {image_size} → {new_size}"
            image_size = new_size

    return image_size, warning


def list_backbones():
    print("\nAvailable backbones:")
    print("-" * 80)
    print(f"{'Shortname':<20} {'Model':<40} {'Sizes'}")
    print("-" * 80)

    for name, model in BACKBONE_OPTIONS.items():
        # Get family
        name_lower = model.lower()
        if 'swinv2' in name_lower:
            sizes = "256, 384, 512, 768"
        elif 'swin' in name_lower:
            sizes = "224, 384, 512, 768"
        elif 'dinov2' in name_lower:
            sizes = "224, 448, 518"
        elif any(x in name_lower for x in ['convnext', 'resnet', 'dla', 'efficientnet']):
            sizes = "any ÷32"
        else:
            sizes = "any ÷32"

        print(f"  {name:<18} {model:<40} {sizes}")

    print("-" * 80)
    print("\nRecommended for point detection:")
    print("  1. dla34        - What HerdNet uses, proven, lightweight")
    print("  2. swin_small   - Hierarchical attention, good for detection")
    print("  3. convnext_small - Modern CNN, efficient")
    print("  4. resnet50     - Classic baseline")
    print("-" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')

    parser.add_argument('--backbone', default='swin_small',
                        help='Backbone name (use --list_backbones to see options)')
    parser.add_argument('--list_backbones', action='store_true',
                        help='List available backbones and exit')
    parser.add_argument('--fpn_channels', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size (will be validated for backbone)')

    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--peak_weight', type=float, default=10.0)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--unfreeze_epoch', type=int, default=15)

    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--match_radius', type=float, default=25)

    parser.add_argument('--output_dir', default='./outputs_multibackbone')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.list_backbones:
        list_backbones()
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Resolve backbone name
    backbone_name = BACKBONE_OPTIONS.get(args.backbone, args.backbone)

    # Validate and potentially adjust image size
    image_size, warning = validate_image_size(backbone_name, args.image_size)
    if warning:
        print(f"WARNING: {warning}")

    # Get recommended size if significantly different
    recommended = get_recommended_size(backbone_name, image_size)
    if recommended != image_size:
        print(f"TIP: Consider using --image_size {recommended} for optimal {args.backbone} performance")

    args.backbone = backbone_name
    args.image_size = image_size

    config = vars(args)

    # Data
    print(f"\nUsing image size: {image_size}x{image_size}")
    train_ds = PointDataset(args.train_csv, args.train_image_dir, image_size, augment=True)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)

    val_loader = None
    if args.val_csv and args.val_image_dir:
        val_ds = PointDataset(args.val_csv, args.val_image_dir, image_size, augment=False)
        val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = MultiBackbonePointDetector(
        backbone=backbone_name,
        freeze_backbone=True,
        fpn_channels=args.fpn_channels,
        image_size=image_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}, Trainable: {n_train:,}")

    # Loss
    criterion = FocalMSELoss(sigma=args.sigma, peak_weight=args.peak_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    # Train
    trainer = Trainer(model, criterion, optimizer, scheduler,
                      train_loader, val_loader, device, args.output_dir, config)
    trainer.train(args.epochs, args.patience, args.unfreeze_epoch)

    # Final evaluation
    if val_loader:
        best_path = Path(args.output_dir) / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        threshold_sweep(model, val_loader, device, args.match_radius)


if __name__ == '__main__':
    main()