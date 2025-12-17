"""
DINOv3 → CNN Decoder → DLA → FIDT Point Detector

Architecture:
1. DINOv3 backbone: Extract rich 32×32 features
2. CNN Decoder: Upsample to multi-scale pyramid (32→64→128→256)
3. DLA (Deep Layer Aggregation): Fuse multi-scale features
4. FIDT (Focal Inverse Distance Transform): Adaptive density targets

Usage:
    python train_dinov3_dla_fidt.py \
        --train_csv /path/to/train.csv \
        --train_image_dir /path/to/images \
        --val_csv /path/to/val.csv \
        --val_image_dir /path/to/val_images \
        --output_dir ./outputs_dla_fidt
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
# VISUALIZATION HELPERS
# =============================================================================

def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Convert normalized tensor back to displayable image."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def match_predictions(pred_pts, gt_pts, match_radius):
    """Match predictions to GT, return (tp_pairs, fp_indices, fn_indices)."""
    n_pred, n_gt = len(pred_pts), len(gt_pts)

    if n_pred == 0:
        return [], [], list(range(n_gt))
    if n_gt == 0:
        return [], list(range(n_pred)), []

    dists = torch.cdist(
        torch.from_numpy(pred_pts).float(),
        torch.from_numpy(gt_pts).float()
    ).numpy()

    matched_p, matched_g = set(), set()
    tp_pairs = []

    for idx in np.argsort(dists.flatten()):
        pi, gi = idx // n_gt, idx % n_gt
        if dists[pi, gi] > match_radius:
            break
        if pi not in matched_p and gi not in matched_g:
            matched_p.add(pi)
            matched_g.add(gi)
            tp_pairs.append((pi, gi, dists[pi, gi]))

    return tp_pairs, [i for i in range(n_pred) if i not in matched_p], \
           [i for i in range(n_gt) if i not in matched_g]


def visualize_predictions(image, gt_pts, pred_pts, pred_scores, density_map,
                          match_radius, image_name, save_path):
    """Save visualization of predictions for one image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    tp_pairs, fp_idx, fn_idx = match_predictions(pred_pts, gt_pts, match_radius)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Detections
    ax1 = axes[0]
    ax1.imshow(image)

    for pi, gi, _ in tp_pairs:
        ax1.plot(pred_pts[pi, 0], pred_pts[pi, 1], 'o', color='lime',
                markersize=14, markerfacecolor='none', markeredgewidth=2)
    for pi in fp_idx:
        ax1.plot(pred_pts[pi, 0], pred_pts[pi, 1], 'x', color='red',
                markersize=12, markeredgewidth=2)
    for gi in fn_idx:
        ax1.plot(gt_pts[gi, 0], gt_pts[gi, 1], 's', color='blue',
                markersize=12, markerfacecolor='none', markeredgewidth=2)

    handles = []
    if tp_pairs: handles.append(mpatches.Patch(color='lime', label=f'TP: {len(tp_pairs)}'))
    if fp_idx: handles.append(mpatches.Patch(color='red', label=f'FP: {len(fp_idx)}'))
    if fn_idx: handles.append(mpatches.Patch(color='blue', label=f'FN: {len(fn_idx)}'))
    ax1.legend(handles=handles, loc='upper right')

    p = len(tp_pairs) / max(len(tp_pairs) + len(fp_idx), 1)
    r = len(tp_pairs) / max(len(tp_pairs) + len(fn_idx), 1)
    f1 = 2*p*r / max(p+r, 1e-6)
    ax1.set_title(f'{image_name}\nP={p:.2f} R={r:.2f} F1={f1:.2f}')
    ax1.axis('off')

    # Plot 2: FIDT density heatmap
    ax2 = axes[1]
    ax2.imshow(image)
    density_up = np.array(Image.fromarray((density_map * 255).astype(np.uint8)).resize(
        (image.shape[1], image.shape[0]), Image.BILINEAR)) / 255.0
    im = ax2.imshow(density_up, cmap='hot', vmin=0, vmax=1, alpha=0.5)
    plt.colorbar(im, ax=ax2, fraction=0.046)
    if len(gt_pts) > 0:
        ax2.scatter(gt_pts[:, 0], gt_pts[:, 1], c='cyan', s=40, marker='+', linewidths=2)
    ax2.set_title(f'FIDT Density (max={density_map.max():.3f})')
    ax2.axis('off')

    # Plot 3: Score histogram
    ax3 = axes[2]
    if len(pred_scores) > 0:
        tp_scores = [pred_scores[pi] for pi, _, _ in tp_pairs]
        fp_scores = [pred_scores[pi] for pi in fp_idx]
        bins = np.linspace(0, 1, 21)
        if tp_scores: ax3.hist(tp_scores, bins=bins, alpha=0.7, color='lime', label='TP')
        if fp_scores: ax3.hist(fp_scores, bins=bins, alpha=0.7, color='red', label='FP')
        ax3.legend()
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Count')
    ax3.set_title('Score Distribution')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {'tp': len(tp_pairs), 'fp': len(fp_idx), 'fn': len(fn_idx), 'f1': f1}


# =============================================================================
# DINOV3 BACKBONE
# =============================================================================

class DINOv3Backbone(nn.Module):
    """DINOv3 backbone with optional intermediate layer extraction."""

    def __init__(self, model_name: str = 'vit_large_patch16_dinov3.sat493m',
                 pretrained: bool = True, freeze: bool = True,
                 image_size: int = 512):
        super().__init__()

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

        print(f"DINOv3 Backbone: {model_name}")
        print(f"  Spatial: {self.spatial_size}x{self.spatial_size}, Dim: {self.feat_dim}")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = self.backbone.forward_features(x)
        feat = feat[:, self.num_prefix_tokens:, :]  # Remove CLS token
        feat = feat.view(B, self.spatial_size, self.spatial_size, self.feat_dim)
        feat = feat.permute(0, 3, 1, 2)  # B, C, H, W
        return feat

    def unfreeze_last_n(self, n: int):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in self.backbone.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"Unfroze last {n} blocks")


# =============================================================================
# CNN DECODER: Creates multi-scale features from flat DINOv3 output
# =============================================================================

class CNNDecoder(nn.Module):
    """
    Upsample DINOv3 features to create multi-scale pyramid.

    Input: 32×32 × 1024 (DINOv3 output)
    Output: Multi-scale features
        - P2: 128×128 × 64
        - P3: 64×64 × 128
        - P4: 32×32 × 256
        - P5: 16×16 × 512 (downsampled)
    """

    def __init__(self, in_channels: int = 1024,
                 out_channels: List[int] = [64, 128, 256, 512]):
        super().__init__()

        self.out_channels = out_channels

        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # P5: 32→16 (downsample for context)
        self.down_p5 = nn.Sequential(
            nn.Conv2d(512, out_channels[3], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[3], out_channels[3], 3, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

        # P4: 32×32 (same resolution)
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(512, out_channels[2], 3, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[2], out_channels[2], 3, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True)
        )

        # P3: 32→64 (upsample)
        self.up_p3 = nn.Sequential(
            nn.ConvTranspose2d(512, out_channels[1], 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[1], out_channels[1], 3, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True)
        )

        # P2: 64→128 (upsample from P3)
        self.up_p2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels[1], out_channels[0], 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[0], out_channels[0], 3, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: DINOv3 features (B, 1024, 32, 32)
        Returns:
            Dict with multi-scale features
        """
        x = self.proj(x)  # B, 512, 32, 32

        p5 = self.down_p5(x)   # B, 512, 16, 16
        p4 = self.conv_p4(x)   # B, 256, 32, 32
        p3 = self.up_p3(x)     # B, 128, 64, 64
        p2 = self.up_p2(p3)    # B, 64, 128, 128

        return {'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}


# =============================================================================
# DLA (Deep Layer Aggregation)
# =============================================================================

class IDAUp(nn.Module):
    """Iterative Deep Aggregation upsampling module."""

    def __init__(self, in_channels: int, out_channels: int, up_factor: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                      kernel_size=up_factor * 2,
                                      stride=up_factor,
                                      padding=up_factor // 2,
                                      output_padding=0,
                                      bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Fusion after concatenation
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            low_feat: Lower resolution, higher-level features (to be upsampled)
            high_feat: Higher resolution, lower-level features
        """
        up = self.relu(self.bn(self.up(low_feat)))

        # Handle size mismatch
        if up.shape[-2:] != high_feat.shape[-2:]:
            up = F.interpolate(up, size=high_feat.shape[-2:], mode='bilinear', align_corners=False)

        out = self.fuse(torch.cat([up, high_feat], dim=1))
        return out


class DLAModule(nn.Module):
    """
    Deep Layer Aggregation module.

    Fuses multi-scale features using Iterative Deep Aggregation (IDA).
    """

    def __init__(self, channels: List[int] = [64, 128, 256, 512],
                 out_channels: int = 64):
        super().__init__()

        self.channels = channels

        # Channel reduction for each scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for c in channels
        ])

        # IDA upsamplers: fuse from P5→P4→P3→P2
        # P5 (16) → P4 (32): 2x
        self.ida_p5_p4 = IDAUp(out_channels, out_channels, up_factor=2)
        # P4 (32) → P3 (64): 2x
        self.ida_p4_p3 = IDAUp(out_channels, out_channels, up_factor=2)
        # P3 (64) → P2 (128): 2x
        self.ida_p3_p2 = IDAUp(out_channels, out_channels, up_factor=2)

        # Final refinement
        self.final = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Dict with 'p2', 'p3', 'p4', 'p5'
        Returns:
            Aggregated features at highest resolution (128×128)
        """
        # Reduce channels
        p2 = self.convs[0](features['p2'])  # 128×128
        p3 = self.convs[1](features['p3'])  # 64×64
        p4 = self.convs[2](features['p4'])  # 32×32
        p5 = self.convs[3](features['p5'])  # 16×16

        # Iterative aggregation: bottom-up
        p4 = self.ida_p5_p4(p5, p4)  # 32×32
        p3 = self.ida_p4_p3(p4, p3)  # 64×64
        p2 = self.ida_p3_p2(p3, p2)  # 128×128

        out = self.final(p2)
        return out


# =============================================================================
# FIDT HEAD: Focal Inverse Distance Transform prediction
# =============================================================================

class FIDTHead(nn.Module):
    """
    FIDT prediction head.

    Predicts:
    - Density map (FIDT values)
    - Offset for sub-pixel localization
    """

    def __init__(self, in_channels: int = 64, hidden_channels: int = 64):
        super().__init__()

        # Shared features
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Density head (FIDT values: 0-1)
        self.density_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )

        # Offset head (sub-pixel refinement)
        self.offset_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize density to predict moderate values (not too suppressed)
        # sigmoid(0) = 0.5, sigmoid(-1) = 0.27
        nn.init.zeros_(self.density_head[-1].weight)
        nn.init.constant_(self.density_head[-1].bias, -1.0)  # Changed from -2.0

        # Initialize offset to zero
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(x)
        density = self.density_head(shared).squeeze(1)  # B, H, W
        offset = self.offset_head(shared)  # B, 2, H, W

        return {'density': density, 'offset': offset}


# =============================================================================
# FULL MODEL
# =============================================================================

class DINOv3_DLA_FIDT(nn.Module):
    """
    Full model: DINOv3 → CNN Decoder → DLA → FIDT
    """

    def __init__(self,
                 backbone: str = 'vit_large_patch16_dinov3.sat493m',
                 freeze_backbone: bool = True,
                 image_size: int = 512,
                 dla_channels: int = 64,
                 decoder_channels: List[int] = [64, 128, 256, 512]):
        super().__init__()

        self.image_size = image_size

        # DINOv3 backbone
        self.backbone = DINOv3Backbone(backbone, freeze=freeze_backbone,
                                        image_size=image_size)

        # CNN Decoder: create multi-scale pyramid
        self.decoder = CNNDecoder(self.backbone.feat_dim, decoder_channels)

        # DLA: aggregate multi-scale features
        self.dla = DLAModule(decoder_channels, dla_channels)

        # FIDT head
        self.head = FIDTHead(dla_channels, dla_channels)

        # Output info
        self.output_stride = image_size // 128  # 512/128 = 4
        self.output_size = 128

        print(f"  Output: {self.output_size}×{self.output_size} (stride={self.output_stride})")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone
        feat = self.backbone(x)  # B, 1024, 32, 32

        # Decoder: create multi-scale
        multi_scale = self.decoder(feat)  # p2, p3, p4, p5

        # DLA: aggregate
        aggregated = self.dla(multi_scale)  # B, 64, 128, 128

        # Head: predict density + offset
        out = self.head(aggregated)

        return out

    def unfreeze_backbone(self, n_blocks: int = 6):
        self.backbone.unfreeze_last_n(n_blocks)


# =============================================================================
# FIDT LOSS
# =============================================================================

class FIDTLoss(nn.Module):
    """
    Focal Inverse Distance Transform Loss.

    FIDT target: For each pixel, value = 1 / (1 + min_distance_to_any_point)
    This creates adaptive peaks - sharper when objects are dense.

    Loss: Focal loss to focus on hard examples near decision boundary.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0,
                 offset_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Focal loss alpha (positive weight)
        self.beta = beta    # Focal loss beta (negative down-weight)
        self.offset_weight = offset_weight
        self._step = 0

        print(f"FIDT Loss: alpha={alpha}, beta={beta}")

    def _compute_fidt_target(self, points: torch.Tensor, H: int, W: int,
                              device: torch.device) -> torch.Tensor:
        """
        Compute FIDT target map.

        Args:
            points: GT points (N, 2) in pixel coordinates
            H, W: Target map size
            device: torch device

        Returns:
            FIDT map (H, W) with values in [0, 1]
        """
        if len(points) == 0:
            return torch.zeros(H, W, device=device)

        # Create coordinate grid
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1)  # H, W, 2

        # Compute distance to each point
        coords_flat = coords.view(-1, 2)  # H*W, 2
        dists = torch.cdist(coords_flat, points)  # H*W, N

        # Minimum distance to any point
        min_dist = dists.min(dim=1).values.view(H, W)

        # Inverse distance transform: 1 / (1 + d)
        # This gives 1.0 at point locations, decaying outward
        fidt = 1.0 / (1.0 + min_dist)

        return fidt

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: List[Dict],
                image_size: int, stride: int) -> Tuple[torch.Tensor, Dict]:

        pred_density = outputs['density']  # B, H, W (logits)
        pred_offset = outputs['offset']    # B, 2, H, W

        B, H, W = pred_density.shape
        device = pred_density.device

        # Build FIDT targets
        target_fidt = torch.zeros_like(pred_density)
        target_offset = torch.zeros_like(pred_offset)
        pos_mask = torch.zeros_like(pred_density, dtype=torch.bool)

        total_n_pos = 0

        for b in range(B):
            points = targets[b]['points'].to(device)
            n_points = targets[b]['n_points'].item()

            if n_points == 0:
                continue

            points = points[:n_points]
            total_n_pos += n_points

            # Scale points to feature map coordinates
            points_scaled = points / stride

            # Compute FIDT target
            target_fidt[b] = self._compute_fidt_target(points_scaled, H, W, device)

            # Offset targets at peak locations
            grid_x = points_scaled[:, 0].long().clamp(0, W - 1)
            grid_y = points_scaled[:, 1].long().clamp(0, H - 1)

            center_x = grid_x.float() + 0.5
            center_y = grid_y.float() + 0.5

            for i in range(n_points):
                gx, gy = grid_x[i].item(), grid_y[i].item()
                target_offset[b, 0, gy, gx] = points_scaled[i, 0] - center_x[i]
                target_offset[b, 1, gy, gx] = points_scaled[i, 1] - center_y[i]
                pos_mask[b, gy, gx] = True

        # Debug: first forward pass
        if self._step == 0:
            print(f"  [DEBUG] First forward: B={B}, H={H}, W={W}, n_pos={total_n_pos}")
            print(f"  [DEBUG] target_fidt: min={target_fidt.min():.4f} max={target_fidt.max():.4f} mean={target_fidt.mean():.4f}")
            print(f"  [DEBUG] pred_density: min={pred_density.min():.4f} max={pred_density.max():.4f}")

        # === BCE LOSS with pos_weight ===
        pred_sigmoid = torch.sigmoid(pred_density)

        # Compute class weights based on target distribution
        # More weight on positive pixels (which are rare)
        n_pos_pixels = (target_fidt > 0.5).sum().float()
        n_neg_pixels = (target_fidt <= 0.5).sum().float()

        # Use BCEWithLogits for numerical stability
        # Create per-pixel weights
        weight = torch.ones_like(target_fidt)
        weight[target_fidt > 0.5] = 50.0   # High weight for center pixels
        weight[(target_fidt > 0.1) & (target_fidt <= 0.5)] = 10.0  # Medium for nearby

        density_loss = F.binary_cross_entropy_with_logits(
            pred_density, target_fidt, weight=weight, reduction='mean'
        )

        # === OFFSET LOSS at peak locations ===
        if pos_mask.any():
            pos_mask_exp = pos_mask.unsqueeze(1).expand_as(pred_offset)
            pred_off_pos = pred_offset[pos_mask_exp].view(2, -1)
            target_off_pos = target_offset[pos_mask_exp].view(2, -1)
            offset_loss = F.smooth_l1_loss(pred_off_pos, target_off_pos)
        else:
            offset_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total = density_loss + self.offset_weight * offset_loss

        # Logging
        self._step += 1
        if self._step % 50 == 0:
            with torch.no_grad():
                pos_pred_mean = pred_sigmoid[pos_mask].mean().item() if pos_mask.any() else 0
                neg_pred_mean = pred_sigmoid[target_fidt < 0.1].mean().item() if (target_fidt < 0.1).any() else 0
                max_pred = pred_sigmoid.max().item()

                print(f"  [Step {self._step}] loss={total.item():.4f} "
                      f"density={density_loss.item():.4f} offset={offset_loss.item():.4f} | "
                      f"pos={pos_pred_mean:.3f} neg={neg_pred_mean:.3f} max={max_pred:.3f} n_pos={total_n_pos}")

        return total, {
            'density_loss': density_loss.item(),
            'offset_loss': offset_loss.item(),
            'n_pos': total_n_pos
        }


# =============================================================================
# LOCAL MAXIMUM DETECTION
# =============================================================================

def local_maximum_detection(density: torch.Tensor, offset: torch.Tensor,
                            threshold: float, stride: int,
                            kernel_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect points via local maximum on density map.

    Args:
        density: Density logits (H, W)
        offset: Offset predictions (2, H, W)
        threshold: Detection threshold
        stride: Output stride (for converting back to image coords)
        kernel_size: NMS kernel size

    Returns:
        points: (N, 2) detected points in image coordinates
        scores: (N,) confidence scores
    """
    prob = torch.sigmoid(density)

    # Local maximum suppression
    pad = kernel_size // 2
    prob_pad = F.pad(prob.unsqueeze(0).unsqueeze(0), [pad]*4, mode='constant', value=0)
    prob_max = F.max_pool2d(prob_pad, kernel_size, stride=1).squeeze()

    # Keep only local maxima above threshold
    keep = (prob == prob_max) & (prob >= threshold)

    if not keep.any():
        return torch.zeros((0, 2), device=density.device), torch.zeros((0,), device=density.device)

    y_idx, x_idx = torch.where(keep)
    scores = prob[keep]

    # Apply offset refinement
    off_x = offset[0, y_idx, x_idx]
    off_y = offset[1, y_idx, x_idx]

    # Convert to image coordinates
    x = (x_idx.float() + 0.5 + off_x) * stride
    y = (y_idx.float() + 0.5 + off_y) * stride

    return torch.stack([x, y], dim=1), scores


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, dataloader, device, image_size, stride,
             threshold=0.3, match_radius=25, visualize=False,
             vis_dir='./visualizations', num_vis=20):
    """Evaluate model with optional visualization."""
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    all_max_conf = []
    all_results = []
    all_data = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            for b in range(len(targets)):
                gt_points = targets[b]['points'].to(device)
                n_gt = targets[b]['n_points'].item()
                gt_points = gt_points[:n_gt]

                pred_density = outputs['density'][b]
                pred_offset = outputs['offset'][b]

                prob = torch.sigmoid(pred_density)
                all_max_conf.append(prob.max().item())

                pred_points, pred_scores = local_maximum_detection(
                    pred_density, pred_offset, threshold, stride
                )

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

                batch_tp = len(matched_pred)
                batch_fp = n_pred - len(matched_pred)
                batch_fn = n_gt - len(matched_gt)

                total_tp += batch_tp
                total_fp += batch_fp
                total_fn += batch_fn

                # Store for visualization
                if visualize:
                    p = batch_tp / max(batch_tp + batch_fp, 1)
                    r = batch_tp / max(batch_tp + batch_fn, 1)
                    f1 = 2*p*r / max(p+r, 1e-6)

                    all_results.append({
                        'name': targets[b]['image_name'],
                        'f1': f1, 'tp': batch_tp, 'fp': batch_fp, 'fn': batch_fn
                    })
                    all_data.append({
                        'image': denormalize_image(images[b]),
                        'gt_points': gt_points.cpu().numpy(),
                        'pred_points': pred_points.cpu().numpy(),
                        'pred_scores': pred_scores.cpu().numpy(),
                        'density_map': prob.cpu().numpy(),
                        'name': targets[b]['image_name']
                    })

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # Save visualizations
    if visualize and all_data:
        vis_path = Path(vis_dir)
        vis_path.mkdir(parents=True, exist_ok=True)

        # Sort by F1 (worst first)
        sorted_idx = sorted(range(len(all_results)), key=lambda i: all_results[i]['f1'])

        print(f"\nSaving {min(num_vis, len(sorted_idx))} visualizations to {vis_path}")

        for rank, idx in enumerate(sorted_idx[:num_vis]):
            data = all_data[idx]
            result = all_results[idx]

            save_path = vis_path / f'{rank:03d}_{Path(data["name"]).stem}_f1={result["f1"]:.2f}.png'

            visualize_predictions(
                image=data['image'],
                gt_pts=data['gt_points'],
                pred_pts=data['pred_points'],
                pred_scores=data['pred_scores'],
                density_map=data['density_map'],
                match_radius=match_radius,
                image_name=data['name'],
                save_path=str(save_path)
            )

            print(f"  [{rank+1}] {data['name']}: F1={result['f1']:.3f} TP={result['tp']} FP={result['fp']} FN={result['fn']}")

        # Summary
        print(f"\nWorst 5 by F1:")
        for r in sorted(all_results, key=lambda x: x['f1'])[:5]:
            print(f"  {r['name']}: F1={r['f1']:.3f}")

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'avg_max_conf': np.mean(all_max_conf) if all_max_conf else 0
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
        print(f"Training DINOv3-DLA-FIDT for {epochs} epochs")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Output: {self.model.output_size}×{self.model.output_size}")
        print(f"  Stride: {self.model.output_stride}")
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
                log += f" | max={val_metrics['avg_max_conf']:.3f} TP={val_metrics['tp']} FP={val_metrics['fp']} FN={val_metrics['fn']}"
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
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--dla_channels', type=int, default=64)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--unfreeze_epoch', type=int, default=None)
    parser.add_argument('--unfreeze_blocks', type=int, default=6)

    # FIDT loss parameters
    parser.add_argument('--focal_alpha', type=float, default=2.0)
    parser.add_argument('--focal_beta', type=float, default=4.0)

    parser.add_argument('--threshold', type=float, default=0.1)  # Lower default for debugging
    parser.add_argument('--match_radius', type=float, default=25)

    parser.add_argument('--visualize', action='store_true',
                        help='Save visualizations after training')
    parser.add_argument('--num_vis', type=int, default=30,
                        help='Number of images to visualize')

    parser.add_argument('--output_dir', type=str, default='./outputs_dla_fidt')
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

    if len(train_dataset) < args.batch_size:
        print(f"WARNING: Training set ({len(train_dataset)} images) < batch_size ({args.batch_size})")
        print(f"  Consider using --batch_size {max(1, len(train_dataset))}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True, drop_last=False)

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

    model = DINOv3_DLA_FIDT(
        backbone=args.backbone,
        freeze_backbone=True,
        image_size=args.image_size,
        dla_channels=args.dla_channels
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}, Trainable: {n_trainable:,}")

    # Loss
    criterion = FIDTLoss(
        alpha=args.focal_alpha,
        beta=args.focal_beta
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
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
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

        # Final visualization at best threshold
        if args.visualize:
            print("\n" + "="*60)
            print(f"VISUALIZATION (threshold={best_t:.2f})")
            print("="*60)

            vis_dir = Path(args.output_dir) / 'visualizations'
            evaluate(model, val_loader, device, args.image_size,
                    model.output_stride, best_t, args.match_radius,
                    visualize=True, vis_dir=str(vis_dir), num_vis=args.num_vis)


if __name__ == '__main__':
    main()
