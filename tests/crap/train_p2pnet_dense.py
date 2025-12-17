"""
P2PNet Dense - Predict at Every Feature Location

Instead of sparse queries (196), predict at EVERY feature map location.
For a 512x512 image with DINOv2-large (patch=14), feature map is 37x37 = 1369 locations.

This is similar to:
- FCOS (fully convolutional one-stage detection)
- HerdNet (but with points instead of heatmaps)
- Original P2PNet

Key differences from sparse P2P:
- No Hungarian matching needed during inference (use NMS instead)
- Much denser coverage = better recall
- Simpler architecture

Target: F1 > 0.8
"""

import os
import argparse
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.optimize import linear_sum_assignment
import timm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from albumentations.core.transforms_interface import DualTransform
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# =============================================================================
# DATASET
# =============================================================================

if HAS_ALBUMENTATIONS:
    class ObjectAwareRandomCrop(DualTransform):
        def __init__(self, height, width, min_edge_distance=10, empty_probability=0.0,
                     max_attempts=10, always_apply=False, p=1.0):
            super().__init__(always_apply, p)
            self.height, self.width = height, width
            self.min_edge_distance = min_edge_distance
            self.empty_probability = empty_probability
            self.max_attempts = max_attempts

        def _get_crop_with_keypoint(self, keypoint_coords, h, w):
            available = keypoint_coords.copy()
            random.shuffle(available)
            for attempt in range(min(self.max_attempts, len(available) * 2)):
                kp_x, kp_y = available[attempt % len(available)]
                x_min = max(0, int(kp_x - self.width + self.min_edge_distance))
                x_max = min(w - self.width, int(kp_x - self.min_edge_distance))
                y_min = max(0, int(kp_y - self.height + self.min_edge_distance))
                y_max = min(h - self.height, int(kp_y - self.min_edge_distance))
                if x_min <= x_max and y_min <= y_max:
                    return random.randint(x_min, x_max), random.randint(y_min, y_max)
            kp_x, kp_y = random.choice(keypoint_coords)
            return (max(0, min(int(kp_x - self.width // 2), w - self.width)),
                    max(0, min(int(kp_y - self.height // 2), h - self.height)))

        def apply(self, img, crop_x=0, crop_y=0, **params):
            return img[crop_y:crop_y + self.height, crop_x:crop_x + self.width]

        def apply_to_keypoint(self, keypoint, crop_x=0, crop_y=0, **params):
            return keypoint[0] - crop_x, keypoint[1] - crop_y, keypoint[2], keypoint[3]

        def get_params_dependent_on_targets(self, params):
            img, keypoints = params['image'], params.get('keypoints', [])
            h, w = img.shape[:2]
            kp_coords = [(kp[0], kp[1]) for kp in keypoints]
            if not kp_coords or random.random() < self.empty_probability:
                return {'crop_x': random.randint(0, w - self.width),
                        'crop_y': random.randint(0, h - self.height)}
            return dict(zip(['crop_x', 'crop_y'], self._get_crop_with_keypoint(kp_coords, h, w)))

        @property
        def targets_as_params(self):
            return ['image', 'keypoints']

        def get_transform_init_args_names(self):
            return ('height', 'width', 'min_edge_distance', 'empty_probability', 'max_attempts')


class PointDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_size=512, augment=False, max_images=None,
                 use_object_aware_crop=False, min_edge_distance=20, empty_probability=0.1):
        self.image_dir, self.image_size, self.augment = image_dir, image_size, augment
        self.use_object_aware_crop = use_object_aware_crop

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()
        if max_images:
            self.image_names = self.image_names[:max_images]
            self.df = self.df[self.df['images'].isin(self.image_names)]

        self.annotations = {name: {'points': grp[['x', 'y']].values.astype(np.float32),
                                   'labels': grp['labels'].values.astype(np.int64) if 'labels' in grp else np.ones(len(grp), dtype=np.int64)}
                           for name, grp in self.df.groupby('images') if name in self.image_names}

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.transform = self._build_transform(min_edge_distance, empty_probability)
        print(f"Loaded {len(self.image_names)} images, {sum(len(a['points']) for a in self.annotations.values())} annotations")

    def _build_transform(self, min_edge_distance, empty_probability):
        if not HAS_ALBUMENTATIONS:
            return None
        if not self.augment:
            return A.Compose([A.Resize(self.image_size, self.image_size),
                              A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()), ToTensorV2()],
                             keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

        transforms = ([ObjectAwareRandomCrop(self.image_size, self.image_size, min_edge_distance, empty_probability)]
                      if self.use_object_aware_crop else [A.Resize(self.image_size, self.image_size)])
        transforms += [
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.5),
            A.OneOf([A.RandomBrightnessContrast(0.3, 0.3, p=1), A.HueSaturationValue(20, 30, 30, p=1)], p=0.7),
            A.OneOf([A.GaussNoise(var_limit=(10, 80), p=1), A.GaussianBlur((3, 7), p=1)], p=0.3),
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()), ToTensorV2()
        ]
        return A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, img_name)).convert('RGB'))
        anno = self.annotations[img_name]
        points, labels = anno['points'].copy(), anno['labels'].copy()

        if self.transform:
            result = self.transform(image=image, keypoints=[(p[0], p[1]) for p in points])
            image = result['image']
            points = np.array(result['keypoints'], dtype=np.float32) if result['keypoints'] else np.zeros((0, 2), dtype=np.float32)
            labels = labels[:len(points)] if len(points) > 0 else np.array([], dtype=np.int64)

        if len(points) > 0:
            valid = (points[:, 0] >= 0) & (points[:, 0] < self.image_size) & (points[:, 1] >= 0) & (points[:, 1] < self.image_size)
            points, labels = points[valid], labels[valid] if len(labels) == len(valid) else labels[:sum(valid)]

        return image, {'points': torch.from_numpy(points).float(),
                       'labels': torch.from_numpy(labels).long() if len(labels) else torch.zeros(len(points), dtype=torch.long),
                       'image_name': img_name}


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# =============================================================================
# DENSE P2PNET MODEL
# =============================================================================

class DenseP2PNet(nn.Module):
    """
    Dense prediction model - predict at every feature map location.

    For each location (i, j) in the feature map:
    - Predict: Is there an object here? (binary classification)
    - Predict: Offset (dx, dy) to the exact object center

    Much denser than sparse queries: 37x37=1369 vs 196 queries.
    """

    def __init__(
        self,
        backbone: str = 'vit_large_patch14_dinov2.lvd142m',
        hidden_dim: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.is_vit = 'vit' in backbone.lower() or 'dino' in backbone.lower()

        # Build backbone
        if self.is_vit:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            with torch.no_grad():
                dummy = torch.randn(1, 3, 512, 512)
                feat = self.backbone.forward_features(dummy)
                self.feat_dim = feat.shape[-1]
                num_tokens = feat.shape[1]
                self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)
                spatial_tokens = num_tokens - self.num_prefix_tokens
                self.spatial_size = int(np.sqrt(spatial_tokens))
            print(f"ViT: {backbone}, feat_dim={self.feat_dim}, spatial={self.spatial_size}x{self.spatial_size}")
            print(f"  -> {self.spatial_size * self.spatial_size} prediction locations")
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=[-1])
            with torch.no_grad():
                feat = self.backbone(torch.randn(1, 3, 512, 512))[-1]
                self.feat_dim = feat.shape[1]
                self.spatial_size = feat.shape[2]
            print(f"CNN: {backbone}, feat_dim={self.feat_dim}, spatial={self.spatial_size}x{self.spatial_size}")

        if freeze_backbone:
            print("Freezing backbone")
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Feature processing - local context is crucial for dense prediction
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.feat_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Classification head - predict objectness at each location
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

        # Regression head - predict offset to object center
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1),  # (dx, dy) offset
        )

        self._init_weights()

        # Pre-compute grid coordinates (normalized 0-1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5 / self.spatial_size, 1 - 0.5 / self.spatial_size, self.spatial_size),
            torch.linspace(0.5 / self.spatial_size, 1 - 0.5 / self.spatial_size, self.spatial_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

    def _init_weights(self):
        for m in [self.feature_proj, self.cls_head, self.reg_head]:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize classification to predict background
        nn.init.zeros_(self.cls_head[-1].weight)
        nn.init.constant_(self.cls_head[-1].bias[0], 2.0)   # Background
        nn.init.constant_(self.cls_head[-1].bias[1], -2.0)  # Foreground

        # Initialize regression to predict zero offset
        nn.init.zeros_(self.reg_head[-1].weight)
        nn.init.zeros_(self.reg_head[-1].bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]

        # Extract features
        if self.is_vit:
            features = self.backbone.forward_features(x)
            if self.num_prefix_tokens > 0:
                features = features[:, self.num_prefix_tokens:, :]
            features = features.view(B, self.spatial_size, self.spatial_size, self.feat_dim)
            features = features.permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            features = self.backbone(x)[-1]

        # Process features
        feat = self.feature_proj(features)  # [B, hidden_dim, H, W]

        # Predict
        cls_logits = self.cls_head(feat)  # [B, 2, H, W]
        reg_offset = self.reg_head(feat)  # [B, 2, H, W]

        # Compute final point locations
        # Grid gives center of each cell, offset adjusts within cell
        # Offset is in units of grid cell size, clamp to reasonable range
        offset_scale = 1.0 / self.spatial_size  # One grid cell
        reg_offset = torch.tanh(reg_offset) * offset_scale * 2  # Allow ±2 cells movement

        # Final points: grid + offset
        points_x = self.grid_x.unsqueeze(0).expand(B, -1, -1) + reg_offset[:, 0]
        points_y = self.grid_y.unsqueeze(0).expand(B, -1, -1) + reg_offset[:, 1]
        points = torch.stack([points_x, points_y], dim=-1)  # [B, H, W, 2]
        points = points.clamp(0, 1)

        return {
            'logits': cls_logits.permute(0, 2, 3, 1).reshape(B, -1, 2),  # [B, H*W, 2]
            'pred_points_normalized': points.reshape(B, -1, 2),  # [B, H*W, 2]
            'reg_offset': reg_offset,  # [B, 2, H, W] for visualization
        }

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# =============================================================================
# LOSS - Stable version with better balancing
# =============================================================================

class DensePointLoss(nn.Module):
    """
    Focal loss with online hard negative mining.
    Much better for extreme class imbalance (1369 locations, ~10 positives).
    """

    def __init__(
        self,
        image_size: int = 512,
        cls_weight: float = 1.0,
        reg_weight: float = 5.0,
        assignment_radius: float = 0.08,
        focal_alpha: float = 0.75,  # Weight for positives
        focal_gamma: float = 2.0,   # Focus on hard examples
        neg_pos_ratio: int = 3,     # Max negatives per positive
    ):
        super().__init__()
        self.image_size = image_size
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.assignment_radius = assignment_radius
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.neg_pos_ratio = neg_pos_ratio
        self._step = 0

    def _normalize_gt(self, pts):
        pts = pts.float()
        return pts / self.image_size if pts.numel() > 0 and pts.max() > 1.0 else pts

    def forward(self, outputs, targets):
        logits = outputs['logits']  # [B, H*W, 2]
        points = outputs['pred_points_normalized']  # [B, H*W, 2]
        device = logits.device
        B, N, C = logits.shape

        # Build targets
        cls_targets = torch.zeros(B, N, dtype=torch.long, device=device)
        reg_targets = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        reg_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        total_assigned = 0
        total_gt = 0

        for b in range(B):
            gt_pts = self._normalize_gt(targets[b]['points']).to(device)
            total_gt += len(gt_pts)
            if len(gt_pts) == 0:
                continue

            # Distance from each GT to each prediction
            dists = torch.cdist(gt_pts, points[b])  # [M, N]

            # Greedy assignment: each GT gets nearest unassigned pred
            assigned_preds = set()
            for gt_idx in range(len(gt_pts)):
                pred_dists = dists[gt_idx].clone()
                for p in assigned_preds:
                    pred_dists[p] = float('inf')

                min_dist, pred_idx = pred_dists.min(dim=0)
                pred_idx = pred_idx.item()

                if min_dist < self.assignment_radius:
                    assigned_preds.add(pred_idx)
                    cls_targets[b, pred_idx] = 1
                    reg_targets[b, pred_idx] = gt_pts[gt_idx]
                    reg_mask[b, pred_idx] = True
                    total_assigned += 1

        # Focal loss per sample
        probs = logits.softmax(-1)
        ce_loss = F.cross_entropy(logits.view(-1, C), cls_targets.view(-1), reduction='none')
        ce_loss = ce_loss.view(B, N)

        pt = probs.gather(-1, cls_targets.unsqueeze(-1)).squeeze(-1)  # probability of true class
        focal_weight = (1 - pt) ** self.focal_gamma

        # Alpha weighting
        alpha_t = torch.where(cls_targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_loss = alpha_t * focal_weight * ce_loss

        # Hard negative mining per image
        total_cls_loss = 0.0
        total_samples = 0

        for b in range(B):
            pos_mask = cls_targets[b] == 1
            neg_mask = cls_targets[b] == 0

            n_pos = pos_mask.sum().item()
            n_neg_keep = min(neg_mask.sum().item(), max(n_pos * self.neg_pos_ratio, 50))

            # All positive losses
            if n_pos > 0:
                total_cls_loss += focal_loss[b, pos_mask].sum()
                total_samples += n_pos

            # Top-k hardest negatives
            if n_neg_keep > 0:
                neg_losses = focal_loss[b, neg_mask]
                hard_neg_losses, _ = neg_losses.topk(n_neg_keep)
                total_cls_loss += hard_neg_losses.sum()
                total_samples += n_neg_keep

        cls_loss = total_cls_loss / max(total_samples, 1)

        # Regression loss
        if reg_mask.sum() > 0:
            pred_pts = points[reg_mask]
            target_pts = reg_targets[reg_mask]
            reg_loss = F.smooth_l1_loss(pred_pts, target_pts, beta=0.1)
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        # Debug
        self._step += 1
        if self._step % 50 == 0:
            with torch.no_grad():
                pos_probs = probs[:, :, 1]
                pos_mask = cls_targets == 1
                neg_mask = cls_targets == 0
                pos_score = pos_probs[pos_mask].mean().item() if pos_mask.sum() > 0 else 0
                neg_score = pos_probs[neg_mask].mean().item() if neg_mask.sum() > 0 else 0
                print(f"  [Step {self._step}] cls={cls_loss.item():.4f} reg={reg_loss.item():.4f} | "
                      f"assigned={total_assigned}/{total_gt} | "
                      f"pos={pos_score:.3f} neg={neg_score:.3f} gap={pos_score-neg_score:.3f}")

        return total, {'cls_loss': cls_loss.item(), 'reg_loss': reg_loss.item(), 'n_assigned': total_assigned}


# =============================================================================
# EMA (Exponential Moving Average) for stable evaluation
# =============================================================================

class EMA:
    """Exponential Moving Average of model weights for stable evaluation."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._update_shadow()

    def _update_shadow(self):
        """Update shadow dict with current trainable parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()

    def update(self):
        # First, check for any new trainable parameters (e.g., after unfreeze)
        self._update_shadow()

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# NMS
# =============================================================================

def point_nms(points: torch.Tensor, scores: torch.Tensor, radius: float = 0.03) -> torch.Tensor:
    """
    Non-maximum suppression for points.

    Args:
        points: [N, 2] normalized coordinates
        scores: [N] confidence scores
        radius: Suppression radius (normalized)

    Returns:
        keep: Indices of points to keep
    """
    if len(points) == 0:
        return torch.tensor([], dtype=torch.long, device=points.device)

    # Sort by score
    order = scores.argsort(descending=True)
    points = points[order]
    scores = scores[order]

    keep = []
    suppressed = torch.zeros(len(points), dtype=torch.bool, device=points.device)

    for i in range(len(points)):
        if suppressed[i]:
            continue
        keep.append(order[i])

        # Suppress nearby points
        dists = (points[i+1:] - points[i]).pow(2).sum(dim=1).sqrt()
        suppressed[i+1:] |= dists < radius

    return torch.tensor(keep, dtype=torch.long, device=points.device)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, dataloader, device, image_size=512, threshold=0.5, match_radius=25, nms_radius=0.03, find_best_threshold=True):
    """Evaluate with optional threshold optimization."""
    model.eval()

    if not find_best_threshold:
        # Original fixed threshold evaluation
        total_tp, total_fp, total_fn = 0, 0, 0

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                outputs = model(images)

                probs = outputs['logits'].softmax(-1)[:, :, 1]
                points_norm = outputs['pred_points_normalized']

                for b in range(len(targets)):
                    gt_pts = targets[b]['points'].float().to(device)
                    if gt_pts.numel() > 0 and gt_pts.max() <= 1.0:
                        gt_pts = gt_pts * image_size

                    scores = probs[b]
                    mask = scores >= threshold
                    pred_scores = scores[mask]
                    pred_pts_norm = points_norm[b, mask]

                    if len(pred_pts_norm) > 0:
                        keep = point_nms(pred_pts_norm, pred_scores, radius=nms_radius)
                        pred_pts_norm = pred_pts_norm[keep]

                    pred_pts = pred_pts_norm * image_size
                    n_pred, n_gt = len(pred_pts), len(gt_pts)
                    matched_gt, matched_pred = set(), set()

                    if n_pred > 0 and n_gt > 0:
                        dists = torch.cdist(pred_pts, gt_pts)
                        flat = dists.flatten()
                        for idx in flat.argsort():
                            if flat[idx] > match_radius:
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

        return {'precision': precision, 'recall': recall, 'f1': f1,
                'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'threshold': threshold}

    # Quick threshold search for training
    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            probs = outputs['logits'].softmax(-1)[:, :, 1]
            points_norm = outputs['pred_points_normalized']

            for b in range(len(targets)):
                gt_pts = targets[b]['points'].float()
                if gt_pts.numel() > 0 and gt_pts.max() <= 1.0:
                    gt_pts = gt_pts * image_size

                all_preds.append((probs[b].cpu(), points_norm[b].cpu()))
                all_gts.append(gt_pts)

    best_f1, best_thresh, best_metrics = 0, 0.5, None

    # Quick search with fewer thresholds
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        total_tp, total_fp, total_fn = 0, 0, 0

        for (scores, pts_norm), gt_pts in zip(all_preds, all_gts):
            mask = scores >= thresh
            pred_scores = scores[mask]
            pred_pts_norm = pts_norm[mask]

            if len(pred_pts_norm) > 0:
                keep = point_nms(pred_pts_norm, pred_scores, radius=nms_radius)
                pred_pts_norm = pred_pts_norm[keep]

            pred_pts = pred_pts_norm * image_size
            n_pred, n_gt = len(pred_pts), len(gt_pts)
            matched_gt, matched_pred = set(), set()

            if n_pred > 0 and n_gt > 0:
                dists = torch.cdist(pred_pts, gt_pts)
                flat = dists.flatten()
                for idx in flat.argsort():
                    if flat[idx] > match_radius:
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

        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
            best_metrics = {'precision': precision, 'recall': recall, 'f1': f1,
                           'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'threshold': thresh}

    return best_metrics if best_metrics else {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'threshold': 0.5}


def evaluate_with_optimal_threshold(model, dataloader, device, image_size=512, match_radius=25, nms_radius=0.03):
    """Find optimal threshold."""
    model.eval()

    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            probs = outputs['logits'].softmax(-1)[:, :, 1]
            points_norm = outputs['pred_points_normalized']

            for b in range(len(targets)):
                gt_pts = targets[b]['points'].float()
                if gt_pts.numel() > 0 and gt_pts.max() <= 1.0:
                    gt_pts = gt_pts * image_size

                all_preds.append((probs[b].cpu(), points_norm[b].cpu()))
                all_gts.append(gt_pts)

    print("\nThreshold optimization:")
    print(f"{'Thresh':>7} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 32)

    best_f1, best_thresh, best_metrics = 0, 0.5, None

    for thresh in np.arange(0.1, 0.95, 0.05):
        total_tp, total_fp, total_fn = 0, 0, 0

        for (scores, pts_norm), gt_pts in zip(all_preds, all_gts):
            mask = scores >= thresh
            pred_scores = scores[mask]
            pred_pts_norm = pts_norm[mask]

            if len(pred_pts_norm) > 0:
                keep = point_nms(pred_pts_norm, pred_scores, radius=nms_radius)
                pred_pts_norm = pred_pts_norm[keep]

            pred_pts = pred_pts_norm * image_size
            n_pred, n_gt = len(pred_pts), len(gt_pts)
            matched_gt, matched_pred = set(), set()

            if n_pred > 0 and n_gt > 0:
                dists = torch.cdist(pred_pts, gt_pts)
                flat = dists.flatten()
                for idx in flat.argsort():
                    if flat[idx] > match_radius:
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

        marker = " *" if f1 > best_f1 else ""
        print(f"{thresh:>7.2f} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}{marker}")

        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
            best_metrics = {'precision': precision, 'recall': recall, 'f1': f1,
                           'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'threshold': thresh}

    print("-" * 32)
    print(f"Best: threshold={best_thresh:.2f} -> F1={best_f1:.3f}")
    return best_metrics


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, train_loader, val_loader, device, output_dir, config):
        self.model, self.criterion, self.optimizer, self.scheduler = model, criterion, optimizer, scheduler
        self.train_loader, self.val_loader, self.device = train_loader, val_loader, device
        self.output_dir, self.config = Path(output_dir), config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch, self.best_f1, self.patience_counter = 0, 0.0, 0

        # EMA for stable evaluation
        self.ema = EMA(model, decay=0.999)

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self):
        self.model.train()
        total_loss, total_cls, total_reg, n = 0, 0, 0, 0
        for images, targets in self.train_loader:
            images = images.to(self.device)
            self.optimizer.zero_grad()
            loss, loss_dict = self.criterion(self.model(images), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Tighter clipping
            self.optimizer.step()
            self.ema.update()  # Update EMA after each step
            total_loss += loss.item()
            total_cls += loss_dict['cls_loss']
            total_reg += loss_dict['reg_loss']
            n += 1
        return {'loss': total_loss/n, 'cls_loss': total_cls/n, 'reg_loss': total_reg/n}

    def save_checkpoint(self, name, use_ema=False):
        if use_ema:
            self.ema.apply_shadow()
        torch.save({'epoch': self.epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'best_f1': self.best_f1},
                   self.output_dir / f'{name}.pth')
        if use_ema:
            self.ema.restore()

    def train(self, epochs, patience=20, unfreeze_epoch=None):
        print(f"\nTraining for {epochs} epochs (with EMA for stability)")
        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            if unfreeze_epoch and epoch == unfreeze_epoch:
                print(f"\n*** Unfreezing backbone ***")
                self.model.unfreeze_backbone()
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= 0.1

            train_metrics = self.train_epoch()
            if self.scheduler:
                self.scheduler.step()

            val_metrics = {}
            if self.val_loader:
                # Use EMA weights for evaluation (more stable)
                self.ema.apply_shadow()

                # Try multiple thresholds, pick best
                best_metrics = None
                for thresh in [0.2, 0.3, 0.4, 0.5]:
                    metrics = evaluate(self.model, self.val_loader, self.device,
                                       self.config['image_size'], thresh,
                                       self.config['match_radius'], self.config.get('nms_radius', 0.03))
                    metrics['threshold'] = thresh
                    if best_metrics is None or metrics['f1'] > best_metrics['f1']:
                        best_metrics = metrics

                val_metrics = best_metrics
                self.ema.restore()

                if val_metrics['f1'] > self.best_f1:
                    self.best_f1 = val_metrics['f1']
                    self.patience_counter = 0
                    self.save_checkpoint('best', use_ema=True)
                else:
                    self.patience_counter += 1

            lr = self.optimizer.param_groups[0]['lr']
            log = f"Epoch {epoch:3d} ({time.time()-t0:.1f}s) | loss={train_metrics['loss']:.4f} cls={train_metrics['cls_loss']:.4f} reg={train_metrics['reg_loss']:.4f} | lr={lr:.6f}"
            if val_metrics:
                thresh = val_metrics.get('threshold', 0.5)
                log += f" | P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} F1={val_metrics['f1']:.3f} @{thresh:.1f}"
                if val_metrics['f1'] >= self.best_f1:
                    log += " *"
            print(log)

            if epoch % 10 == 0:
                self.save_checkpoint('latest', use_ema=True)

            if patience > 0 and self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.save_checkpoint('final', use_ema=True)
        print(f"\nDone. Best F1: {self.best_f1:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--train_image_dir', type=str, required=True)
    parser.add_argument('--val_csv', type=str, default=None)
    parser.add_argument('--val_image_dir', type=str, default=None)

    parser.add_argument('--backbone', type=str, default='vit_large_patch14_dinov2.lvd142m')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--unfreeze_epoch', type=int, default=None)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (lower for stability)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)

    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--reg_weight', type=float, default=5.0)
    parser.add_argument('--pos_weight', type=float, default=5.0, help='Base positive class weight')
    parser.add_argument('--assignment_radius', type=float, default=0.08, help='GT-to-prediction assignment radius')

    parser.add_argument('--use_object_aware_crop', action='store_true')
    parser.add_argument('--min_edge_distance', type=int, default=20)
    parser.add_argument('--empty_probability', type=float, default=0.1)

    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--eval_threshold', type=float, default=0.3, help='Confidence threshold (lower for dense pred)')
    parser.add_argument('--match_radius', type=float, default=25)
    parser.add_argument('--nms_radius', type=float, default=0.03)

    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = vars(args)

    # Data
    print("\nLoading data...")
    train_dataset = PointDataset(args.train_csv, args.train_image_dir, args.image_size, augment=True,
                                  use_object_aware_crop=args.use_object_aware_crop,
                                  min_edge_distance=args.min_edge_distance, empty_probability=args.empty_probability)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True)

    val_loader = None
    if args.val_csv and args.val_image_dir:
        val_dataset = PointDataset(args.val_csv, args.val_image_dir, args.image_size, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    print("\nBuilding model...")
    is_vit = 'vit' in args.backbone.lower() or 'dino' in args.backbone.lower()

    model = DenseP2PNet(
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        pretrained=True,
        freeze_backbone=args.freeze_backbone or is_vit,
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}, "
          f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss with focal + hard negative mining
    criterion = DensePointLoss(
        image_size=args.image_size,
        cls_weight=args.cls_weight,
        reg_weight=args.reg_weight,
        assignment_radius=args.assignment_radius,
        focal_alpha=0.75,
        focal_gamma=2.0,
        neg_pos_ratio=3,
    )

    # Optimizer with warmup
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Train
    trainer = Trainer(model, criterion, optimizer, scheduler, train_loader, val_loader, device, args.output_dir, config)
    trainer.train(epochs=args.epochs, patience=args.patience, unfreeze_epoch=args.unfreeze_epoch)

    # Final eval
    if val_loader:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        best_path = Path(args.output_dir) / 'best.pth'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
        metrics = evaluate_with_optimal_threshold(model, val_loader, device, args.image_size, args.match_radius, args.nms_radius)
        print(f"\nFINAL: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} @ thresh={metrics['threshold']:.2f}")


if __name__ == '__main__':
    main()