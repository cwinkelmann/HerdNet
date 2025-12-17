"""
Iguana Multi-Head Detector

Three heads on DINOv2 backbone:
1. Tile Classification: Is there an iguana in this crop? (CLS token)
2. Patch Detection: Which patches contain iguanas? (patch tokens → 37x37 heatmap)
3. Regression: How many iguanas + offset refinement (count + per-patch offset)

Architecture:
- DINOv2 backbone (frozen initially)
- Shared patch tokens (37x37 for 518x518 input with patch14)
- Three lightweight heads

For 518x518 input with patch_size=14:
- 37x37 = 1369 patches
- Each patch covers 14x14 pixels
- Localization resolution: ~14 pixels
"""

import os
import argparse
import random
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict

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
except ImportError:
    raise ImportError("albumentations required: pip install albumentations")


# =============================================================================
# DATASET: Returns crops with point locations for all three tasks
# =============================================================================

class IguanaMultiTaskDataset(Dataset):
    """
    Dataset for multi-task iguana detection.

    Returns:
        - image: [3, H, W] tensor
        - targets: dict with:
            - 'tile_label': 0 or 1 (is there an iguana?)
            - 'points': [N, 2] tensor of point coordinates in crop (0-1 normalized)
            - 'count': number of iguanas in crop
            - 'patch_labels': [grid_h, grid_w] binary mask of which patches have iguanas
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        crop_size: int = 518,
        patch_size: int = 14,
        crops_per_image: int = 8,
        positive_ratio: float = 0.5,
        min_edge_margin: int = 30,
        point_radius: int = 20,  # Radius around point to mark patches as positive
        augment: bool = True,
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.crops_per_image = crops_per_image
        self.positive_ratio = positive_ratio
        self.min_edge_margin = min_edge_margin
        self.point_radius = point_radius
        self.augment = augment

        # Grid size for patch tokens
        self.grid_size = crop_size // patch_size  # 37 for 518/14

        # Load annotations
        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        # Store points per image
        self.annotations = {}
        for name, group in self.df.groupby('images'):
            self.annotations[name] = group[['x', 'y']].values.astype(np.float32)

        # Normalization (ImageNet stats work well for DINOv2)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.photometric_transform = self._build_photometric() if augment else None
        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])

        self._image_sizes = {}

        total_points = sum(len(pts) for pts in self.annotations.values())
        print(f"IguanaMultiTaskDataset: {len(self.image_names)} images, {total_points} points")
        print(f"  Crop: {crop_size}, Patch: {patch_size}, Grid: {self.grid_size}x{self.grid_size}")
        print(f"  Positive ratio: {positive_ratio}, Point radius: {point_radius}px")

    def _build_photometric(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
        ])

    def _get_image_size(self, name: str) -> Tuple[int, int]:
        if name not in self._image_sizes:
            img_path = os.path.join(self.image_dir, name)
            with Image.open(img_path) as img:
                self._image_sizes[name] = img.size
        return self._image_sizes[name]

    def _sample_positive_crop(self, points: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int]:
        valid_points = []
        for pt in points:
            px, py = pt
            if (px >= self.min_edge_margin and px <= img_w - self.min_edge_margin and
                py >= self.min_edge_margin and py <= img_h - self.min_edge_margin):
                valid_points.append(pt)

        if not valid_points:
            valid_points = points.tolist()

        pt = random.choice(valid_points)
        px, py = pt

        x_min = max(0, int(px - self.crop_size + self.min_edge_margin))
        x_max = min(img_w - self.crop_size, int(px - self.min_edge_margin))
        y_min = max(0, int(py - self.crop_size + self.min_edge_margin))
        y_max = min(img_h - self.crop_size, int(py - self.min_edge_margin))

        x_min = min(x_min, max(0, img_w - self.crop_size))
        x_max = max(x_max, 0)
        y_min = min(y_min, max(0, img_h - self.crop_size))
        y_max = max(y_max, 0)

        crop_x = random.randint(min(x_min, x_max), max(x_min, x_max))
        crop_y = random.randint(min(y_min, y_max), max(y_min, y_max))

        return crop_x, crop_y

    def _sample_negative_crop(self, points: np.ndarray, img_w: int, img_h: int,
                               max_attempts: int = 50) -> Optional[Tuple[int, int]]:
        max_x = max(0, img_w - self.crop_size)
        max_y = max(0, img_h - self.crop_size)

        for _ in range(max_attempts):
            crop_x = random.randint(0, max_x) if max_x > 0 else 0
            crop_y = random.randint(0, max_y) if max_y > 0 else 0

            in_crop = (
                (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
                (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
            )

            if not in_crop.any():
                return crop_x, crop_y

        return None

    def _create_patch_labels(self, points_in_crop: np.ndarray) -> torch.Tensor:
        """
        Create binary patch labels.

        A patch is positive if any iguana point falls within it (+ radius).
        """
        patch_labels = torch.zeros(self.grid_size, self.grid_size)

        if len(points_in_crop) == 0:
            return patch_labels

        for pt in points_in_crop:
            px, py = pt  # In crop coordinates (0 to crop_size)

            # Convert to patch coordinates
            patch_x = int(px / self.patch_size)
            patch_y = int(py / self.patch_size)

            # Mark this patch and neighbors within radius
            radius_patches = max(1, self.point_radius // self.patch_size)

            for dy in range(-radius_patches, radius_patches + 1):
                for dx in range(-radius_patches, radius_patches + 1):
                    py_idx = patch_y + dy
                    px_idx = patch_x + dx

                    if 0 <= py_idx < self.grid_size and 0 <= px_idx < self.grid_size:
                        # Check actual distance
                        patch_center_x = (px_idx + 0.5) * self.patch_size
                        patch_center_y = (py_idx + 0.5) * self.patch_size
                        dist = np.sqrt((px - patch_center_x)**2 + (py - patch_center_y)**2)

                        if dist <= self.point_radius:
                            patch_labels[py_idx, px_idx] = 1.0

        return patch_labels

    def __len__(self):
        return len(self.image_names) * self.crops_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        name = self.image_names[img_idx]
        points = self.annotations[name]

        # Load image
        img_path = os.path.join(self.image_dir, name)
        img = np.array(Image.open(img_path).convert('RGB'))
        img_h, img_w = img.shape[:2]

        # Sample crop
        want_positive = random.random() < self.positive_ratio

        if want_positive and len(points) > 0:
            crop_x, crop_y = self._sample_positive_crop(points, img_w, img_h)
        else:
            result = self._sample_negative_crop(points, img_w, img_h)
            if result is not None:
                crop_x, crop_y = result
            elif len(points) > 0:
                crop_x, crop_y = self._sample_positive_crop(points, img_w, img_h)
            else:
                max_x = max(0, img_w - self.crop_size)
                max_y = max(0, img_h - self.crop_size)
                crop_x = random.randint(0, max_x) if max_x > 0 else 0
                crop_y = random.randint(0, max_y) if max_y > 0 else 0

        # Extract crop
        crop = img[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]

        if crop.shape[0] < self.crop_size or crop.shape[1] < self.crop_size:
            padded = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        # Find points in crop
        points_in_crop = []
        if len(points) > 0:
            for pt in points:
                px, py = pt
                if (crop_x <= px < crop_x + self.crop_size and
                    crop_y <= py < crop_y + self.crop_size):
                    points_in_crop.append([px - crop_x, py - crop_y])

        points_in_crop = np.array(points_in_crop, dtype=np.float32) if points_in_crop else np.zeros((0, 2), dtype=np.float32)

        # Create targets
        tile_label = 1.0 if len(points_in_crop) > 0 else 0.0
        count = len(points_in_crop)
        patch_labels = self._create_patch_labels(points_in_crop)

        # Normalize points to 0-1
        points_normalized = points_in_crop / self.crop_size if len(points_in_crop) > 0 else points_in_crop

        # Apply augmentations
        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']

        crop = self.normalize_transform(image=crop)['image']

        return crop, {
            'tile_label': torch.tensor(tile_label, dtype=torch.float32),
            'points': torch.from_numpy(points_normalized).float(),
            'count': torch.tensor(count, dtype=torch.float32),
            'patch_labels': patch_labels,
            'name': name,
        }


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {
        'tile_label': torch.stack([b[1]['tile_label'] for b in batch]),
        'count': torch.stack([b[1]['count'] for b in batch]),
        'patch_labels': torch.stack([b[1]['patch_labels'] for b in batch]),
        'points': [b[1]['points'] for b in batch],  # Variable length
        'name': [b[1]['name'] for b in batch],
    }
    return images, targets


# =============================================================================
# MODEL: Multi-head detector
# =============================================================================

class IguanaMultiHead(nn.Module):
    """
    Multi-head iguana detector using DINOv2 backbone.

    Heads:
    1. Tile Classification (CLS token → binary)
    2. Patch Detection (patch tokens → grid_size x grid_size heatmap)
    3. Regression (CLS token → count, patch tokens → offset)
    """

    def __init__(
        self,
        backbone: str = 'vit_small_patch14_dinov2.lvd142m',
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size[0]

        # Get number of prefix tokens (CLS + registers)
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)

        print(f"Backbone: {backbone}")
        print(f"  Feature dim: {self.feat_dim}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Prefix tokens: {self.num_prefix_tokens}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen")

        # === Head 1: Tile Classification (uses CLS token) ===
        self.tile_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # === Head 2: Patch Detection (uses patch tokens) ===
        self.patch_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # === Head 3: Regression ===
        # Count prediction (from CLS token)
        self.count_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Offset prediction (per patch, predicts offset to exact point location)
        self.offset_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # dx, dy offset within patch
        )

        self._init_heads()

    def _init_heads(self):
        for head in [self.tile_head, self.patch_head, self.count_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        grid_size = H // self.patch_size

        # Get all tokens from backbone
        features = self.backbone.forward_features(x)  # [B, num_tokens, feat_dim]

        # Split into CLS and patch tokens
        cls_token = features[:, 0]  # [B, feat_dim]
        patch_tokens = features[:, self.num_prefix_tokens:]  # [B, grid_size*grid_size, feat_dim]

        # Reshape patch tokens to grid
        patch_tokens = patch_tokens.view(B, grid_size, grid_size, self.feat_dim)

        # === Head 1: Tile classification ===
        tile_logits = self.tile_head(cls_token).squeeze(-1)  # [B]

        # === Head 2: Patch detection ===
        patch_logits = self.patch_head(patch_tokens).squeeze(-1)  # [B, grid_size, grid_size]

        # === Head 3: Regression ===
        count_pred = self.count_head(cls_token).squeeze(-1)  # [B]
        count_pred = F.softplus(count_pred)  # Ensure non-negative

        offset_pred = self.offset_head(patch_tokens)  # [B, grid_size, grid_size, 2]
        offset_pred = torch.tanh(offset_pred) * 0.5  # Offset within [-0.5, 0.5] of patch

        return {
            'tile_logits': tile_logits,
            'patch_logits': patch_logits,
            'count': count_pred,
            'offsets': offset_pred,
        }

    def unfreeze_backbone(self, n_blocks: Optional[int] = None):
        if n_blocks is None:
            print("Unfreezing entire backbone")
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            if hasattr(self.backbone, 'blocks'):
                total = len(self.backbone.blocks)
                print(f"Unfreezing last {n_blocks} of {total} blocks")
                for i, block in enumerate(self.backbone.blocks):
                    if i >= total - n_blocks:
                        for p in block.parameters():
                            p.requires_grad = True


# =============================================================================
# LOSS
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for all three heads.
    """

    def __init__(
        self,
        tile_weight: float = 1.0,
        patch_weight: float = 1.0,
        count_weight: float = 0.5,
        offset_weight: float = 1.0,
        patch_pos_weight: float = 10.0,
    ):
        super().__init__()
        self.tile_weight = tile_weight
        self.patch_weight = patch_weight
        self.count_weight = count_weight
        self.offset_weight = offset_weight
        self.patch_pos_weight = patch_pos_weight

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:

        device = outputs['tile_logits'].device
        B = outputs['tile_logits'].shape[0]

        # === Tile classification loss ===
        tile_loss = F.binary_cross_entropy_with_logits(
            outputs['tile_logits'],
            targets['tile_label'].to(device)
        )

        # === Patch detection loss (focal-style to handle imbalance) ===
        patch_logits = outputs['patch_logits']
        patch_labels = targets['patch_labels'].to(device)

        # Weighted BCE for patch detection
        pos_weight = torch.tensor(self.patch_pos_weight, device=device)
        patch_loss = F.binary_cross_entropy_with_logits(
            patch_logits.view(B, -1),
            patch_labels.view(B, -1),
            pos_weight=pos_weight
        )

        # === Count regression loss ===
        count_loss = F.smooth_l1_loss(
            outputs['count'],
            targets['count'].to(device)
        )

        # === Offset loss (only for positive patches) ===
        offset_loss = torch.tensor(0.0, device=device)
        n_offset_samples = 0

        grid_size = patch_logits.shape[1]
        patch_size = 518 // grid_size  # Assuming 518 input

        for b in range(B):
            points = targets['points'][b].to(device)
            if len(points) == 0:
                continue

            for pt in points:
                px, py = pt[0].item() * 518, pt[1].item() * 518  # Unnormalize

                patch_x = int(px / patch_size)
                patch_y = int(py / patch_size)

                if 0 <= patch_x < grid_size and 0 <= patch_y < grid_size:
                    # Target offset: distance from patch center to point
                    patch_center_x = (patch_x + 0.5) * patch_size
                    patch_center_y = (patch_y + 0.5) * patch_size

                    target_dx = (px - patch_center_x) / patch_size
                    target_dy = (py - patch_center_y) / patch_size

                    pred_offset = outputs['offsets'][b, patch_y, patch_x]
                    target_offset = torch.tensor([target_dx, target_dy], device=device)

                    offset_loss += F.smooth_l1_loss(pred_offset, target_offset)
                    n_offset_samples += 1

        if n_offset_samples > 0:
            offset_loss = offset_loss / n_offset_samples

        # === Total loss ===
        total_loss = (
            self.tile_weight * tile_loss +
            self.patch_weight * patch_loss +
            self.count_weight * count_loss +
            self.offset_weight * offset_loss
        )

        return total_loss, {
            'tile_loss': tile_loss.item(),
            'patch_loss': patch_loss.item(),
            'count_loss': count_loss.item(),
            'offset_loss': offset_loss.item() if isinstance(offset_loss, torch.Tensor) else offset_loss,
        }


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch: int = 0):
    model.train()
    total_loss = 0
    loss_components = {'tile_loss': 0, 'patch_loss': 0, 'count_loss': 0, 'offset_loss': 0}
    n_batches = 0

    # Track tile accuracy
    tile_correct = 0
    tile_total = 0

    for images, targets in loader:
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss, components = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
        n_batches += 1

        # Track tile accuracy
        tile_preds = (torch.sigmoid(outputs['tile_logits']) > 0.5).float()
        tile_correct += (tile_preds.cpu() == targets['tile_label']).sum().item()
        tile_total += len(targets['tile_label'])

    return {
        'loss': total_loss / n_batches,
        'tile_acc': tile_correct / tile_total,
        **{k: v / n_batches for k, v in loss_components.items()}
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()

    total_loss = 0
    loss_components = {'tile_loss': 0, 'patch_loss': 0, 'count_loss': 0, 'offset_loss': 0}
    n_batches = 0

    # Tile metrics
    tile_tp, tile_fp, tile_fn, tile_tn = 0, 0, 0, 0

    # Patch metrics
    patch_tp, patch_fp, patch_fn = 0, 0, 0

    # Count error
    count_errors = []

    # Point detection metrics (using patch predictions + offsets)
    point_tp, point_fp, point_fn = 0, 0, 0

    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)

        loss, components = criterion(outputs, targets)
        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
        n_batches += 1

        B = images.shape[0]
        grid_size = outputs['patch_logits'].shape[1]

        for b in range(B):
            # Tile classification
            tile_pred = torch.sigmoid(outputs['tile_logits'][b]).item() > threshold
            tile_true = targets['tile_label'][b].item() > 0.5

            if tile_pred and tile_true:
                tile_tp += 1
            elif tile_pred and not tile_true:
                tile_fp += 1
            elif not tile_pred and tile_true:
                tile_fn += 1
            else:
                tile_tn += 1

            # Count error
            count_pred = outputs['count'][b].item()
            count_true = targets['count'][b].item()
            count_errors.append(abs(count_pred - count_true))

            # Patch detection
            patch_preds = torch.sigmoid(outputs['patch_logits'][b]) > threshold
            patch_true = targets['patch_labels'][b].to(device) > 0.5

            patch_tp += ((patch_preds == 1) & (patch_true == 1)).sum().item()
            patch_fp += ((patch_preds == 1) & (patch_true == 0)).sum().item()
            patch_fn += ((patch_preds == 0) & (patch_true == 1)).sum().item()

            # Point detection (extract points from positive patches)
            gt_points = targets['points'][b]  # [N, 2] normalized

            if patch_preds.any():
                # Get predicted points
                pred_points = []
                patch_probs = torch.sigmoid(outputs['patch_logits'][b])
                offsets = outputs['offsets'][b]

                for py in range(grid_size):
                    for px in range(grid_size):
                        if patch_preds[py, px]:
                            # Patch center + offset
                            cx = (px + 0.5 + offsets[py, px, 0].item()) / grid_size
                            cy = (py + 0.5 + offsets[py, px, 1].item()) / grid_size
                            pred_points.append([cx, cy, patch_probs[py, px].item()])

                pred_points = torch.tensor(pred_points) if pred_points else torch.zeros(0, 3)

                # Match predictions to ground truth
                if len(gt_points) > 0 and len(pred_points) > 0:
                    pred_xy = pred_points[:, :2]
                    dists = torch.cdist(pred_xy, gt_points.to(pred_xy.device))

                    match_radius = 0.05  # 5% of image = ~26 pixels
                    matched_gt = set()
                    matched_pred = set()

                    for idx in dists.flatten().argsort():
                        if dists.flatten()[idx] > match_radius:
                            break
                        pi, gi = (idx // len(gt_points)).item(), (idx % len(gt_points)).item()
                        if pi not in matched_pred and gi not in matched_gt:
                            matched_pred.add(pi)
                            matched_gt.add(gi)

                    point_tp += len(matched_pred)
                    point_fp += len(pred_points) - len(matched_pred)
                    point_fn += len(gt_points) - len(matched_gt)
                elif len(pred_points) > 0:
                    point_fp += len(pred_points)
                elif len(gt_points) > 0:
                    point_fn += len(gt_points)
            elif len(gt_points) > 0:
                point_fn += len(gt_points)

    # Compute metrics
    tile_precision = tile_tp / max(tile_tp + tile_fp, 1)
    tile_recall = tile_tp / max(tile_tp + tile_fn, 1)
    tile_f1 = 2 * tile_precision * tile_recall / max(tile_precision + tile_recall, 1e-6)

    patch_precision = patch_tp / max(patch_tp + patch_fp, 1)
    patch_recall = patch_tp / max(patch_tp + patch_fn, 1)
    patch_f1 = 2 * patch_precision * patch_recall / max(patch_precision + patch_recall, 1e-6)

    point_precision = point_tp / max(point_tp + point_fp, 1)
    point_recall = point_tp / max(point_tp + point_fn, 1)
    point_f1 = 2 * point_precision * point_recall / max(point_precision + point_recall, 1e-6)

    return {
        'loss': total_loss / n_batches,
        **{k: v / n_batches for k, v in loss_components.items()},
        'tile_precision': tile_precision,
        'tile_recall': tile_recall,
        'tile_f1': tile_f1,
        'patch_precision': patch_precision,
        'patch_recall': patch_recall,
        'patch_f1': patch_f1,
        'point_precision': point_precision,
        'point_recall': point_recall,
        'point_f1': point_f1,
        'count_mae': np.mean(count_errors),
    }


def visualize_predictions(model, dataset, device, output_dir: Path, n_samples: int = 20):
    """Visualize model predictions."""
    import matplotlib.pyplot as plt

    model.eval()
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for i, idx in enumerate(indices):
        img_tensor, targets = dataset[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array(dataset.std) + np.array(dataset.mean)
        img = np.clip(img, 0, 1)

        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0).to(device))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image with GT points
        axes[0].imshow(img)
        gt_points = targets['points'].numpy() * dataset.crop_size
        for pt in gt_points:
            axes[0].plot(pt[0], pt[1], 'go', markersize=10, markeredgewidth=2, markerfacecolor='none')
        axes[0].set_title(f"GT: {len(gt_points)} iguanas")
        axes[0].axis('off')

        # Patch heatmap
        patch_probs = torch.sigmoid(outputs['patch_logits'][0]).cpu().numpy()
        axes[1].imshow(patch_probs, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f"Patch probs (max={patch_probs.max():.2f})")
        axes[1].axis('off')

        # Predicted points overlay
        axes[2].imshow(img)

        tile_prob = torch.sigmoid(outputs['tile_logits'][0]).item()
        count_pred = outputs['count'][0].item()

        grid_size = patch_probs.shape[0]
        offsets = outputs['offsets'][0].cpu().numpy()

        for py in range(grid_size):
            for px in range(grid_size):
                if patch_probs[py, px] > 0.5:
                    cx = (px + 0.5 + offsets[py, px, 0]) * (dataset.crop_size / grid_size)
                    cy = (py + 0.5 + offsets[py, px, 1]) * (dataset.crop_size / grid_size)
                    axes[2].plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)

        axes[2].set_title(f"Pred: tile={tile_prob:.2f}, count={count_pred:.1f}")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(vis_dir / f'{i:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(indices)} visualizations to {vis_dir}")


def main(just_eval=False):
    parser = argparse.ArgumentParser(description="Iguana Multi-Head Detector")

    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')

    parser.add_argument('--backbone', default='vit_small_patch14_dinov2.lvd142m')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--crops_per_image', type=int, default=16)
    parser.add_argument('--positive_ratio', type=float, default=0.5)
    parser.add_argument('--point_radius', type=int, default=20)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--unfreeze_epoch', type=int, default=10)
    parser.add_argument('--unfreeze_blocks', type=int, default=4)

    parser.add_argument('--output_dir', default='./outputs_multihead')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_ds = IguanaMultiTaskDataset(
        args.train_csv, args.train_image_dir,
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        positive_ratio=args.positive_ratio,
        point_radius=args.point_radius,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
    )

    val_loader = None
    val_ds = None
    if args.val_csv and args.val_image_dir:
        val_ds = IguanaMultiTaskDataset(
            args.val_csv, args.val_image_dir,
            crop_size=args.crop_size,
            crops_per_image=args.crops_per_image,
            positive_ratio=0.5,
            point_radius=args.point_radius,
            augment=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn,
        )

    # Model
    model = IguanaMultiHead(
        backbone=args.backbone,
        freeze_backbone=True,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}, Trainable: {n_train:,}")

    criterion = MultiTaskLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training
    best_f1 = 0
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    if not just_eval:
        for epoch in range(args.epochs):
            if epoch == args.unfreeze_epoch and args.unfreeze_blocks > 0:
                print(f"\n*** Unfreezing last {args.unfreeze_blocks} backbone blocks ***")
                model.unfreeze_backbone(args.unfreeze_blocks)

                optimizer = torch.optim.AdamW([
                    {'params': [p for n, p in model.named_parameters()
                               if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.01},
                    {'params': [p for n, p in model.named_parameters()
                               if 'backbone' not in n and p.requires_grad], 'lr': args.lr * 0.1},
                ], weight_decay=args.weight_decay)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch, eta_min=1e-7
                )

            t0 = time.time()
            train_m = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            scheduler.step()

            log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | "
            log += f"loss={train_m['loss']:.4f} tile_acc={train_m['tile_acc']:.3f}"

            if val_loader:
                val_m = evaluate(model, val_loader, criterion, device)
                log += f" | tile_F1={val_m['tile_f1']:.3f} patch_F1={val_m['patch_f1']:.3f} "
                log += f"point_F1={val_m['point_f1']:.3f} MAE={val_m['count_mae']:.2f}"

                # Use point F1 as main metric
                if val_m['point_f1'] > best_f1:
                    best_f1 = val_m['point_f1']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_f1': best_f1,
                    }, output_dir / 'best.pth')
                    log += " ★"

            print(log)

        print("\n" + "=" * 80)
        print(f"Training complete! Best Point F1: {best_f1:.4f}")
        print("=" * 80)

    # Final evaluation & visualization
    if val_loader and val_ds:
        best_path = output_dir / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

        print("\nFinal evaluation:")
        val_m = evaluate(model, val_loader, criterion, device)
        print(f"  Tile:  P={val_m['tile_precision']:.3f} R={val_m['tile_recall']:.3f} F1={val_m['tile_f1']:.3f}")
        print(f"  Patch: P={val_m['patch_precision']:.3f} R={val_m['patch_recall']:.3f} F1={val_m['patch_f1']:.3f}")
        print(f"  Point: P={val_m['point_precision']:.3f} R={val_m['point_recall']:.3f} F1={val_m['point_f1']:.3f}")
        print(f"  Count MAE: {val_m['count_mae']:.2f}")

        visualize_predictions(model, val_ds, device, output_dir)


if __name__ == '__main__':
    main(just_eval=True)

