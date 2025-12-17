"""
Iguana Presence Classifier

Simple binary classifier: Does this tile contain an iguana?

Approach:
1. Random crops from full images
2. Label as positive if crop contains any iguana point, negative otherwise
3. Use DINOv3 backbone with frozen features + simple classification head
4. Once this works well, localization becomes trivial via patch embeddings

Expected: >95% accuracy on single-iguana tiles should be achievable.
"""

import os
import argparse
import random
import time
from pathlib import Path
from typing import Tuple, List, Optional

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
    raise ImportError("albumentations required: pip install albumentations")


# =============================================================================
# DATASET: Deterministic tiled validation dataset
# =============================================================================

class IguanaTiledDataset(Dataset):
    """
    Deterministic dataset that tiles full images into non-overlapping crops.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        crop_size: int = 518,
        overlap: int = 0,
        augment: bool = False,
        patch_size: int = 14,
        point_radius: int = 1,
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.overlap = overlap
        self.stride = crop_size - overlap
        self.augment = augment
        self.patch_size = patch_size
        self.point_radius = point_radius
        self.grid_size = crop_size // patch_size

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        self.annotations = {}
        for name, group in self.df.groupby('images'):
            self.annotations[name] = group[['x', 'y']].values.astype(np.float32)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.photometric_transform = self._build_photometric() if augment else None
        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])

        self.tiles = []
        self._image_sizes = {}

        for name in self.image_names:
            img_path = os.path.join(self.image_dir, name)
            with Image.open(img_path) as img:
                w, h = img.size
                self._image_sizes[name] = (w, h)

            pad_w = (self.stride - (w % self.stride)) % self.stride if w % self.stride != 0 else 0
            pad_h = (self.stride - (h % self.stride)) % self.stride if h % self.stride != 0 else 0
            padded_w = w + pad_w
            padded_h = h + pad_h

            n_tiles_x = max(1, (padded_w - self.overlap) // self.stride)
            n_tiles_y = max(1, (padded_h - self.overlap) // self.stride)

            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    crop_x = tx * self.stride
                    crop_y = ty * self.stride
                    self.tiles.append((name, crop_x, crop_y))

        total_points = sum(len(pts) for pts in self.annotations.values())
        print(f"IguanaTiledDataset: {len(self.image_names)} images, {total_points} points")
        print(f"  Crop size: {crop_size}, Stride: {self.stride}, Overlap: {overlap}")
        print(f"  Total tiles: {len(self.tiles)}")

        n_pos = sum(1 for name, cx, cy in self.tiles if self._has_points_in_tile(name, cx, cy))
        print(f"  Positive tiles: {n_pos}, Negative tiles: {len(self.tiles) - n_pos}")
        print(f"  Augmentation: {augment}")

    def _build_photometric(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64,
                            min_holes=1, min_height=16, min_width=16, fill_value=0, p=0.3),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1,
                           num_shadows_upper=3, shadow_dimension=5, p=0.2),
        ])

    def _has_points_in_tile(self, name: str, crop_x: int, crop_y: int) -> bool:
        points = self.annotations[name]
        if len(points) == 0:
            return False
        in_crop = (
            (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
            (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
        )
        return in_crop.any()

    def _get_points_in_tile(self, name: str, crop_x: int, crop_y: int) -> np.ndarray:
        points = self.annotations[name]
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        in_crop = (
            (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
            (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
        )
        points_in_crop = points[in_crop].copy()
        points_in_crop[:, 0] -= crop_x
        points_in_crop[:, 1] -= crop_y
        return points_in_crop

    def _create_patch_labels(self, points_in_crop: np.ndarray) -> torch.Tensor:
        patch_labels = torch.zeros(self.grid_size, self.grid_size)
        if len(points_in_crop) == 0:
            return patch_labels
        for pt in points_in_crop:
            px, py = pt
            patch_x = int(px / self.patch_size)
            patch_y = int(py / self.patch_size)
            patch_x = max(0, min(patch_x, self.grid_size - 1))
            patch_y = max(0, min(patch_y, self.grid_size - 1))
            for dy in range(-self.point_radius + 1, self.point_radius):
                for dx in range(-self.point_radius + 1, self.point_radius):
                    py_idx = patch_y + dy
                    px_idx = patch_x + dx
                    if 0 <= py_idx < self.grid_size and 0 <= px_idx < self.grid_size:
                        patch_labels[py_idx, px_idx] = 1.0
        return patch_labels

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        name, crop_x, crop_y = self.tiles[idx]
        img_path = os.path.join(self.image_dir, name)
        img = np.array(Image.open(img_path).convert('RGB'))
        img_h, img_w = img.shape[:2]

        pad_right = max(0, crop_x + self.crop_size - img_w)
        pad_bottom = max(0, crop_y + self.crop_size - img_h)
        if pad_right > 0 or pad_bottom > 0:
            img = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)

        crop = img[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]
        points_in_crop = self._get_points_in_tile(name, crop_x, crop_y)
        label = 1.0 if len(points_in_crop) > 0 else 0.0
        patch_labels = self._create_patch_labels(points_in_crop)

        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']
        crop = self.normalize_transform(image=crop)['image']

        return crop, {
            'label': torch.tensor(label, dtype=torch.float32),
            'patch_labels': patch_labels,
            'points': torch.from_numpy(points_in_crop).float(),
            'name': name,
            'crop_x': crop_x,
            'crop_y': crop_y,
        }


def tiled_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {
        'label': torch.stack([b[1]['label'] for b in batch]),
        'patch_labels': torch.stack([b[1]['patch_labels'] for b in batch]),
        'points': [b[1]['points'] for b in batch],
        'name': [b[1]['name'] for b in batch],
        'crop_x': [b[1]['crop_x'] for b in batch],
        'crop_y': [b[1]['crop_y'] for b in batch],
    }
    return images, targets


# =============================================================================
# DATASET: Random crops with presence labels (for training)
# =============================================================================

"""
Updated IguanaPresenceDataset with uniform iguana placement.

Key change: _sample_positive_crop now places iguanas anywhere in the crop
(edge, corner, center) rather than forcing them to be centered with a margin.
This matches tiled evaluation distribution.
"""

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


class IguanaPresenceDataset(Dataset):
    """
    Dataset that extracts random crops and labels them by iguana presence.

    Iguanas can appear anywhere in positive crops (edge, corner, center).
    """

    def __init__(
            self,
            csv_path: str,
            image_dir: str,
            crop_size: int = 518,
            crops_per_image: int = 8,
            positive_ratio: float = 0.5,
            augment: bool = True,
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.positive_ratio = positive_ratio
        self.augment = augment

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        self.annotations = {}
        for name, group in self.df.groupby('images'):
            self.annotations[name] = group[['x', 'y']].values.astype(np.float32)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.photometric_transform = self._build_photometric() if augment else None
        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])

        self._image_sizes = {}
        total_points = sum(len(pts) for pts in self.annotations.values())
        print(f"IguanaPresenceDataset: {len(self.image_names)} images")
        print(f"  Crop size: {crop_size}, Crops per image: {crops_per_image}")
        print(f"  Positive ratio: {positive_ratio}")
        print(f"  Total crops per epoch: {len(self)}")
        print(f"  Total annotations: {total_points}")

    def _build_photometric(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64,
                            min_holes=1, min_height=16, min_width=16, fill_value=0, p=0.3),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1,
                           num_shadows_upper=3, shadow_dimension=5, p=0.2),
        ])

    def _get_image_size(self, name: str) -> Tuple[int, int]:
        if name not in self._image_sizes:
            img_path = os.path.join(self.image_dir, name)
            with Image.open(img_path) as img:
                self._image_sizes[name] = img.size
        return self._image_sizes[name]

    def _sample_positive_crop(self, points: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int]:
        """Sample a crop that contains at least one point - point can be anywhere in crop."""
        # Pick a random point
        pt = random.choice(points)
        px, py = pt

        # Calculate valid crop range that keeps point inside the crop (anywhere, not just centered)
        # Point at (px, py) is inside crop if: crop_x <= px < crop_x + crop_size
        # So: px - crop_size < crop_x <= px
        # Clamped to image bounds: max(0, px - crop_size + 1) <= crop_x <= min(img_w - crop_size, px)

        x_min = max(0, int(px - self.crop_size + 1))
        x_max = min(img_w - self.crop_size, int(px))
        y_min = max(0, int(py - self.crop_size + 1))
        y_max = min(img_h - self.crop_size, int(py))

        # Handle edge cases where image is smaller than crop
        x_min = min(x_min, max(0, img_w - self.crop_size))
        x_max = max(x_max, 0)
        y_min = min(y_min, max(0, img_h - self.crop_size))
        y_max = max(y_max, 0)

        crop_x = random.randint(min(x_min, x_max), max(x_min, x_max))
        crop_y = random.randint(min(y_min, y_max), max(y_min, y_max))

        return crop_x, crop_y

    def _sample_negative_crop(self, points: np.ndarray, img_w: int, img_h: int, max_attempts: int = 50):
        """Sample a crop that contains no points."""
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

    def __len__(self):
        return len(self.image_names) * self.crops_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        name = self.image_names[img_idx]
        points = self.annotations[name]

        img_path = os.path.join(self.image_dir, name)
        img = np.array(Image.open(img_path).convert('RGB'))
        img_h, img_w = img.shape[:2]

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

        crop = img[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]
        if crop.shape[0] < self.crop_size or crop.shape[1] < self.crop_size:
            padded = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        # Verify label by checking actual point presence
        label = 0.0
        if len(points) > 0:
            in_crop = (
                    (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
                    (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
            )
            label = 1.0 if in_crop.any() else 0.0

        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']
        crop = self.normalize_transform(image=crop)['image']

        return crop, torch.tensor(label, dtype=torch.float32)


# =============================================================================
# MODEL
# =============================================================================

class IguanaClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = 'vit_large_patch14_reg4_dinov2.lvd142m',
        freeze_backbone: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)

        print(f"Backbone: {backbone}")
        print(f"  Feature dim: {self.feat_dim}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Prefix tokens: {self.num_prefix_tokens}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen")

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.patch_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        for head in [self.head, self.patch_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_patches: bool = False, upsample_patches: bool = False):
        B, C, H, W = x.shape
        grid_size = H // self.patch_size

        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.head(cls_token).squeeze(-1)

        if return_patches:
            patch_tokens = features[:, self.num_prefix_tokens:]
            patch_tokens = patch_tokens.view(B, grid_size, grid_size, self.feat_dim)
            patch_logits = self.patch_head(patch_tokens).squeeze(-1)
            if upsample_patches:
                patch_logits = F.interpolate(patch_logits.unsqueeze(1), size=(H, W),
                                             mode='bilinear', align_corners=False).squeeze(1)
            return logits, patch_logits
        return logits

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
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, epoch: int = 0, is_tiled: bool = False,
                patch_loss_weight: float = 1.0, pos_weight: float = 1.0):
    model.train()
    total_loss = total_tile_loss = total_patch_loss = 0
    total_correct = total_samples = 0
    tile_pos_weight = torch.tensor([pos_weight], device=device)

    pos_correct = pos_total = neg_correct = neg_total = 0
    all_probs = []
    all_labels_list = []

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        if is_tiled:
            labels = targets['label'].to(device)
            patch_labels = targets['patch_labels'].to(device) if patch_loss_weight > 0 else None
        else:
            labels = targets.to(device)
            patch_labels = None

        optimizer.zero_grad()

        if patch_labels is not None and patch_loss_weight > 0:
            tile_logits, patch_logits = model(images, return_patches=True)
            tile_loss = F.binary_cross_entropy_with_logits(tile_logits, labels,
                                                           pos_weight=tile_pos_weight.expand_as(tile_logits))
            n_pos_patches = patch_labels.sum()
            n_neg_patches = patch_labels.numel() - n_pos_patches
            pw = n_neg_patches / (n_pos_patches + 1e-6)
            pw = torch.clamp(torch.tensor(pw), 1.0, 50.0).to(device)
            patch_loss = F.binary_cross_entropy_with_logits(patch_logits, patch_labels,
                                                            pos_weight=pw.expand_as(patch_logits))
            loss = tile_loss + patch_loss_weight * patch_loss
            total_tile_loss += tile_loss.item() * len(labels)
            total_patch_loss += patch_loss.item() * len(labels)
            logits = tile_logits
        else:
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels,
                                                       pos_weight=tile_pos_weight.expand_as(logits))
            total_tile_loss += loss.item() * len(labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_correct += (preds[pos_mask] == labels[pos_mask]).sum().item()
        pos_total += pos_mask.sum().item()
        neg_correct += (preds[neg_mask] == labels[neg_mask]).sum().item()
        neg_total += neg_mask.sum().item()

        all_probs.extend(probs.detach().cpu().numpy())
        all_labels_list.extend(labels.cpu().numpy())

    if epoch == 0:
        all_probs = np.array(all_probs)
        all_labels_arr = np.array(all_labels_list)
        print(f"\n  [Epoch 0 Debug] Label distribution: {pos_total} pos, {neg_total} neg "
              f"({100*pos_total/(pos_total+neg_total):.1f}% positive)")
        print(f"  [Epoch 0 Debug] Prob distribution: min={all_probs.min():.3f}, "
              f"max={all_probs.max():.3f}, mean={all_probs.mean():.3f}")
        print(f"  [Epoch 0 Debug] Pos probs: mean={all_probs[all_labels_arr==1].mean():.3f}" if pos_total > 0 else "")
        print(f"  [Epoch 0 Debug] Neg probs: mean={all_probs[all_labels_arr==0].mean():.3f}" if neg_total > 0 else "")

    return {
        'loss': total_loss / total_samples,
        'tile_loss': total_tile_loss / total_samples,
        'patch_loss': total_patch_loss / total_samples if total_patch_loss > 0 else 0,
        'acc': total_correct / total_samples,
        'pos_acc': pos_correct / max(pos_total, 1),
        'neg_acc': neg_correct / max(neg_total, 1),
        'pos_total': pos_total,
        'neg_total': neg_total,
    }


@torch.no_grad()
def evaluate(model, loader, device, is_tiled: bool = False, threshold: float = 0.5):
    model.eval()
    total_loss = total_samples = 0
    all_preds, all_labels, all_probs = [], [], []

    for images, targets in loader:
        images = images.to(device)
        if is_tiled:
            labels = targets['label'].to(device)
        else:
            labels = targets.to(device)

        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    f3 = 10 * precision * recall / max(9 * precision + recall, 1e-6)

    return {
        'loss': total_loss / total_samples, 'acc': accuracy,
        'precision': precision, 'recall': recall, 'f1': f1, 'f3': f3,
        'all_probs': all_probs, 'all_labels': all_labels,
    }


def find_optimal_threshold(all_probs, all_labels, beta: float = 3.0):
    best_threshold, best_fbeta = 0.5, 0
    for thresh in np.arange(0.05, 0.95, 0.025):
        preds = (all_probs > thresh).astype(float)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        beta_sq = beta ** 2
        fbeta = (1 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-6)
        if fbeta > best_fbeta:
            best_fbeta, best_threshold = fbeta, thresh
    return best_threshold, best_fbeta


def main():
    parser = argparse.ArgumentParser(description="Iguana Presence Classifier")
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')
    parser.add_argument('--backbone', default='vit_large_patch14_reg4_dinov2.lvd142m')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--crops_per_image', type=int, default=8)
    parser.add_argument('--positive_ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--unfreeze_epoch', type=int, default=10)
    parser.add_argument('--unfreeze_blocks', type=int, default=4)
    parser.add_argument('--pos_weight', type=float, default=3.0)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output_dir', default='./outputs_classifier')
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tiled_train', action='store_true')
    parser.add_argument('--tiled_val', action='store_true')
    parser.add_argument('--tile_overlap', type=int, default=0)
    parser.add_argument('--patch_loss_weight', type=float, default=1.0)
    parser.add_argument('--load_from', type=str, default=None,
                        help='Path to checkpoint to load model weights from (no optimizer/scheduler state)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (includes optimizer/scheduler)')
    parser.add_argument('--early_stopping', type=int, default=15)
    parser.add_argument('--min_epochs', type=int, default=10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if args.tiled_train:
        train_ds = IguanaTiledDataset(args.train_csv, args.train_image_dir, crop_size=args.crop_size,
                                      overlap=args.tile_overlap, augment=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                  collate_fn=tiled_collate_fn)
    else:
        train_ds = IguanaPresenceDataset(args.train_csv, args.train_image_dir, crop_size=args.crop_size,
                                         crops_per_image=args.crops_per_image, positive_ratio=args.positive_ratio,
                                          augment=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    if args.val_csv and args.val_image_dir:
        if args.tiled_val:
            val_ds = IguanaTiledDataset(args.val_csv, args.val_image_dir, crop_size=args.crop_size,
                                        overlap=args.tile_overlap)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, collate_fn=tiled_collate_fn)
        else:
            val_ds = IguanaPresenceDataset(args.val_csv, args.val_image_dir, crop_size=args.crop_size,
                                           crops_per_image=args.crops_per_image, positive_ratio=0.5, augment=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

    # Model
    model = IguanaClassifier(backbone=args.backbone, freeze_backbone=True,
                             hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}, Trainable: {n_train:,}")

    # Load weights only (no optimizer state) - NEW
    if args.load_from:
        load_path = Path(args.load_from)
        if load_path.exists():
            print(f"\nLoading model weights from: {load_path}")
            ckpt = torch.load(load_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            loaded_f3 = ckpt.get('best_f3', ckpt.get('optimal_f3', 'N/A'))
            loaded_thresh = ckpt.get('threshold', ckpt.get('optimal_threshold_f3', 'N/A'))
            print(f"  Loaded weights (F3={loaded_f3}, threshold={loaded_thresh})")
        else:
            print(f"\nWarning: Load path not found: {load_path}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch, best_f3 = 0, 0
    epochs_without_improvement, backbone_unfrozen = 0, False

    # Resume training (with optimizer state)
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_f3 = ckpt.get('best_f3', 0)
            epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
            backbone_unfrozen = ckpt.get('backbone_unfrozen', False)

            if backbone_unfrozen:
                print(f"  Restoring unfrozen backbone state ({args.unfreeze_blocks} blocks)")
                model.unfreeze_backbone(args.unfreeze_blocks)
                optimizer = torch.optim.AdamW([
                    {'params': [p for n, p in model.named_parameters()
                               if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.01},
                    {'params': [p for n, p in model.named_parameters()
                               if 'backbone' not in n and p.requires_grad], 'lr': args.lr * 0.1},
                ], weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - start_epoch, eta_min=1e-7)

            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except ValueError as e:
                    print(f"  Warning: Could not restore optimizer state: {e}")
            if 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except:
                    print("  Warning: Could not restore scheduler state")

            print(f"  Resuming from epoch {start_epoch}, best F3: {best_f3:.4f}")

    print("\n" + "=" * 70)
    print(f"TRAINING (optimizing for F3, pos_weight={args.pos_weight}, threshold={args.threshold})")
    if args.early_stopping > 0:
        print(f"Early stopping: patience={args.early_stopping}, min_epochs={args.min_epochs}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.unfreeze_epoch and args.unfreeze_blocks > 0 and not backbone_unfrozen:
            print(f"\n*** Unfreezing last {args.unfreeze_blocks} backbone blocks ***")
            model.unfreeze_backbone(args.unfreeze_blocks)
            backbone_unfrozen = True
            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.named_parameters()
                           if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.01},
                {'params': [p for n, p in model.named_parameters()
                           if 'backbone' not in n and p.requires_grad], 'lr': args.lr * 0.1},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-7)

        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, device, epoch,
                              is_tiled=args.tiled_train, patch_loss_weight=args.patch_loss_weight,
                              pos_weight=args.pos_weight)
        scheduler.step()

        log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} "
        log += f"[pos={train_m['pos_acc']:.3f}({train_m['pos_total']}) neg={train_m['neg_acc']:.3f}({train_m['neg_total']})]"

        improved = False
        if val_loader:
            val_m = evaluate(model, val_loader, device, is_tiled=args.tiled_val, threshold=args.threshold)
            log += f" | P={val_m['precision']:.3f} R={val_m['recall']:.3f} F1={val_m['f1']:.3f} F3={val_m['f3']:.3f}"

            if val_m['f3'] > best_f3:
                best_f3 = val_m['f3']
                epochs_without_improvement = 0
                improved = True
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f3': best_f3, 'threshold': args.threshold, 'backbone_unfrozen': backbone_unfrozen,
                }, output_dir / 'best.pth')
                log += " ★"
            else:
                epochs_without_improvement += 1
                if args.early_stopping > 0:
                    log += f" (no improvement: {epochs_without_improvement}/{args.early_stopping})"

        print(log)

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f3': best_f3, 'backbone_unfrozen': backbone_unfrozen,
            'epochs_without_improvement': epochs_without_improvement,
        }, output_dir / 'latest.pth')

        if args.early_stopping > 0 and epoch >= args.min_epochs:
            if epochs_without_improvement >= args.early_stopping:
                print(f"\n*** Early stopping after {epochs_without_improvement} epochs without improvement ***")
                break

    print("\n" + "=" * 70)
    print(f"Training complete! Best F3: {best_f3:.4f}")
    print("=" * 70)

    # Final evaluation with optimal threshold
    if val_loader:
        best_path = output_dir / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

        val_m = evaluate(model, val_loader, device, is_tiled=args.tiled_val, threshold=args.threshold)
        opt_thresh, opt_f3 = find_optimal_threshold(val_m['all_probs'], val_m['all_labels'])
        print(f"\nOptimal threshold: {opt_thresh:.3f}, F3: {opt_f3:.4f}")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimal_threshold_f3': opt_thresh, 'optimal_f3': opt_f3,
            'backbone': args.backbone, 'hidden_dim': args.hidden_dim,
        }, output_dir / 'best_with_threshold.pth')


if __name__ == '__main__':
    main()