"""
Iguana Presence Classifier with Count Regression

Extended version that includes:
1. Binary classification: Does this tile contain an iguana?
2. Count regression: How many iguanas are in this tile?

The count head can be trained jointly with classification, or used standalone.
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
from loguru import logger
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

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for training
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# DATASET: Deterministic tiled validation dataset
# =============================================================================

class IguanaTiledDataset(Dataset):
    """
    Deterministic dataset that tiles full images into non-overlapping crops.

    For validation/inference:
    - Pads images to multiple of crop_size
    - Extracts all tiles in a grid pattern
    - Each tile is labeled based on whether it contains any iguana points
    - Also provides count of iguanas in each tile

    This gives reproducible evaluation on the full image coverage.

    For training, can enable augmentation.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        crop_size: int = 518,
        overlap: int = 0,  # Overlap between tiles (0 = non-overlapping)
        augment: bool = False,  # Enable augmentation for training
        patch_size: int = 14,  # ViT patch size for creating patch labels
        point_radius: int = 1,  # Radius in patches for labeling (1 = just the patch containing the point)
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.overlap = overlap
        self.stride = crop_size - overlap
        self.augment = augment
        self.patch_size = patch_size
        self.point_radius = point_radius
        self.grid_size = crop_size // patch_size

        # Load annotations
        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        # Store points per image
        self.annotations = {}
        for name, group in self.df.groupby('images'):
            self.annotations[name] = group[['x', 'y']].values.astype(np.float32)

        # Normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Augmentation (photometric only - geometric would require point transformation)
        self.photometric_transform = self._build_photometric() if augment else None

        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])

        # Pre-compute all tiles
        self.tiles = []  # List of (image_name, crop_x, crop_y)
        self._image_sizes = {}

        for name in self.image_names:
            img_path = os.path.join(self.image_dir, name)
            with Image.open(img_path) as img:
                w, h = img.size
                self._image_sizes[name] = (w, h)

            # Compute padded size
            pad_w = (self.stride - (w % self.stride)) % self.stride if w % self.stride != 0 else 0
            pad_h = (self.stride - (h % self.stride)) % self.stride if h % self.stride != 0 else 0
            padded_w = w + pad_w
            padded_h = h + pad_h

            # Generate tile positions
            n_tiles_x = max(1, (padded_w - self.overlap) // self.stride)
            n_tiles_y = max(1, (padded_h - self.overlap) // self.stride)

            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    crop_x = tx * self.stride
                    crop_y = ty * self.stride
                    self.tiles.append((name, crop_x, crop_y))

        # Statistics
        total_points = sum(len(pts) for pts in self.annotations.values())
        print(f"IguanaTiledDataset: {len(self.image_names)} images, {total_points} points")
        print(f"  Crop size: {crop_size}, Stride: {self.stride}, Overlap: {overlap}")
        print(f"  Total tiles: {len(self.tiles)}")

        # Count positive/negative tiles and count distribution
        counts = []
        n_pos = 0
        for name, cx, cy in self.tiles:
            count = self._count_points_in_tile(name, cx, cy)
            counts.append(count)
            if count > 0:
                n_pos += 1

        counts = np.array(counts)
        print(f"  Positive tiles: {n_pos}, Negative tiles: {len(self.tiles) - n_pos}")
        print(f"  Count distribution: min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.2f}, median={np.median(counts):.1f}")
        print(f"  Augmentation: {augment}")

    def _build_photometric(self):
        """Build photometric augmentation pipeline (no geometric - preserves point coords)."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            # Dropout augmentations - simulate occlusion
            A.CoarseDropout(
                max_holes=8,
                max_height=64,
                max_width=64,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.3,
            ),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.2,
            ),
        ])

    def _has_points_in_tile(self, name: str, crop_x: int, crop_y: int) -> bool:
        """Check if any points fall within this tile."""
        points = self.annotations[name]
        if len(points) == 0:
            return False

        in_crop = (
            (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
            (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
        )
        return in_crop.any()

    def _count_points_in_tile(self, name: str, crop_x: int, crop_y: int) -> int:
        """Count how many points fall within this tile."""
        points = self.annotations[name]
        if len(points) == 0:
            return 0

        in_crop = (
            (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
            (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
        )
        return in_crop.sum()

    def _get_points_in_tile(self, name: str, crop_x: int, crop_y: int) -> np.ndarray:
        """Get all points within this tile (in tile coordinates)."""
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
        """
        Create binary patch labels from points.

        Args:
            points_in_crop: [N, 2] array of (x, y) points in tile coordinates
        Returns:
            patch_labels: [grid_size, grid_size] binary tensor
        """
        patch_labels = torch.zeros(self.grid_size, self.grid_size)

        if len(points_in_crop) == 0:
            return patch_labels

        for pt in points_in_crop:
            px, py = pt

            # Convert pixel coords to patch coords
            patch_x = int(px / self.patch_size)
            patch_y = int(py / self.patch_size)

            # Clamp to valid range
            patch_x = max(0, min(patch_x, self.grid_size - 1))
            patch_y = max(0, min(patch_y, self.grid_size - 1))

            # Mark patches within radius
            for dy in range(-self.point_radius + 1, self.point_radius):
                for dx in range(-self.point_radius + 1, self.point_radius):
                    py_idx = patch_y + dy
                    px_idx = patch_x + dx

                    if 0 <= py_idx < self.grid_size and 0 <= px_idx < self.grid_size:
                        patch_labels[py_idx, px_idx] = 1.0

        return patch_labels

    def _create_density_map(self, points_in_crop: np.ndarray) -> torch.Tensor:
        """
        Create density map for count regression at patch level.

        Each patch gets the count of points that fall within it.
        This can be used for density-based counting.

        Args:
            points_in_crop: [N, 2] array of (x, y) points in tile coordinates
        Returns:
            density_map: [grid_size, grid_size] count tensor
        """
        density_map = torch.zeros(self.grid_size, self.grid_size)

        if len(points_in_crop) == 0:
            return density_map

        for pt in points_in_crop:
            px, py = pt

            # Convert pixel coords to patch coords
            patch_x = int(px / self.patch_size)
            patch_y = int(py / self.patch_size)

            # Clamp to valid range
            patch_x = max(0, min(patch_x, self.grid_size - 1))
            patch_y = max(0, min(patch_y, self.grid_size - 1))

            # Increment count at this patch
            density_map[patch_y, patch_x] += 1.0

        return density_map

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        name, crop_x, crop_y = self.tiles[idx]

        # Load image
        img_path = os.path.join(self.image_dir, name)
        img = np.array(Image.open(img_path).convert('RGB'))
        img_h, img_w = img.shape[:2]

        # Pad image if needed
        pad_right = max(0, crop_x + self.crop_size - img_w)
        pad_bottom = max(0, crop_y + self.crop_size - img_h)

        if pad_right > 0 or pad_bottom > 0:
            img = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)

        # Extract crop
        crop = img[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]

        # Get label and count
        points_in_crop = self._get_points_in_tile(name, crop_x, crop_y)
        label = 1.0 if len(points_in_crop) > 0 else 0.0
        count = float(len(points_in_crop))

        # Create patch-level labels for supervision
        patch_labels = self._create_patch_labels(points_in_crop)
        density_map = self._create_density_map(points_in_crop)

        # Apply augmentation (if enabled)
        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']

        # Normalize
        crop = self.normalize_transform(image=crop)['image']

        return crop, {
            'label': torch.tensor(label, dtype=torch.float32),
            'count': torch.tensor(count, dtype=torch.float32),
            'patch_labels': patch_labels,
            'density_map': density_map,
            'points': torch.from_numpy(points_in_crop).float(),
            'name': name,
            'crop_x': crop_x,
            'crop_y': crop_y,
        }


def tiled_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {
        'label': torch.stack([b[1]['label'] for b in batch]),
        'count': torch.stack([b[1]['count'] for b in batch]),
        'patch_labels': torch.stack([b[1]['patch_labels'] for b in batch]),
        'density_map': torch.stack([b[1]['density_map'] for b in batch]),
        'points': [b[1]['points'] for b in batch],
        'name': [b[1]['name'] for b in batch],
        'crop_x': [b[1]['crop_x'] for b in batch],
        'crop_y': [b[1]['crop_y'] for b in batch],
    }
    return images, targets


# =============================================================================
# DATASET: Random crops with presence labels (for training)
# =============================================================================

class IguanaPresenceDataset(Dataset):
    """
    Dataset that extracts random crops and labels them by iguana presence.

    Positive: Crop contains at least one iguana point (with margin from edge)
    Negative: Crop contains no iguana points

    Also provides count of iguanas in each crop.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        crop_size: int = 512,
        crops_per_image: int = 4,
        positive_ratio: float = 0.5,
        min_edge_margin: int = 20,
        augment: bool = True,
        patch_size: int = 14,
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.positive_ratio = positive_ratio
        self.min_edge_margin = min_edge_margin
        self.augment = augment
        self.patch_size = patch_size
        self.grid_size = crop_size // patch_size

        # Load annotations
        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        # Store points per image in original pixel coordinates
        self.annotations = {}
        for name, group in self.df.groupby('images'):
            self.annotations[name] = group[['x', 'y']].values.astype(np.float32)

        # Normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Build transforms
        self.photometric_transform = self._build_photometric() if augment else None
        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])

        # Cache image sizes for efficient sampling
        self._image_sizes = {}

        print(f"IguanaPresenceDataset: {len(self.image_names)} images")
        print(f"  Crop size: {crop_size}, Crops per image: {crops_per_image}")
        print(f"  Positive ratio: {positive_ratio}, Edge margin: {min_edge_margin}")
        print(f"  Total crops per epoch: {len(self)}")

        # Diagnostic: Check image sizes and point density
        total_points = sum(len(pts) for pts in self.annotations.values())
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
            A.CoarseDropout(
                max_holes=8,
                max_height=64,
                max_width=64,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.3,
            ),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.2,
            ),
        ])

    def _get_image_size(self, name: str) -> Tuple[int, int]:
        """Get image dimensions (cached)."""
        if name not in self._image_sizes:
            img_path = os.path.join(self.image_dir, name)
            with Image.open(img_path) as img:
                self._image_sizes[name] = img.size  # (width, height)
        return self._image_sizes[name]

    def _sample_positive_crop(self, points: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int]:
        """Sample a crop that contains at least one point with margin."""
        valid_points = []
        for pt in points:
            px, py = pt
            if (px >= self.min_edge_margin and
                px <= img_w - self.min_edge_margin and
                py >= self.min_edge_margin and
                py <= img_h - self.min_edge_margin):
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

    def _get_points_in_crop(self, points: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray:
        """Get points within crop in crop-local coordinates."""
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

    def _create_density_map(self, points_in_crop: np.ndarray) -> torch.Tensor:
        """Create density map for count regression."""
        density_map = torch.zeros(self.grid_size, self.grid_size)

        if len(points_in_crop) == 0:
            return density_map

        for pt in points_in_crop:
            px, py = pt
            patch_x = int(px / self.patch_size)
            patch_y = int(py / self.patch_size)
            patch_x = max(0, min(patch_x, self.grid_size - 1))
            patch_y = max(0, min(patch_y, self.grid_size - 1))
            density_map[patch_y, patch_x] += 1.0

        return density_map

    def __len__(self):
        return len(self.image_names) * self.crops_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image

        name = self.image_names[img_idx]
        points = self.annotations[name]

        # Load image
        img_path = os.path.join(self.image_dir, name)
        img = np.array(Image.open(img_path).convert('RGB'))
        img_h, img_w = img.shape[:2]

        # Decide positive or negative based on ratio
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

        # Get points in crop and compute label/count
        points_in_crop = self._get_points_in_crop(points, crop_x, crop_y)
        label = 1.0 if len(points_in_crop) > 0 else 0.0
        count = float(len(points_in_crop))

        # Create density map
        density_map = self._create_density_map(points_in_crop)

        # Apply augmentations
        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']

        # Normalize and convert to tensor
        crop = self.normalize_transform(image=crop)['image']

        return crop, {
            'label': torch.tensor(label, dtype=torch.float32),
            'count': torch.tensor(count, dtype=torch.float32),
            'density_map': density_map,
        }


def presence_collate_fn(batch):
    """Collate function for IguanaPresenceDataset with count."""
    images = torch.stack([b[0] for b in batch])
    targets = {
        'label': torch.stack([b[1]['label'] for b in batch]),
        'count': torch.stack([b[1]['count'] for b in batch]),
        'density_map': torch.stack([b[1]['density_map'] for b in batch]),
    }
    return images, targets


# =============================================================================
# MODEL: DINOv3 backbone + classification head + count regression head
# =============================================================================

class IguanaClassifierWithCount(nn.Module):
    """
    Binary classifier + count regressor using DINOv2 backbone.

    Outputs:
    1. Binary classification logit (from CLS token)
    2. Count prediction (from CLS token or sum of density predictions)
    3. Optional: patch-level predictions for visualization

    Count prediction strategies:
    - 'cls': Direct regression from CLS token
    - 'density': Sum of patch-level density predictions
    - 'both': Both approaches (can ensemble at inference)

    Register token usage:
    - DINOv2 reg variants have register tokens that absorb global information
    - These can be leveraged for better global representations (counting)
    - use_registers=True concatenates register features with CLS for global heads
    """

    def __init__(
        self,
        backbone: str = 'vit_large_patch14_reg4_dinov2.lvd142m',
        freeze_backbone: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        count_strategy: str = 'both',  # 'cls', 'density', or 'both'
        max_count: int = 50,  # Maximum expected count per tile (for scaling)
        use_registers: bool = True,  # Whether to use register tokens for global features
        pretrained: bool = True,  # Whether to load pretrained backbone weights
    ):
        super().__init__()

        self.count_strategy = count_strategy
        self.max_count = max_count
        self.use_registers = use_registers

        # Load backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)

        # Determine number of register tokens (prefix tokens - 1 for CLS)
        self.num_register_tokens = max(0, self.num_prefix_tokens - 1)

        # Global feature dimension: CLS + (optionally) register tokens
        if use_registers and self.num_register_tokens > 0:
            # Option 1: Concatenate CLS + mean of registers
            # self.global_feat_dim = self.feat_dim * 2

            # Option 2: Concatenate CLS + all registers (more expressive)
            # self.global_feat_dim = self.feat_dim * (1 + self.num_register_tokens)

            # Option 3: Use attention-weighted combination (best for counting)
            # We'll use a small attention mechanism to weight CLS + registers
            self.global_feat_dim = self.feat_dim
            self.register_attention = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 4, 1),
            )
        else:
            self.global_feat_dim = self.feat_dim
            self.register_attention = None

        print(f"Backbone: {backbone}")
        print(f"  Feature dim: {self.feat_dim}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Prefix tokens: {self.num_prefix_tokens} (1 CLS + {self.num_register_tokens} registers)")
        print(f"  Use registers: {use_registers}")
        print(f"  Global feat dim: {self.global_feat_dim}")
        print(f"  Count strategy: {count_strategy}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen")

        # Classification head (from global features) - binary presence
        self.cls_head = nn.Sequential(
            nn.Linear(self.global_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Count regression head from global features
        if count_strategy in ['cls', 'both']:
            self.count_head_cls = nn.Sequential(
                nn.Linear(self.global_feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.count_head_cls = None

        # Patch-level density head (for localization and density-based counting)
        if count_strategy in ['density', 'both']:
            self.density_head = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.density_head = None

        # Patch-level binary head (for visualization)
        self.patch_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize heads
        for head in [self.cls_head, self.count_head_cls, self.density_head, self.patch_head, self.register_attention]:
            if head is not None:
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def _get_global_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract global features from CLS and register tokens.

        Args:
            features: [B, num_tokens, feat_dim] output from backbone

        Returns:
            global_feat: [B, global_feat_dim] aggregated global features
        """
        cls_token = features[:, 0]  # [B, feat_dim]

        if self.use_registers and self.num_register_tokens > 0 and self.register_attention is not None:
            # Get register tokens
            register_tokens = features[:, 1:1+self.num_register_tokens]  # [B, num_reg, feat_dim]

            # Combine CLS with registers: [B, 1+num_reg, feat_dim]
            global_tokens = torch.cat([cls_token.unsqueeze(1), register_tokens], dim=1)

            # Compute attention weights
            attn_logits = self.register_attention(global_tokens).squeeze(-1)  # [B, 1+num_reg]
            attn_weights = F.softmax(attn_logits, dim=-1)  # [B, 1+num_reg]

            # Weighted combination
            global_feat = (attn_weights.unsqueeze(-1) * global_tokens).sum(dim=1)  # [B, feat_dim]

            return global_feat
        else:
            return cls_token

    def forward(
        self,
        x: torch.Tensor,
        return_patches: bool = False,
        return_density: bool = False,
        upsample_patches: bool = False,
        return_attention_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, 3, H, W] input images
            return_patches: if True, return patch-level binary predictions
            return_density: if True, return patch-level density predictions
            upsample_patches: if True, upsample patch predictions to image size
            return_attention_weights: if True, return register attention weights

        Returns:
            Dictionary with:
                'cls_logit': [B] binary classification logits
                'count_cls': [B] count prediction from CLS (if enabled)
                'count_density': [B] count prediction from density sum (if enabled)
                'count': [B] final count prediction (best available)
                'patch_logits': [B, grid_h, grid_w] binary patch predictions (if requested)
                'density_map': [B, grid_h, grid_w] density predictions (if requested)
                'register_attn': [B, 1+num_reg] attention weights (if requested)
        """
        B, C, H, W = x.shape
        grid_size = H // self.patch_size

        # Get all tokens from backbone
        features = self.backbone.forward_features(x)  # [B, num_tokens, feat_dim]

        # Get global features (CLS + registers with attention)
        global_feat = self._get_global_features(features)  # [B, global_feat_dim]

        # Patch tokens (skip all prefix tokens: CLS + registers)
        patch_tokens = features[:, self.num_prefix_tokens:]  # [B, N, feat_dim]
        patch_tokens = patch_tokens.view(B, grid_size, grid_size, self.feat_dim)

        output = {}

        # Binary classification from global features
        output['cls_logit'] = self.cls_head(global_feat).squeeze(-1)  # [B]

        # Count from global features
        if self.count_head_cls is not None:
            # Use softplus to ensure non-negative count
            count_cls = F.softplus(self.count_head_cls(global_feat).squeeze(-1))  # [B]
            output['count_cls'] = count_cls

        # Count from density (sum of patch predictions)
        if self.density_head is not None:
            # Predict density per patch
            density_logits = self.density_head(patch_tokens).squeeze(-1)  # [B, grid_h, grid_w]
            # Use softplus for non-negative density
            density_pred = F.softplus(density_logits)  # [B, grid_h, grid_w]

            # Sum for total count
            count_density = density_pred.sum(dim=(1, 2))  # [B]
            output['count_density'] = count_density

            if return_density:
                if upsample_patches:
                    density_pred = F.interpolate(
                        density_pred.unsqueeze(1),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                output['density_map'] = density_pred

        # Determine final count prediction
        if self.count_strategy == 'cls':
            output['count'] = output['count_cls']
        elif self.count_strategy == 'density':
            output['count'] = output['count_density']
        else:  # 'both' - use average
            output['count'] = (output['count_cls'] + output['count_density']) / 2

        # Patch-level binary predictions (for visualization)
        if return_patches:
            patch_logits = self.patch_head(patch_tokens).squeeze(-1)  # [B, grid_h, grid_w]

            if upsample_patches:
                patch_logits = F.interpolate(
                    patch_logits.unsqueeze(1),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            output['patch_logits'] = patch_logits

        # Return register attention weights if requested (for analysis)
        if return_attention_weights and self.use_registers and self.register_attention is not None:
            cls_token = features[:, 0:1]
            register_tokens = features[:, 1:1+self.num_register_tokens]
            global_tokens = torch.cat([cls_token, register_tokens], dim=1)
            attn_logits = self.register_attention(global_tokens).squeeze(-1)
            output['register_attn'] = F.softmax(attn_logits, dim=-1)

        return output

    def unfreeze_backbone(self, n_blocks: Optional[int] = None):
        """Unfreeze backbone (all or last n blocks)."""
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

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int = 0,
    is_tiled: bool = False,
    cls_loss_weight: float = 1.0,
    count_loss_weight: float = 1.0,
    density_loss_weight: float = 0.5,
    pos_weight: float = 1.0,
):
    """
    Train for one epoch with combined classification and count loss.
    """
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_count_loss = 0
    total_density_loss = 0
    total_correct = 0
    total_samples = 0

    total_count_mae = 0
    pos_total = 0
    neg_total = 0

    cls_pos_weight = torch.tensor([pos_weight], device=device)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        labels = targets['label'].to(device)
        counts = targets['count'].to(device)

        if 'density_map' in targets:
            density_gt = targets['density_map'].to(device)
        else:
            density_gt = None

        optimizer.zero_grad()

        # Forward pass
        output = model(images, return_density=(density_gt is not None))

        # Classification loss
        cls_loss = F.binary_cross_entropy_with_logits(
            output['cls_logit'], labels,
            pos_weight=cls_pos_weight.expand_as(labels)
        )

        # Count regression loss (Smooth L1 / Huber loss)
        count_loss = F.smooth_l1_loss(output['count'], counts)

        # Optional density loss (per-patch supervision)
        density_loss = torch.tensor(0.0, device=device)
        if density_gt is not None and 'density_map' in output:
            density_loss = F.smooth_l1_loss(output['density_map'], density_gt)

        # Combined loss
        loss = (
            cls_loss_weight * cls_loss +
            count_loss_weight * count_loss +
            density_loss_weight * density_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(labels)
        total_cls_loss += cls_loss.item() * len(labels)
        total_count_loss += count_loss.item() * len(labels)
        total_density_loss += density_loss.item() * len(labels)

        # Classification accuracy
        probs = torch.sigmoid(output['cls_logit'])
        preds = (probs > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        # Count MAE
        total_count_mae += torch.abs(output['count'] - counts).sum().item()

        pos_total += (labels == 1).sum().item()
        neg_total += (labels == 0).sum().item()

    n = total_samples
    return {
        'loss': total_loss / n,
        'cls_loss': total_cls_loss / n,
        'count_loss': total_count_loss / n,
        'density_loss': total_density_loss / n,
        'acc': total_correct / n,
        'count_mae': total_count_mae / n,
        'pos_total': pos_total,
        'neg_total': neg_total,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    is_tiled: bool = False,
    threshold: float = 0.5,
):
    """Evaluate model on classification and count metrics."""
    model.eval()

    total_loss = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []
    all_counts_pred = []
    all_counts_gt = []

    for images, targets in loader:
        images = images.to(device)
        labels = targets['label'].to(device)
        counts = targets['count'].to(device)

        output = model(images)

        # Classification loss
        cls_loss = F.binary_cross_entropy_with_logits(output['cls_logit'], labels)
        # Count loss
        count_loss = F.smooth_l1_loss(output['count'], counts)

        loss = cls_loss + count_loss

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

        probs = torch.sigmoid(output['cls_logit'])
        preds = (probs > threshold).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_counts_pred.extend(output['count'].cpu().numpy())
        all_counts_gt.extend(counts.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_counts_pred = np.array(all_counts_pred)
    all_counts_gt = np.array(all_counts_gt)

    # Classification metrics
    accuracy = (all_preds == all_labels).mean()

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    pos_acc = (all_preds[pos_mask] == all_labels[pos_mask]).mean() if pos_mask.any() else 0
    neg_acc = (all_preds[neg_mask] == all_labels[neg_mask]).mean() if neg_mask.any() else 0

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    f3 = 10 * precision * recall / max(9 * precision + recall, 1e-6)

    # Count metrics
    count_mae = np.abs(all_counts_pred - all_counts_gt).mean()
    count_rmse = np.sqrt(((all_counts_pred - all_counts_gt) ** 2).mean())

    # Count metrics on positive tiles only (where count > 0)
    pos_count_mask = all_counts_gt > 0
    if pos_count_mask.any():
        count_mae_pos = np.abs(all_counts_pred[pos_count_mask] - all_counts_gt[pos_count_mask]).mean()
        count_mape = np.abs(
            (all_counts_pred[pos_count_mask] - all_counts_gt[pos_count_mask]) /
            np.maximum(all_counts_gt[pos_count_mask], 1)
        ).mean() * 100
    else:
        count_mae_pos = 0
        count_mape = 0

    return {
        'loss': total_loss / total_samples,
        'acc': accuracy,
        'pos_acc': pos_acc,
        'neg_acc': neg_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f3': f3,
        'n_pos': pos_mask.sum(),
        'n_neg': neg_mask.sum(),
        'count_mae': count_mae,
        'count_rmse': count_rmse,
        'count_mae_pos': count_mae_pos,
        'count_mape': count_mape,
        'all_probs': all_probs,
        'all_labels': all_labels,
        'all_counts_pred': all_counts_pred,
        'all_counts_gt': all_counts_gt,
    }


def visualize_validation_errors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    threshold: float = 0.3,
    max_samples: int = 16,
    is_tiled: bool = True,
):
    """
    Visualize false positives and false negatives during validation.

    Saves a summary grid for each error type to output_dir/epoch_XXX/

    Args:
        model: Trained model
        loader: Validation DataLoader
        device: torch device
        output_dir: Base output directory
        epoch: Current epoch number
        threshold: Classification threshold
        max_samples: Maximum samples to visualize per category
        is_tiled: Whether using tiled dataset
    """
    if not HAS_MATPLOTLIB:
        return

    model.eval()

    # Collect all predictions
    all_results = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            if is_tiled:
                labels = targets['label']
                counts = targets['count']
                names = targets['name']
                crop_xs = targets['crop_x']
                crop_ys = targets['crop_y']
                points_list = targets['points']
            else:
                labels = targets['label']
                counts = targets['count']
                names = [None] * len(labels)
                crop_xs = [None] * len(labels)
                crop_ys = [None] * len(labels)
                points_list = [None] * len(labels)

            output = model(images, return_patches=True, return_density=True)
            probs = torch.sigmoid(output['cls_logit']).cpu().numpy()
            count_preds = output['count'].cpu().numpy()

            # Get patch heatmaps
            if 'density_map' in output:
                density_maps = output['density_map'].cpu().numpy()
            else:
                density_maps = [None] * len(labels)

            for i in range(len(labels)):
                # Unnormalize image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

                all_results.append({
                    'image': img,
                    'prob': float(probs[i]),
                    'label': int(labels[i]),
                    'count_pred': float(count_preds[i]),
                    'count_gt': float(counts[i]),
                    'name': names[i] if is_tiled else None,
                    'crop_x': crop_xs[i] if is_tiled else None,
                    'crop_y': crop_ys[i] if is_tiled else None,
                    'points': points_list[i].numpy() if is_tiled and points_list[i] is not None else None,
                    'density_map': density_maps[i] if density_maps[i] is not None else None,
                })

    # Classify results
    false_negatives = [r for r in all_results if r['label'] == 1 and r['prob'] <= threshold]
    false_positives = [r for r in all_results if r['label'] == 0 and r['prob'] > threshold]
    true_positives = [r for r in all_results if r['label'] == 1 and r['prob'] > threshold]
    true_negatives = [r for r in all_results if r['label'] == 0 and r['prob'] <= threshold]

    # Sort by confidence
    false_negatives.sort(key=lambda x: x['prob'])  # Most confident misses first
    false_positives.sort(key=lambda x: -x['prob'])  # Most confident false alarms first
    true_positives.sort(key=lambda x: -x['prob'])
    true_negatives.sort(key=lambda x: x['prob'])

    # Create epoch directory
    epoch_dir = output_dir / f'epoch_{epoch:03d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Plot summary grid for each category
    categories = [
        (false_negatives, 'false_negatives', 'FALSE NEGATIVES (Missed)', 'lime'),
        (false_positives, 'false_positives', 'FALSE POSITIVES (False Alarm)', 'red'),
    ]

    for samples, name, title, point_color in categories:
        if len(samples) == 0:
            continue

        n_show = min(max_samples, len(samples))
        n_cols = 4
        n_rows = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, r in enumerate(samples[:n_show]):
            ax = axes[i]
            ax.imshow(r['image'])

            # Draw GT points if available
            if r['points'] is not None and len(r['points']) > 0:
                for pt in r['points']:
                    circle = plt.Circle((pt[0], pt[1]), 12, color=point_color, fill=False, linewidth=2)
                    ax.add_patch(circle)

            # Title with probabilities and counts
            title_str = f"p={r['prob']:.3f}"
            if r['count_gt'] > 0 or r['count_pred'] > 0.5:
                title_str += f"\nGT:{int(r['count_gt'])} Pred:{r['count_pred']:.1f}"
            ax.set_title(title_str, fontsize=9)
            ax.axis('off')

        # Hide unused axes
        for i in range(n_show, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f"{title} (n={len(samples)}) - Epoch {epoch} @ threshold={threshold:.2f}", fontsize=12)
        plt.tight_layout()
        plt.savefig(epoch_dir / f'{name}.png', dpi=120, bbox_inches='tight')
        plt.close()

    # Also create a combined overview
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    categories_overview = [
        (false_negatives, 'FN', 0, 'lime'),
        (false_positives, 'FP', 1, 'red'),
        (true_positives, 'TP', 2, 'lime'),
        (true_negatives, 'TN', 3, 'blue'),
    ]

    for samples, label, col, point_color in categories_overview:
        for row in range(2):
            ax = axes[row, col]
            idx = row
            if idx < len(samples):
                r = samples[idx]
                ax.imshow(r['image'])
                if r['points'] is not None and len(r['points']) > 0:
                    for pt in r['points']:
                        circle = plt.Circle((pt[0], pt[1]), 10, color=point_color, fill=False, linewidth=2)
                        ax.add_patch(circle)
                ax.set_title(f"p={r['prob']:.2f} c={r['count_pred']:.1f}", fontsize=8)
            ax.axis('off')
            if row == 0:
                ax.set_xlabel(f"{label} (n={len(samples)})", fontsize=10)

    fig.suptitle(f"Epoch {epoch} Overview @ threshold={threshold:.2f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(epoch_dir / 'overview.png', dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'fn': len(false_negatives),
        'fp': len(false_positives),
        'tp': len(true_positives),
        'tn': len(true_negatives),
    }


def visualize_count_errors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    is_tiled: bool = True,
    max_samples: int = 16,
):
    """
    Visualize tiles with largest count prediction errors.

    Shows tiles where |count_pred - count_gt| is largest.
    """
    if not HAS_MATPLOTLIB:
        return

    model.eval()

    all_results = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            if is_tiled:
                labels = targets['label']
                counts = targets['count']
                points_list = targets['points']
            else:
                labels = targets['label']
                counts = targets['count']
                points_list = [None] * len(labels)

            output = model(images, return_density=True)
            count_preds = output['count'].cpu().numpy()

            density_maps = output.get('density_map', None)
            if density_maps is not None:
                density_maps = density_maps.cpu().numpy()

            for i in range(len(labels)):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

                count_error = abs(count_preds[i] - float(counts[i]))

                all_results.append({
                    'image': img,
                    'count_pred': float(count_preds[i]),
                    'count_gt': float(counts[i]),
                    'count_error': count_error,
                    'points': points_list[i].numpy() if is_tiled and points_list[i] is not None else None,
                    'density_map': density_maps[i] if density_maps is not None else None,
                })

    # Sort by count error (largest first)
    all_results.sort(key=lambda x: -x['count_error'])

    # Only show tiles with actual counts
    results_with_counts = [r for r in all_results if r['count_gt'] > 0]

    if len(results_with_counts) == 0:
        return

    epoch_dir = output_dir / f'epoch_{epoch:03d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    n_show = min(max_samples, len(results_with_counts))
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, r in enumerate(results_with_counts[:n_show]):
        ax = axes[i]
        ax.imshow(r['image'])

        if r['points'] is not None and len(r['points']) > 0:
            for pt in r['points']:
                circle = plt.Circle((pt[0], pt[1]), 12, color='lime', fill=False, linewidth=2)
                ax.add_patch(circle)

        error_sign = '+' if r['count_pred'] > r['count_gt'] else ''
        error = r['count_pred'] - r['count_gt']
        ax.set_title(f"GT:{int(r['count_gt'])} Pred:{r['count_pred']:.1f}\nError:{error_sign}{error:.1f}", fontsize=9)
        ax.axis('off')

    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Largest Count Errors - Epoch {epoch}", fontsize=12)
    plt.tight_layout()
    plt.savefig(epoch_dir / 'count_errors.png', dpi=120, bbox_inches='tight')
    plt.close()


def find_optimal_threshold(all_probs, all_labels, beta: float = 3.0):
    """Find threshold that maximizes F-beta score."""
    best_threshold = 0.5
    best_fbeta = 0
    best_metrics = {}

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
            best_fbeta = fbeta
            best_threshold = thresh
            best_metrics = {'precision': precision, 'recall': recall, 'f_beta': fbeta}

    return best_threshold, best_fbeta, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Iguana Classifier with Count Regression")

    # Data
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')

    # Model
    parser.add_argument('--backbone', default='vit_large_patch14_reg4_dinov2.lvd142m',
                        help='timm model name (DINOv2/v3)')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--count_strategy', default='both',
                        choices=['cls', 'density', 'both'],
                        help='Count prediction strategy')
    parser.add_argument('--use_registers', action='store_true', default=True,
                        help='Use register tokens for global features (default: True)')
    parser.add_argument('--no_registers', dest='use_registers', action='store_false',
                        help='Disable register token usage')

    # Dataset
    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--crops_per_image', type=int, default=8)
    parser.add_argument('--positive_ratio', type=float, default=0.5)
    parser.add_argument('--min_edge_margin', type=int, default=30)

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--unfreeze_epoch', type=int, default=10)
    parser.add_argument('--unfreeze_blocks', type=int, default=4)
    parser.add_argument('--pos_weight', type=float, default=3.0)
    parser.add_argument('--threshold', type=float, default=0.3)

    # Loss weights
    parser.add_argument('--cls_loss_weight', type=float, default=1.0,
                        help='Weight for classification loss')
    parser.add_argument('--count_loss_weight', type=float, default=1.0,
                        help='Weight for count regression loss')
    parser.add_argument('--density_loss_weight', type=float, default=0.5,
                        help='Weight for density map loss (patch-level supervision)')

    # Output
    parser.add_argument('--output_dir', default='./outputs_classifier_count')
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tiled_train', action='store_true')
    parser.add_argument('--tiled_val', action='store_true')
    parser.add_argument('--tile_overlap', type=int, default=0)

    # Resume and early stopping
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint (restores epoch, optimizer, best scores)')
    parser.add_argument('--load_from', type=str, default=None,
                        help='Load model weights from checkpoint (starts fresh training)')
    parser.add_argument('--early_stopping', type=int, default=15)
    parser.add_argument('--min_epochs', type=int, default=10)

    # Visualization
    parser.add_argument('--visualize_every', type=int, default=5,
                        help='Visualize validation errors every N epochs (0 = disabled)')
    parser.add_argument('--max_vis_samples', type=int, default=16,
                        help='Max samples per category to visualize')

    args = parser.parse_args()

    # Validate arguments
    if args.resume and args.load_from:
        raise ValueError("Cannot use both --resume and --load_from. "
                        "Use --resume to continue training, --load_from for fresh training with pretrained weights.")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_tiled_train = args.tiled_train
    use_tiled_val = args.tiled_val

    # Training data
    if use_tiled_train:
        train_ds = IguanaTiledDataset(
            args.train_csv, args.train_image_dir,
            crop_size=args.crop_size,
            overlap=args.tile_overlap,
            augment=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            collate_fn=tiled_collate_fn,
        )
    else:
        train_ds = IguanaPresenceDataset(
            args.train_csv, args.train_image_dir,
            crop_size=args.crop_size,
            crops_per_image=args.crops_per_image,
            positive_ratio=args.positive_ratio,
            min_edge_margin=args.min_edge_margin,
            augment=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            collate_fn=presence_collate_fn,
        )

    # Validation data
    val_loader = None
    if args.val_csv and args.val_image_dir:
        if use_tiled_val:
            val_ds = IguanaTiledDataset(
                args.val_csv, args.val_image_dir,
                crop_size=args.crop_size,
                overlap=args.tile_overlap,
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                collate_fn=tiled_collate_fn,
            )
        else:
            val_ds = IguanaPresenceDataset(
                args.val_csv, args.val_image_dir,
                crop_size=args.crop_size,
                crops_per_image=args.crops_per_image,
                positive_ratio=0.5,
                min_edge_margin=args.min_edge_margin,
                augment=False,
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                collate_fn=presence_collate_fn,
            )

    # Model
    model = IguanaClassifierWithCount(
        backbone=args.backbone,
        freeze_backbone=True,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        count_strategy=args.count_strategy,
        use_registers=args.use_registers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}, Trainable: {n_train:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Load weights only (fresh training)
    start_epoch = 0
    best_f3 = 0
    best_count_mae = float('inf')
    epochs_without_improvement = 0
    backbone_unfrozen = False

    if args.load_from:
        load_path = Path(args.load_from)
        if load_path.exists():
            print(f"\nLoading weights from {load_path} (fresh training)")
            ckpt = torch.load(load_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

            # Check if backbone was unfrozen in the checkpoint
            if ckpt.get('backbone_unfrozen', False):
                print("  Note: Loaded checkpoint had unfrozen backbone, but starting with frozen backbone")

            print(f"  Loaded weights from epoch {ckpt.get('epoch', '?')}, "
                  f"checkpoint F3: {ckpt.get('best_f3', '?')}, MAE: {ckpt.get('best_count_mae', '?')}")
            print(f"  Starting fresh training from epoch 0")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    # Resume training (restore everything)
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            start_epoch = ckpt.get('epoch', 0) + 1
            best_f3 = ckpt.get('best_f3', 0)
            best_count_mae = ckpt.get('best_count_mae', float('inf'))
            epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
            backbone_unfrozen = ckpt.get('backbone_unfrozen', False)

            if backbone_unfrozen:
                model.unfreeze_backbone(args.unfreeze_blocks)
                optimizer = torch.optim.AdamW([
                    {'params': [p for n, p in model.named_parameters()
                               if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.01},
                    {'params': [p for n, p in model.named_parameters()
                               if 'backbone' not in n and p.requires_grad], 'lr': args.lr * 0.1},
                ], weight_decay=args.weight_decay)

            print(f"  Resuming from epoch {start_epoch}, best F3: {best_f3:.4f}, best MAE: {best_count_mae:.4f}")

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING (Classification + Count Regression)")
    print(f"  Loss weights: cls={args.cls_loss_weight}, count={args.count_loss_weight}, "
          f"density={args.density_loss_weight}")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze backbone
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

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-7
            )

        t0 = time.time()
        train_m = train_epoch(
            model, train_loader, optimizer, device, epoch,
            is_tiled=use_tiled_train,
            cls_loss_weight=args.cls_loss_weight,
            count_loss_weight=args.count_loss_weight,
            density_loss_weight=args.density_loss_weight,
            pos_weight=args.pos_weight,
        )
        scheduler.step()

        log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | "
        log += f"loss={train_m['loss']:.4f} "
        log += f"(cls={train_m['cls_loss']:.3f} cnt={train_m['count_loss']:.3f}) "
        log += f"acc={train_m['acc']:.3f} MAE={train_m['count_mae']:.2f}"

        improved = False
        if val_loader:
            val_m = evaluate(model, val_loader, device, is_tiled=use_tiled_val, threshold=args.threshold)
            log += f" | P={val_m['precision']:.3f} R={val_m['recall']:.3f} "
            log += f"F3={val_m['f3']:.3f} MAE={val_m['count_mae']:.2f}"

            # Save best model (by F3, but also track count MAE)
            if val_m['f3'] > best_f3:
                best_f3 = val_m['f3']
                best_count_mae = val_m['count_mae']
                epochs_without_improvement = 0
                improved = True
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f3': best_f3,
                    'best_count_mae': best_count_mae,
                    'threshold': args.threshold,
                    'backbone': args.backbone,
                    'hidden_dim': args.hidden_dim,
                    'dropout': args.dropout,
                    'backbone_unfrozen': backbone_unfrozen,
                    'count_strategy': args.count_strategy,
                    'use_registers': args.use_registers,
                }, output_dir / 'best.pth')
                log += " ★"
            else:
                epochs_without_improvement += 1
                if args.early_stopping > 0:
                    log += f" ({epochs_without_improvement}/{args.early_stopping})"

        print(log)

        # Visualize validation errors periodically
        if val_loader and args.visualize_every > 0 and (epoch % args.visualize_every == 0 or improved):
            vis_dir = output_dir / 'visualizations'
            vis_stats = visualize_validation_errors(
                model=model,
                loader=val_loader,
                device=device,
                output_dir=vis_dir,
                epoch=epoch,
                threshold=args.threshold,
                max_samples=args.max_vis_samples,
                is_tiled=use_tiled_val,
            )
            if vis_stats:
                print(f"  [Vis] FN={vis_stats['fn']} FP={vis_stats['fp']} "
                      f"TP={vis_stats['tp']} TN={vis_stats['tn']}")

            # Also visualize count errors
            visualize_count_errors(
                model=model,
                loader=val_loader,
                device=device,
                output_dir=vis_dir,
                epoch=epoch,
                is_tiled=use_tiled_val,
                max_samples=args.max_vis_samples,
            )

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f3': best_f3,
            'best_count_mae': best_count_mae,
            'backbone': args.backbone,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'backbone_unfrozen': backbone_unfrozen,
            'epochs_without_improvement': epochs_without_improvement,
            'count_strategy': args.count_strategy,
            'use_registers': args.use_registers,
        }, output_dir / 'latest.pth')

        # Early stopping
        if args.early_stopping > 0 and epoch >= args.min_epochs:
            if epochs_without_improvement >= args.early_stopping:
                print(f"\n*** Early stopping after {epochs_without_improvement} epochs ***")
                break

    print("\n" + "=" * 80)
    print(f"Training complete! Best F3: {best_f3:.4f}, Best MAE: {best_count_mae:.4f}")
    print("=" * 80)

    # Final evaluation
    if val_loader:
        print("\nFinal evaluation on validation set:")

        best_path = output_dir / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

        val_m = evaluate(model, val_loader, device, is_tiled=use_tiled_val, threshold=args.threshold)

        print(f"\n  Classification (threshold={args.threshold}):")
        print(f"    Precision: {val_m['precision']:.4f}")
        print(f"    Recall: {val_m['recall']:.4f}")
        print(f"    F1: {val_m['f1']:.4f}")
        print(f"    F3: {val_m['f3']:.4f}")

        print(f"\n  Count Regression:")
        print(f"    MAE (all tiles): {val_m['count_mae']:.4f}")
        print(f"    RMSE: {val_m['count_rmse']:.4f}")
        print(f"    MAE (positive tiles): {val_m['count_mae_pos']:.4f}")
        print(f"    MAPE (positive tiles): {val_m['count_mape']:.2f}%")

        # Threshold optimization
        opt_thresh, opt_f3, opt_metrics = find_optimal_threshold(
            val_m['all_probs'], val_m['all_labels'], beta=3.0
        )
        print(f"\n  Optimal threshold (F3): {opt_thresh:.3f}")
        print(f"    F3: {opt_f3:.4f}, P={opt_metrics['precision']:.4f}, R={opt_metrics['recall']:.4f}")

        # Count error analysis
        print("\n  Count Error Analysis:")
        counts_pred = val_m['all_counts_pred']
        counts_gt = val_m['all_counts_gt']

        # By count range
        for low, high in [(0, 0), (1, 1), (2, 5), (6, 10), (11, float('inf'))]:
            mask = (counts_gt >= low) & (counts_gt <= high)
            if mask.any():
                mae = np.abs(counts_pred[mask] - counts_gt[mask]).mean()
                n = mask.sum()
                label = f"{low}" if low == high else f"{low}-{int(high) if high < float('inf') else '+'}"
                print(f"    Count {label}: MAE={mae:.3f} (n={n})")

        # Save with optimal threshold
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimal_threshold_f3': opt_thresh,
            'optimal_f3': opt_f3,
            'count_mae': val_m['count_mae'],
            'backbone': args.backbone,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'count_strategy': args.count_strategy,
            'use_registers': args.use_registers,
        }, output_dir / 'best_with_threshold.pth')
        logger.info(f"Saved to {output_dir / 'best_with_threshold.pth'}")

if __name__ == '__main__':
    main()