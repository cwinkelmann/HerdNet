"""
Iguana Presence Classifier with Count Regression

Extended version that includes:
1. Binary classification: Does this tile contain an iguana?
2. Count regression: How many iguanas are in this tile?
3. Attention map visualization

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
    """Deterministic dataset that tiles full images into non-overlapping crops."""

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

        counts = []
        n_pos = 0
        for name, cx, cy in self.tiles:
            count = self._count_points_in_tile(name, cx, cy)
            counts.append(count)
            if count > 0:
                n_pos += 1

        counts = np.array(counts)
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
        ])

    def _count_points_in_tile(self, name: str, crop_x: int, crop_y: int) -> int:
        points = self.annotations[name]
        if len(points) == 0:
            return 0
        in_crop = (
            (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
            (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
        )
        return in_crop.sum()

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

    def _create_density_map(self, points_in_crop: np.ndarray) -> torch.Tensor:
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
        count = float(len(points_in_crop))
        density_map = self._create_density_map(points_in_crop)

        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']
        crop = self.normalize_transform(image=crop)['image']

        return crop, {
            'label': torch.tensor(label, dtype=torch.float32),
            'count': torch.tensor(count, dtype=torch.float32),
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
    """Dataset that extracts random crops and labels them by iguana presence."""

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
        print(f"IguanaPresenceDataset: {len(self.image_names)} images, {total_points} points")
        print(f"  Crop size: {crop_size}, Crops per image: {crops_per_image}")

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
        ])

    def _sample_positive_crop(self, points: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int]:
        valid_points = [pt for pt in points if (
            pt[0] >= self.min_edge_margin and pt[0] <= img_w - self.min_edge_margin and
            pt[1] >= self.min_edge_margin and pt[1] <= img_h - self.min_edge_margin
        )]
        if not valid_points:
            valid_points = points.tolist()
        pt = random.choice(valid_points)
        px, py = pt
        x_min = max(0, int(px - self.crop_size + self.min_edge_margin))
        x_max = min(img_w - self.crop_size, int(px - self.min_edge_margin))
        y_min = max(0, int(py - self.crop_size + self.min_edge_margin))
        y_max = min(img_h - self.crop_size, int(py - self.min_edge_margin))
        x_min, x_max = min(x_min, max(0, img_w - self.crop_size)), max(x_max, 0)
        y_min, y_max = min(y_min, max(0, img_h - self.crop_size)), max(y_max, 0)
        return random.randint(min(x_min, x_max), max(x_min, x_max)), random.randint(min(y_min, y_max), max(y_min, y_max))

    def _sample_negative_crop(self, points: np.ndarray, img_w: int, img_h: int, max_attempts: int = 50):
        max_x, max_y = max(0, img_w - self.crop_size), max(0, img_h - self.crop_size)
        for _ in range(max_attempts):
            crop_x = random.randint(0, max_x) if max_x > 0 else 0
            crop_y = random.randint(0, max_y) if max_y > 0 else 0
            in_crop = ((points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
                       (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size))
            if not in_crop.any():
                return crop_x, crop_y
        return None

    def _get_points_in_crop(self, points: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        in_crop = ((points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
                   (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size))
        pts = points[in_crop].copy()
        pts[:, 0] -= crop_x
        pts[:, 1] -= crop_y
        return pts

    def _create_density_map(self, points_in_crop: np.ndarray) -> torch.Tensor:
        density_map = torch.zeros(self.grid_size, self.grid_size)
        for pt in points_in_crop:
            px, py = int(pt[0] / self.patch_size), int(pt[1] / self.patch_size)
            px = max(0, min(px, self.grid_size - 1))
            py = max(0, min(py, self.grid_size - 1))
            density_map[py, px] += 1.0
        return density_map

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
            if result:
                crop_x, crop_y = result
            elif len(points) > 0:
                crop_x, crop_y = self._sample_positive_crop(points, img_w, img_h)
            else:
                crop_x = random.randint(0, max(0, img_w - self.crop_size))
                crop_y = random.randint(0, max(0, img_h - self.crop_size))

        crop = img[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]
        if crop.shape[0] < self.crop_size or crop.shape[1] < self.crop_size:
            padded = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        points_in_crop = self._get_points_in_crop(points, crop_x, crop_y)
        label = 1.0 if len(points_in_crop) > 0 else 0.0
        count = float(len(points_in_crop))
        density_map = self._create_density_map(points_in_crop)

        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']
        crop = self.normalize_transform(image=crop)['image']

        return crop, {'label': torch.tensor(label), 'count': torch.tensor(count), 'density_map': density_map}


def presence_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {
        'label': torch.stack([b[1]['label'] for b in batch]),
        'count': torch.stack([b[1]['count'] for b in batch]),
        'density_map': torch.stack([b[1]['density_map'] for b in batch]),
    }
    return images, targets


# =============================================================================
# ATTENTION EXTRACTION
# =============================================================================

class AttentionExtractor:
    """Extract attention maps from DINOv2/ViT models using hooks."""

    def __init__(self, model: nn.Module, layer_indices: List[int] = [-1], num_prefix_tokens: int = 1):
        """
        Args:
            model: The backbone model
            layer_indices: Which transformer blocks to extract attention from
            num_prefix_tokens: Number of prefix tokens (CLS + registers) to skip
        """
        self.model = model
        self.layer_indices = layer_indices
        self.num_prefix_tokens = num_prefix_tokens
        self.attention_maps = {}
        self.hooks = []

        if hasattr(model, 'blocks'):
            blocks = model.blocks
        else:
            logger.warning("Could not find transformer blocks for attention extraction")
            return

        n_blocks = len(blocks)

        for idx in layer_indices:
            actual_idx = idx if idx >= 0 else n_blocks + idx
            if 0 <= actual_idx < n_blocks:
                block = blocks[actual_idx]
                if hasattr(block, 'attn'):
                    hook = block.attn.register_forward_hook(self._make_hook(f'block_{actual_idx}'))
                    self.hooks.append(hook)

        logger.info(f"AttentionExtractor: Registered {len(self.hooks)} hooks on layers {layer_indices}, "
                    f"num_prefix_tokens={num_prefix_tokens}")

    def _make_hook(self, name: str):
        num_prefix = self.num_prefix_tokens
        def hook(module, input, output):
            if hasattr(module, 'qkv'):
                x = input[0]
                B, N, C = x.shape
                qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                scale = (C // module.num_heads) ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                # CLS attention to patches only (skip all prefix tokens: CLS + registers)
                cls_attn = attn[:, :, 0, num_prefix:].mean(dim=1)
                self.attention_maps[name] = cls_attn.detach()
        return hook

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        maps = self.attention_maps.copy()
        self.attention_maps = {}
        return maps

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# MODEL
# =============================================================================

class IguanaClassifierWithCount(nn.Module):
    """Binary classifier + count regressor using DINOv2 backbone with attention extraction."""

    def __init__(
        self,
        backbone: str = 'vit_large_patch14_reg4_dinov2.lvd142m',
        freeze_backbone: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        count_strategy: str = 'both',
        use_registers: bool = True,
        pretrained: bool = True,
        extract_attention: bool = True,
        attention_layers: List[int] = [-4, -3, -2, -1],
    ):
        super().__init__()

        self.count_strategy = count_strategy
        self.use_registers = use_registers
        self.extract_attention = extract_attention

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)
        self.num_register_tokens = max(0, self.num_prefix_tokens - 1)

        self.attention_extractor = None
        if extract_attention:
            self.attention_extractor = AttentionExtractor(
                self.backbone, attention_layers, num_prefix_tokens=self.num_prefix_tokens
            )

        self.global_feat_dim = self.feat_dim
        if use_registers and self.num_register_tokens > 0:
            self.register_attention = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 4, 1),
            )
        else:
            self.register_attention = None

        print(f"Backbone: {backbone}")
        print(f"  Feature dim: {self.feat_dim}, Patch size: {self.patch_size}")
        print(f"  Prefix tokens: {self.num_prefix_tokens} (1 CLS + {self.num_register_tokens} registers)")
        print(f"  Extract attention: {extract_attention}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen")

        self.cls_head = nn.Sequential(
            nn.Linear(self.global_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

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

        if count_strategy in ['density', 'both']:
            self.density_head = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.density_head = None

        # Only create patch head if we're doing counting (for visualization)
        if count_strategy != 'none':
            self.patch_head = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.patch_head = None

        for head in [self.cls_head, self.count_head_cls, self.density_head, self.patch_head, self.register_attention]:
            if head is not None:
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def _get_global_features(self, features: torch.Tensor) -> torch.Tensor:
        cls_token = features[:, 0]
        if self.use_registers and self.num_register_tokens > 0 and self.register_attention is not None:
            register_tokens = features[:, 1:1+self.num_register_tokens]
            global_tokens = torch.cat([cls_token.unsqueeze(1), register_tokens], dim=1)
            attn_logits = self.register_attention(global_tokens).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=-1)
            global_feat = (attn_weights.unsqueeze(-1) * global_tokens).sum(dim=1)
            return global_feat
        return cls_token

    def forward(
        self,
        x: torch.Tensor,
        return_patches: bool = False,
        return_density: bool = False,
        return_attention: bool = False,
        upsample_patches: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        grid_size = H // self.patch_size

        features = self.backbone.forward_features(x)

        attention_maps = {}
        if return_attention and self.attention_extractor is not None:
            attention_maps = self.attention_extractor.get_attention_maps()

        global_feat = self._get_global_features(features)
        patch_tokens = features[:, self.num_prefix_tokens:]
        patch_tokens = patch_tokens.view(B, grid_size, grid_size, self.feat_dim)

        output = {}
        output['cls_logit'] = self.cls_head(global_feat).squeeze(-1)

        if self.count_head_cls is not None:
            output['count_cls'] = F.softplus(self.count_head_cls(global_feat).squeeze(-1))

        if self.density_head is not None:
            density_logits = self.density_head(patch_tokens).squeeze(-1)
            density_pred = F.softplus(density_logits)
            output['count_density'] = density_pred.sum(dim=(1, 2))
            if return_density:
                if upsample_patches:
                    density_pred = F.interpolate(density_pred.unsqueeze(1), size=(H, W),
                                                  mode='bilinear', align_corners=False).squeeze(1)
                output['density_map'] = density_pred

        if self.count_strategy == 'cls':
            output['count'] = output['count_cls']
        elif self.count_strategy == 'density':
            output['count'] = output['count_density']
        elif self.count_strategy == 'both':
            output['count'] = (output['count_cls'] + output['count_density']) / 2
        # For count_strategy='none', no count output is added

        if return_patches and self.patch_head is not None:
            patch_logits = self.patch_head(patch_tokens).squeeze(-1)
            if upsample_patches:
                patch_logits = F.interpolate(patch_logits.unsqueeze(1), size=(H, W),
                                              mode='bilinear', align_corners=False).squeeze(1)
            output['patch_logits'] = patch_logits

        if return_attention and attention_maps:
            attn_list = list(attention_maps.values())
            if attn_list:
                avg_attn = torch.stack(attn_list, dim=0).mean(dim=0)
                attn_map = avg_attn.view(B, grid_size, grid_size)
                if upsample_patches:
                    attn_map = F.interpolate(attn_map.unsqueeze(1), size=(H, W),
                                              mode='bilinear', align_corners=False).squeeze(1)
                output['attention_map'] = attn_map

        if return_attention:
            feat_magnitude = torch.norm(patch_tokens, dim=-1)
            feat_min = feat_magnitude.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
            feat_max = feat_magnitude.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
            feat_magnitude = (feat_magnitude - feat_min) / (feat_max - feat_min + 1e-8)
            if upsample_patches:
                feat_magnitude = F.interpolate(feat_magnitude.unsqueeze(1), size=(H, W),
                                                mode='bilinear', align_corners=False).squeeze(1)
            output['feature_map'] = feat_magnitude

        return output

    def unfreeze_backbone(self, n_blocks: Optional[int] = None):
        if n_blocks is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            if hasattr(self.backbone, 'blocks'):
                total = len(self.backbone.blocks)
                for i, block in enumerate(self.backbone.blocks):
                    if i >= total - n_blocks:
                        for p in block.parameters():
                            p.requires_grad = True

    def __del__(self):
        if hasattr(self, 'attention_extractor') and self.attention_extractor is not None:
            self.attention_extractor.remove_hooks()


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, epoch=0, is_tiled=False,
                cls_loss_weight=1.0, count_loss_weight=1.0, density_loss_weight=0.5, pos_weight=1.0):
    model.train()
    total_loss = total_cls_loss = total_count_loss = total_density_loss = 0
    total_correct = total_samples = total_count_mae = 0
    cls_pos_weight = torch.tensor([pos_weight], device=device)

    for images, targets in loader:
        images = images.to(device)
        labels = targets['label'].to(device)
        counts = targets['count'].to(device)
        density_gt = targets.get('density_map')
        if density_gt is not None:
            density_gt = density_gt.to(device)

        optimizer.zero_grad()
        output = model(images, return_density=(density_gt is not None))

        cls_loss = F.binary_cross_entropy_with_logits(output['cls_logit'], labels,
                                                       pos_weight=cls_pos_weight.expand_as(labels))

        # Count loss (only if count output exists)
        count_loss = torch.tensor(0.0, device=device)
        if 'count' in output:
            count_loss = F.smooth_l1_loss(output['count'], counts)

        density_loss = torch.tensor(0.0, device=device)
        if density_gt is not None and 'density_map' in output:
            density_loss = F.smooth_l1_loss(output['density_map'], density_gt)

        loss = cls_loss_weight * cls_loss + count_loss_weight * count_loss + density_loss_weight * density_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_cls_loss += cls_loss.item() * len(labels)
        total_count_loss += count_loss.item() * len(labels)
        total_density_loss += density_loss.item() * len(labels)

        probs = torch.sigmoid(output['cls_logit'])
        preds = (probs > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        if 'count' in output:
            total_count_mae += torch.abs(output['count'] - counts).sum().item()

    n = total_samples
    return {
        'loss': total_loss / n, 'cls_loss': total_cls_loss / n,
        'count_loss': total_count_loss / n, 'density_loss': total_density_loss / n,
        'acc': total_correct / n, 'count_mae': total_count_mae / n,
    }


@torch.no_grad()
def evaluate(model, loader, device, is_tiled=False, threshold=0.5):
    model.eval()
    total_loss = total_samples = 0
    all_preds, all_labels, all_probs, all_counts_pred, all_counts_gt = [], [], [], [], []

    for images, targets in loader:
        images = images.to(device)
        labels = targets['label'].to(device)
        counts = targets['count'].to(device)

        output = model(images)
        cls_loss = F.binary_cross_entropy_with_logits(output['cls_logit'], labels)

        # Count loss only if available
        if 'count' in output:
            count_loss = F.smooth_l1_loss(output['count'], counts)
            loss = cls_loss + count_loss
        else:
            loss = cls_loss

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

        probs = torch.sigmoid(output['cls_logit'])
        preds = (probs > threshold).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        if 'count' in output:
            all_counts_pred.extend(output['count'].cpu().numpy())
        else:
            all_counts_pred.extend([0.0] * len(labels))
        all_counts_gt.extend(counts.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_counts_pred = np.array(all_counts_pred)
    all_counts_gt = np.array(all_counts_gt)

    accuracy = (all_preds == all_labels).mean()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    f3 = 10 * precision * recall / max(9 * precision + recall, 1e-6)
    count_mae = np.abs(all_counts_pred - all_counts_gt).mean()

    return {
        'loss': total_loss / total_samples, 'acc': accuracy,
        'precision': precision, 'recall': recall, 'f1': f1, 'f3': f3,
        'count_mae': count_mae, 'all_probs': all_probs, 'all_labels': all_labels,
        'all_counts_pred': all_counts_pred, 'all_counts_gt': all_counts_gt,
    }


def visualize_validation_errors(model, loader, device, output_dir, epoch, threshold=0.3, max_samples=16, is_tiled=True):
    if not HAS_MATPLOTLIB:
        return
    model.eval()
    all_results = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            labels = targets['label']
            counts = targets['count']
            points_list = targets.get('points', [None] * len(labels))

            output = model(images, return_patches=True, return_density=True, return_attention=True)
            probs = torch.sigmoid(output['cls_logit']).cpu().numpy()

            # Handle missing count output (classification_only mode)
            if 'count' in output:
                count_preds = output['count'].cpu().numpy()
            else:
                count_preds = np.zeros(len(labels))

            density_maps = output.get('density_map')
            attention_maps = output.get('attention_map')
            feature_maps = output.get('feature_map')

            for i in range(len(labels)):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

                all_results.append({
                    'image': img,
                    'prob': float(probs[i]),
                    'label': int(labels[i]),
                    'count_pred': float(count_preds[i]),
                    'count_gt': float(counts[i]),
                    'points': points_list[i].numpy() if is_tiled and points_list[i] is not None else None,
                    'density_map': density_maps[i].cpu().numpy() if density_maps is not None else None,
                    'attention_map': attention_maps[i].cpu().numpy() if attention_maps is not None else None,
                    'feature_map': feature_maps[i].cpu().numpy() if feature_maps is not None else None,
                })

    false_negatives = sorted([r for r in all_results if r['label'] == 1 and r['prob'] <= threshold], key=lambda x: x['prob'])
    false_positives = sorted([r for r in all_results if r['label'] == 0 and r['prob'] > threshold], key=lambda x: -x['prob'])
    true_positives = sorted([r for r in all_results if r['label'] == 1 and r['prob'] > threshold], key=lambda x: -x['prob'])
    true_negatives = sorted([r for r in all_results if r['label'] == 0 and r['prob'] <= threshold], key=lambda x: x['prob'])

    epoch_dir = output_dir / f'epoch_{epoch:03d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    for samples, name, title, point_color in [
        (false_negatives, 'false_negatives', 'FALSE NEGATIVES', 'lime'),
        (false_positives, 'false_positives', 'FALSE POSITIVES', 'red'),
    ]:
        if not samples:
            continue
        n_show = min(max_samples, len(samples))
        fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)

        for i, r in enumerate(samples[:n_show]):
            axes[i, 0].imshow(r['image'])
            if r['points'] is not None:
                for pt in r['points']:
                    axes[i, 0].plot(pt[0], pt[1], 'o', color=point_color, markersize=10, markerfacecolor='none', markeredgewidth=2)
            axes[i, 0].set_title(f"p={r['prob']:.3f} GT:{int(r['count_gt'])} Pred:{r['count_pred']:.1f}")
            axes[i, 0].axis('off')

            if r['attention_map'] is not None:
                axes[i, 1].imshow(r['attention_map'], cmap='plasma')
            axes[i, 1].set_title("Attention")
            axes[i, 1].axis('off')

            if r['density_map'] is not None:
                axes[i, 2].imshow(r['density_map'], cmap='hot')
            axes[i, 2].set_title(f"Density")
            axes[i, 2].axis('off')

            if r['feature_map'] is not None:
                axes[i, 3].imshow(r['feature_map'], cmap='viridis')
            axes[i, 3].set_title("Feature Mag")
            axes[i, 3].axis('off')

        fig.suptitle(f"{title} (n={len(samples)}) - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(epoch_dir / f'{name}.png', dpi=120, bbox_inches='tight')
        plt.close()

    # Attention visualization for positive tiles
    positive_with_attn = [r for r in all_results if r['label'] == 1 and r['attention_map'] is not None]
    if positive_with_attn:
        n_show = min(16, len(positive_with_attn))
        fig, axes = plt.subplots((n_show + 3) // 4, 8, figsize=(24, 3 * ((n_show + 3) // 4)))
        axes = axes.flatten()
        for i, r in enumerate(positive_with_attn[:n_show]):
            axes[i*2].imshow(r['image'])
            if r['points'] is not None:
                for pt in r['points']:
                    axes[i*2].plot(pt[0], pt[1], 'g+', markersize=8, markeredgewidth=2)
            axes[i*2].set_title(f"GT:{int(r['count_gt'])}")
            axes[i*2].axis('off')
            axes[i*2+1].imshow(r['attention_map'], cmap='plasma')
            axes[i*2+1].set_title(f"p={r['prob']:.2f}")
            axes[i*2+1].axis('off')
        for j in range(n_show*2, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f"Attention Maps - Positive Tiles - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(epoch_dir / 'attention_positive.png', dpi=120, bbox_inches='tight')
        plt.close()

    return {'fn': len(false_negatives), 'fp': len(false_positives), 'tp': len(true_positives), 'tn': len(true_negatives)}


def find_optimal_threshold(all_probs, all_labels, beta=3.0):
    best_threshold, best_fbeta = 0.5, 0
    for thresh in np.arange(0.05, 0.95, 0.025):
        preds = (all_probs > thresh).astype(float)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fbeta = (1 + beta**2) * precision * recall / max(beta**2 * precision + recall, 1e-6)
        if fbeta > best_fbeta:
            best_fbeta, best_threshold = fbeta, thresh
    return best_threshold, best_fbeta


def main():
    parser = argparse.ArgumentParser(description="Iguana Classifier with Count Regression")
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')
    parser.add_argument('--backbone', default='vit_large_patch14_reg4_dinov2.lvd142m')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--count_strategy', default='both', choices=['cls', 'density', 'both'])
    parser.add_argument('--classification_only', action='store_true',
                        help='Disable count regression entirely, train only binary classification')
    parser.add_argument('--use_registers', action='store_true', default=True)
    parser.add_argument('--no_registers', dest='use_registers', action='store_false')
    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--crops_per_image', type=int, default=8)
    parser.add_argument('--positive_ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--unfreeze_epoch', type=int, default=10)
    parser.add_argument('--unfreeze_blocks', type=int, default=4)
    parser.add_argument('--pos_weight', type=float, default=3.0)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--cls_loss_weight', type=float, default=1.0)
    parser.add_argument('--count_loss_weight', type=float, default=1.0)
    parser.add_argument('--density_loss_weight', type=float, default=0.5)
    parser.add_argument('--output_dir', default='./outputs_classifier_count')
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tiled_train', action='store_true')
    parser.add_argument('--tiled_val', action='store_true')
    parser.add_argument('--tile_overlap', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--early_stopping', type=int, default=15)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--visualize_every', type=int, default=5)
    parser.add_argument('--max_vis_samples', type=int, default=16)

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
        train_ds = IguanaTiledDataset(args.train_csv, args.train_image_dir, crop_size=args.crop_size, overlap=args.tile_overlap, augment=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=tiled_collate_fn)
    else:
        train_ds = IguanaPresenceDataset(args.train_csv, args.train_image_dir, crop_size=args.crop_size, crops_per_image=args.crops_per_image, positive_ratio=args.positive_ratio, augment=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=presence_collate_fn)

    val_loader = None
    if args.val_csv and args.val_image_dir:
        if args.tiled_val:
            val_ds = IguanaTiledDataset(args.val_csv, args.val_image_dir, crop_size=args.crop_size, overlap=args.tile_overlap)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=tiled_collate_fn)
        else:
            val_ds = IguanaPresenceDataset(args.val_csv, args.val_image_dir, crop_size=args.crop_size, crops_per_image=args.crops_per_image, positive_ratio=0.5, augment=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=presence_collate_fn)

    # Model
    # Handle classification_only mode
    count_strategy = args.count_strategy
    count_loss_weight = args.count_loss_weight
    density_loss_weight = args.density_loss_weight

    if args.classification_only:
        print("\n*** Classification-only mode: disabling count regression ***")
        count_strategy = 'none'
        count_loss_weight = 0.0
        density_loss_weight = 0.0

    model = IguanaClassifierWithCount(
        backbone=args.backbone, freeze_backbone=True, hidden_dim=args.hidden_dim,
        dropout=args.dropout, count_strategy=count_strategy,
        use_registers=args.use_registers, extract_attention=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}, Trainable: {n_train:,}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch, best_f3, best_count_mae = 0, 0, float('inf')
    epochs_without_improvement, backbone_unfrozen = 0, False

    if args.load_from and Path(args.load_from).exists():
        ckpt = torch.load(args.load_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded weights from {args.load_from}")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt.get('optimizer_state_dict', {}))
        scheduler.load_state_dict(ckpt.get('scheduler_state_dict', {}))
        start_epoch = ckpt.get('epoch', 0) + 1
        best_f3 = ckpt.get('best_f3', 0)
        backbone_unfrozen = ckpt.get('backbone_unfrozen', False)
        if backbone_unfrozen:
            model.unfreeze_backbone(args.unfreeze_blocks)
        print(f"Resuming from epoch {start_epoch}")

    print("\n" + "=" * 80)
    if args.classification_only:
        print("TRAINING (Classification Only)")
    else:
        print("TRAINING (Classification + Count + Attention)")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.unfreeze_epoch and args.unfreeze_blocks > 0 and not backbone_unfrozen:
            print(f"\n*** Unfreezing last {args.unfreeze_blocks} backbone blocks ***")
            model.unfreeze_backbone(args.unfreeze_blocks)
            backbone_unfrozen = True
            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.01},
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': args.lr * 0.1},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-7)

        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, device, epoch, is_tiled=args.tiled_train,
                              cls_loss_weight=args.cls_loss_weight, count_loss_weight=count_loss_weight,
                              density_loss_weight=density_loss_weight, pos_weight=args.pos_weight)
        scheduler.step()

        if args.classification_only:
            log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | loss={train_m['loss']:.4f} acc={train_m['acc']:.3f}"
        else:
            log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} MAE={train_m['count_mae']:.2f}"

        improved = False
        if val_loader:
            val_m = evaluate(model, val_loader, device, is_tiled=args.tiled_val, threshold=args.threshold)
            if args.classification_only:
                log += f" | P={val_m['precision']:.3f} R={val_m['recall']:.3f} F3={val_m['f3']:.3f}"
            else:
                log += f" | P={val_m['precision']:.3f} R={val_m['recall']:.3f} F3={val_m['f3']:.3f} MAE={val_m['count_mae']:.2f}"

            if val_m['f3'] > best_f3:
                best_f3 = val_m['f3']
                best_count_mae = val_m['count_mae']
                epochs_without_improvement = 0
                improved = True
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'best_f3': best_f3, 'best_count_mae': best_count_mae, 'threshold': args.threshold,
                    'backbone': args.backbone, 'hidden_dim': args.hidden_dim, 'backbone_unfrozen': backbone_unfrozen,
                    'count_strategy': args.count_strategy, 'use_registers': args.use_registers,
                }, output_dir / 'best.pth')
                log += " ★"
            else:
                epochs_without_improvement += 1
                log += f" ({epochs_without_improvement}/{args.early_stopping})"

        print(log)

        if val_loader and args.visualize_every > 0 and (epoch % args.visualize_every == 0 or improved):
            vis_stats = visualize_validation_errors(model, val_loader, device, output_dir / 'visualizations', epoch, threshold=args.threshold, max_samples=args.max_vis_samples, is_tiled=args.tiled_val)
            if vis_stats:
                print(f"  [Vis] FN={vis_stats['fn']} FP={vis_stats['fp']} TP={vis_stats['tp']} TN={vis_stats['tn']}")

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_f3': best_f3, 'backbone': args.backbone, 'backbone_unfrozen': backbone_unfrozen,
        }, output_dir / 'latest.pth')

        if args.early_stopping > 0 and epoch >= args.min_epochs and epochs_without_improvement >= args.early_stopping:
            print(f"\n*** Early stopping ***")
            break

    print("\n" + "=" * 80)
    print(f"Training complete! Best F3: {best_f3:.4f}, Best MAE: {best_count_mae:.4f}")
    print("=" * 80)

    if val_loader:
        ckpt = torch.load(output_dir / 'best.pth', map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        val_m = evaluate(model, val_loader, device, is_tiled=args.tiled_val, threshold=args.threshold)
        opt_thresh, opt_f3 = find_optimal_threshold(val_m['all_probs'], val_m['all_labels'])
        print(f"\nOptimal threshold: {opt_thresh:.3f}, F3: {opt_f3:.4f}")

        torch.save({
            'model_state_dict': model.state_dict(), 'optimal_threshold_f3': opt_thresh, 'optimal_f3': opt_f3,
            'count_mae': val_m['count_mae'], 'backbone': args.backbone, 'hidden_dim': args.hidden_dim,
            'count_strategy': args.count_strategy, 'use_registers': args.use_registers,
        }, output_dir / 'best_with_threshold.pth')

        # Save backbone-only weights (compatible with plain timm model)
        # Strip "backbone." prefix so it can be loaded directly into timm.create_model()
        backbone_state_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('backbone.'):
                # Remove "backbone." prefix
                new_key = k[len('backbone.'):]
                backbone_state_dict[new_key] = v

        torch.save({
            'state_dict': backbone_state_dict,
            'model_name': args.backbone,
            'source': 'iguana_classifier_with_count',
            'optimal_threshold_f3': opt_thresh,
            'count_mae': val_m['count_mae'],
        }, output_dir / 'backbone_only.pth')
        print(f"Saved backbone weights to {output_dir / 'backbone_only.pth'}")
        print(f"  Load with: backbone = timm.create_model('{args.backbone}', pretrained=False)")
        print(f"             backbone.load_state_dict(torch.load('backbone_only.pth')['state_dict'])")


if __name__ == '__main__':
    main()