#!/usr/bin/env python3
"""
HerdNet-based Iguana Detector with FIDT and LMDS

This module integrates:
1. HerdNetDINOv2 - DINOv2 backbone with attention-based feature extraction
2. FIDT - Focal Inverse Distance Transform for point-to-heatmap conversion
3. LMDS - Local Maxima Detection Strategy for heatmap-to-point inference

The model learns to predict heatmaps from point annotations and uses LMDS
to extract detected points during inference.

v2: Adds auxiliary classification and count regression heads for multi-task learning
"""

import os
import argparse
import random
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import scipy.ndimage
from loguru import logger

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    raise ImportError("albumentations required: pip install albumentations")

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# FIDT: Focal Inverse Distance Transform
# =============================================================================

def point_buffer(x: int, y: int, mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Create circular buffer around a point."""
    x_t, y_t = torch.arange(0, mask.size(1)), torch.arange(0, mask.size(0))
    buffer = (x_t.unsqueeze(0) - x) ** 2 + (y_t.unsqueeze(1) - y) ** 2 < radius ** 2
    return buffer


class FIDT:
    """
    Focal Inverse Distance Transform.

    Converts point annotations into density-like heatmaps for training.
    """

    def __init__(
            self,
            alpha: float = 0.02,
            beta: float = 0.75,
            c: float = 1.0,
            radius: int = 1,
            down_ratio: int = 4,
    ):
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.radius = radius
        self.down_ratio = down_ratio

    def __call__(
            self,
            points: np.ndarray,
            img_height: int,
            img_width: int,
    ) -> torch.Tensor:
        """Convert points to FIDT map."""
        out_h = img_height // self.down_ratio
        out_w = img_width // self.down_ratio

        # If no points, return zeros (NOT ones - EDT of ones gives corner artifacts)
        if len(points) == 0:
            return torch.zeros((1, out_h, out_w))

        # Create binary mask with points (1 = background, 0 = point location)
        mask = torch.ones((out_h, out_w))
        points_ds = points / self.down_ratio

        for pt in points_ds:
            x, y = int(pt[0]), int(pt[1])
            x = max(0, min(x, out_w - 1))
            y = max(0, min(y, out_h - 1))
            buffer = point_buffer(x, y, mask, self.radius)
            mask[buffer] = 0

        # Compute distance transform
        dist_map = scipy.ndimage.distance_transform_edt(mask.numpy())
        dist_map = torch.from_numpy(dist_map).float()

        # Apply FIDT formula
        fidt_map = 1 / (torch.pow(dist_map, self.alpha * dist_map + self.beta) + self.c)
        fidt_map = torch.where(fidt_map < 0.01, torch.zeros_like(fidt_map), fidt_map)

        return fidt_map.unsqueeze(0)


class GaussianMap:
    """Gaussian density map generator."""

    def __init__(
            self,
            sigma: float = 2.0,
            radius: int = 1,
            down_ratio: int = 4,
    ):
        self.sigma = sigma
        self.radius = radius
        self.down_ratio = down_ratio

    def __call__(
            self,
            points: np.ndarray,
            img_height: int,
            img_width: int,
    ) -> torch.Tensor:
        """Convert points to Gaussian density map."""
        out_h = img_height // self.down_ratio
        out_w = img_width // self.down_ratio

        density = np.zeros((out_h, out_w), dtype=np.float32)

        if len(points) > 0:
            points_ds = points / self.down_ratio
            for pt in points_ds:
                x, y = int(pt[0]), int(pt[1])
                x = max(0, min(x, out_w - 1))
                y = max(0, min(y, out_h - 1))
                pt_map = np.zeros((out_h, out_w), dtype=np.float32)
                pt_map[y, x] = 1.0
                density += scipy.ndimage.gaussian_filter(pt_map, self.sigma, mode='reflect')

        return torch.from_numpy(density).unsqueeze(0)


# =============================================================================
# LMDS: Local Maxima Detection Strategy
# =============================================================================

class LMDS:
    """Local Maxima Detection Strategy for extracting points from heatmaps."""

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (3, 3),
            adapt_ts: float = 0.2,  # Lowered from 0.39
            neg_ts: float = 0.05,  # Lowered from 0.1
            score_threshold: float = 0.1,  # Lowered from 0.3
    ):
        assert kernel_size[0] == kernel_size[1], "Kernel must be square"
        assert kernel_size[0] % 2 == 1, "Kernel size must be odd"

        self.kernel_size = tuple(kernel_size)
        self.adapt_ts = adapt_ts
        self.neg_ts = neg_ts
        self.score_threshold = score_threshold

    def __call__(
            self,
            heatmap: torch.Tensor,
            scale_factor: int = 1,
    ) -> Tuple[List, List, List]:
        """Extract points from heatmap."""
        if heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(1)

        batch_size = heatmap.shape[0]
        b_counts, b_locs, b_scores = [], [], []

        for b in range(batch_size):
            count, locs, scores = self._detect_single(heatmap[b, 0])
            if scale_factor != 1:
                locs = [(y * scale_factor, x * scale_factor) for y, x in locs]
            b_counts.append(count)
            b_locs.append(locs)
            b_scores.append(scores)

        return b_counts, b_locs, b_scores

    def _local_max(self, est_map: torch.Tensor) -> torch.Tensor:
        """Find local maxima using max pooling."""
        pad = int(self.kernel_size[0] / 2)
        est_map_4d = est_map.unsqueeze(0).unsqueeze(0)
        keep = F.max_pool2d(est_map_4d, kernel_size=self.kernel_size, stride=1, padding=pad)
        keep = (keep == est_map_4d).float()
        return (keep * est_map_4d).squeeze(0).squeeze(0)

    def _detect_single(self, est_map: torch.Tensor) -> Tuple[int, List, List]:
        """Detect points in a single heatmap."""
        est_map_max = torch.max(est_map).item()

        local_max_map = self._local_max(est_map)
        threshold = self.adapt_ts * est_map_max
        local_max_map[local_max_map < threshold] = 0
        local_max_map[local_max_map < self.score_threshold] = 0

        if est_map_max < self.neg_ts:
            return 0, [], []

        scores_map = local_max_map.clone()
        local_max_map[local_max_map > 0] = 1

        locs_np = local_max_map.cpu().numpy()
        scores_np = scores_map.cpu().numpy()

        locs = []
        scores = []
        for i, j in np.argwhere(locs_np > 0):
            locs.append((int(i), int(j)))
            scores.append(float(scores_np[i, j]))

        return len(locs), locs, scores


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class SimpleDINOv2Extractor(nn.Module):
    """Simplified DINOv2 feature extractor."""

    def __init__(self, dinov2_model):
        super().__init__()
        self.dinov2 = dinov2_model
        self.hooks = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Returns:
            patch_features: [B, N, D] patch tokens only
            attention_maps: Dict of attention maps
            all_tokens: [B, 1+num_reg+N, D] all tokens including CLS and registers
        """
        all_tokens = self.dinov2.forward_features(x)

        num_prefix = getattr(self.dinov2, 'num_prefix_tokens', 1)
        if num_prefix > 0:
            patch_features = all_tokens[:, num_prefix:]
        else:
            patch_features = all_tokens

        B, N, D = patch_features.shape

        # Create feature-based attention
        feature_attention = torch.norm(patch_features, dim=2)
        feature_attention = (feature_attention - feature_attention.min(dim=1, keepdim=True)[0]) / \
                            (feature_attention.max(dim=1, keepdim=True)[0] -
                             feature_attention.min(dim=1, keepdim=True)[0] + 1e-8)

        attention_maps = {0: feature_attention}

        return patch_features, attention_maps, all_tokens

    def remove_hooks(self):
        pass


class DINOv2AttentionExtractor(nn.Module):
    """Extract spatial attention maps from DINOv2 transformer blocks."""

    def __init__(self, dinov2_model, layer_indices: List[int] = [-4, -3, -2, -1]):
        super().__init__()
        self.dinov2 = dinov2_model
        self.layer_indices = layer_indices
        self.attention_maps = {}
        self.hooks = []

        for i, layer_idx in enumerate(layer_indices):
            target_layer = self.dinov2.blocks[layer_idx].attn
            hook = target_layer.register_forward_hook(
                lambda module, input, output, idx=i: self._save_attention(module, input, output, idx)
            )
            self.hooks.append(hook)

    def _save_attention(self, module, input, output, layer_idx):
        """Extract attention weights."""
        x = input[0]
        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_weights = (q @ k.transpose(-2, -1)) * module.scale
        attn_weights = attn_weights.softmax(dim=-1)

        if N > 1:
            num_prefix = getattr(self.dinov2, 'num_prefix_tokens', 1)
            cls_attention = attn_weights[:, :, 0, num_prefix:].mean(dim=1)
            self.attention_maps[layer_idx] = cls_attention.detach()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Returns:
            patch_features: [B, N, D] patch tokens only
            attention_maps: Dict of attention maps
            all_tokens: [B, 1+num_reg+N, D] all tokens including CLS and registers
        """
        self.attention_maps.clear()

        all_tokens = self.dinov2.forward_features(x)

        num_prefix = getattr(self.dinov2, 'num_prefix_tokens', 1)
        if num_prefix > 0:
            patch_features = all_tokens[:, num_prefix:]
        else:
            patch_features = all_tokens

        return patch_features, self.attention_maps, all_tokens

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class DINOv2SpatialProcessor(nn.Module):
    """Process DINOv2 features into multi-scale representations."""

    def __init__(self, feature_dim: int, output_channels: List[int] = [256, 512, 1024]):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_channels = output_channels

        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, out_ch),
                nn.LayerNorm(out_ch),
                nn.GELU()
            ) for out_ch in output_channels
        ])

        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for out_ch in output_channels
        ])

    def forward(self, patch_features, attention_maps):
        B, N, D = patch_features.shape
        H = W = int(N ** 0.5)

        multi_scale_features = []
        attention_heatmaps = []

        for i, (projector, conv) in enumerate(zip(self.scale_projectors, self.spatial_convs)):
            projected = projector(patch_features)
            spatial_feat = projected.transpose(1, 2).reshape(B, -1, H, W)
            processed_feat = conv(spatial_feat)

            if i == 0:
                scale_feat = F.interpolate(processed_feat, scale_factor=2, mode='bilinear', align_corners=False)
            elif i == 1:
                scale_feat = processed_feat
            else:
                scale_feat = F.avg_pool2d(processed_feat, kernel_size=2, stride=2)

            multi_scale_features.append(scale_feat)

            if attention_maps and i in attention_maps:
                attention = attention_maps[i]
                attention_heatmap = attention.reshape(B, 1, H, W)
                attention_heatmap = F.interpolate(attention_heatmap, size=scale_feat.shape[2:],
                                                  mode='bilinear', align_corners=False)
                attention_heatmaps.append(attention_heatmap)
            else:
                attention_heatmaps.append(
                    torch.ones(B, 1, *scale_feat.shape[2:], device=scale_feat.device) * 0.5
                )

        return multi_scale_features, attention_heatmaps


class DINOv2FeatureUpsampler(nn.Module):
    """Upsample and fuse DINOv2 features."""

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])

        self.attention_fusion = nn.Sequential(
            nn.Conv2d(len(in_channels), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, attention_maps):
        projected_features = []
        target_size = features[0].shape[2:]

        for i, (feat, proj) in enumerate(zip(features, self.projects)):
            projected = proj(feat)
            if projected.shape[2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode='bilinear', align_corners=False)
            projected_features.append(projected)

        fused_features = sum(projected_features)

        attention_stack = []
        for attention in attention_maps:
            if attention.shape[2:] != target_size:
                attention = F.interpolate(attention, size=target_size, mode='bilinear', align_corners=False)
            attention_stack.append(attention)

        if attention_stack:
            attention_tensor = torch.cat(attention_stack, dim=1)
            attention_features = self.attention_fusion(attention_tensor)
            fused_features = fused_features + attention_features

        return fused_features


# =============================================================================
# MAIN MODEL
# =============================================================================

class HerdNetDINOv2(nn.Module):
    """
    HerdNet with DINOv2 backbone for point detection.

    Includes auxiliary heads matching IguanaClassifierWithCount architecture
    for binary classification and count regression. This allows:
    1. Loading pretrained classifier weights
    2. Using classification to filter false positives before point detection
    """

    def __init__(
            self,
            backbone: str = 'vit_large_patch14_dinov2.lvd142m',
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: int = 4,
            head_conv: int = 64,
            hidden_dim: int = 512,
            dropout: float = 0.3,
            output_channels: List[int] = [256, 512, 1024],
            attention_layers: List[int] = [-4, -3, -2, -1],
            freeze_backbone: bool = False,
            use_simple_extractor: bool = False,
            use_registers: bool = True,  # Match classifier
    ):
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], f"Invalid down_ratio: {down_ratio}"

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.use_registers = use_registers

        # Load DINOv2 backbone
        dinov2_model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        if freeze_backbone:
            for param in dinov2_model.parameters():
                param.requires_grad = False

        self.patch_size = dinov2_model.patch_embed.patch_size[0]
        self.embed_dim = dinov2_model.embed_dim
        self.num_prefix_tokens = getattr(dinov2_model, 'num_prefix_tokens', 1)
        self.num_register_tokens = max(0, self.num_prefix_tokens - 1)

        # Global feature dimension (matches classifier)
        self.global_feat_dim = self.embed_dim

        logger.info(f"HerdNetDINOv2 initialized:")
        logger.info(f"  Backbone: {backbone}")
        logger.info(f"  Patch size: {self.patch_size}")
        logger.info(f"  Embedding dim: {self.embed_dim}")
        logger.info(f"  Down ratio: {down_ratio}")
        logger.info(f"  Prefix tokens: {self.num_prefix_tokens} (1 CLS + {self.num_register_tokens} registers)")
        logger.info(f"  Use registers: {use_registers}")

        # Feature extractor (now always returns 3 values)
        if use_simple_extractor:
            self.attention_extractor = SimpleDINOv2Extractor(dinov2_model)
            logger.info(f"  Using simple feature extractor")
        else:
            try:
                self.attention_extractor = DINOv2AttentionExtractor(dinov2_model, attention_layers)
                logger.info(f"  Using hook-based attention extraction")
            except Exception as e:
                logger.warning(f"  Hook-based attention failed ({e}), using simple extractor")
                self.attention_extractor = SimpleDINOv2Extractor(dinov2_model)

        # =====================================================================
        # CLASSIFICATION HEADS (matching IguanaClassifierWithCount exactly)
        # These can be loaded from a pretrained classifier
        # =====================================================================

        # Register attention for combining CLS + register tokens (matches classifier)
        if use_registers and self.num_register_tokens > 0:
            self.register_attention = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 4, 1),
            )
        else:
            self.register_attention = None

        # Classification head - binary presence (matches classifier.cls_head)
        self.cls_head = nn.Sequential(
            nn.Linear(self.global_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Count regression head from CLS (matches classifier.count_head_cls)
        self.count_head_cls = nn.Sequential(
            nn.Linear(self.global_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # =====================================================================
        # LOCALIZATION HEADS (HerdNet-specific)
        # =====================================================================

        # Spatial processor
        self.spatial_processor = DINOv2SpatialProcessor(
            feature_dim=self.embed_dim,
            output_channels=output_channels
        )

        # Feature upsampler
        self.feature_upsampler = DINOv2FeatureUpsampler(
            in_channels=output_channels,
            out_channels=output_channels[0]
        )

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(
            output_channels[0], output_channels[0],
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Localization head (heatmap) - main output for point detection
        self.loc_head = nn.Sequential(
            nn.Conv2d(output_channels[0], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(0.0)

        # Spatial classification head (per-patch, for visualization)
        self.spatial_cls_head = nn.Sequential(
            nn.Conv2d(output_channels[-1], head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.spatial_cls_head[-1].bias.data.fill_(0.0)

        # Initialize classification heads (matches classifier init)
        for head in [self.cls_head, self.count_head_cls, self.register_attention]:
            if head is not None:
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        # Target heatmap size based on down_ratio
        self.target_heatmap_size = 518 // self.down_ratio

    def _get_global_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract global features from CLS and register tokens.
        Matches IguanaClassifierWithCount._get_global_features exactly.

        Args:
            features: [B, num_tokens, feat_dim] all tokens from backbone

        Returns:
            global_feat: [B, global_feat_dim] aggregated global features
        """
        cls_token = features[:, 0]  # [B, feat_dim]

        if self.use_registers and self.num_register_tokens > 0 and self.register_attention is not None:
            # Get register tokens
            register_tokens = features[:, 1:1 + self.num_register_tokens]  # [B, num_reg, feat_dim]

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
            return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, 3, H, W] input images
            return_features: If True, include intermediate features for visualization

        Returns:
            Dict containing:
                - heatmap: [B, 1, H/down_ratio, W/down_ratio] localization heatmap
                - cls_logit: [B] binary presence logit (for filtering)
                - count_cls: [B] count prediction from CLS token
                - clsmap: [B, num_classes, 16, 16] spatial classification (for vis)
        """
        B = x.shape[0]

        # Ensure input matches DINOv2 expected size
        target_size = self.patch_size * 37  # 518 for patch_size=14
        if x.shape[2:] != (target_size, target_size):
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # Extract features
        patch_features, attention_maps, all_tokens = self.attention_extractor(x)

        # Get global features for classification heads (matches classifier)
        global_feat = self._get_global_features(all_tokens)

        # =====================================================================
        # CLASSIFICATION OUTPUTS (can be used for filtering)
        # =====================================================================

        # Binary classification from global features
        cls_logit = self.cls_head(global_feat).squeeze(-1)  # [B]

        # Count from global features
        count_cls = F.softplus(self.count_head_cls(global_feat).squeeze(-1))  # [B]

        # =====================================================================
        # LOCALIZATION OUTPUTS (point detection)
        # =====================================================================

        # Process to multi-scale spatial features
        multi_scale_features, attention_heatmaps = self.spatial_processor(patch_features, attention_maps)

        # Fuse features
        fused_features = self.feature_upsampler(multi_scale_features, attention_heatmaps)

        # Localization heatmap (main output)
        bottleneck_features = self.bottleneck_conv(fused_features)
        heatmap = self.loc_head(bottleneck_features)
        heatmap = F.interpolate(heatmap, size=(self.target_heatmap_size, self.target_heatmap_size),
                                mode='bilinear', align_corners=False)

        # Spatial classification (for visualization)
        clsmap = self.spatial_cls_head(multi_scale_features[-1])
        clsmap = F.interpolate(clsmap, size=(16, 16), mode='bilinear', align_corners=False)

        output = {
            'heatmap': heatmap,
            'cls_logit': cls_logit,
            'count_cls': count_cls,
            'count_pred': count_cls,  # Alias for loss function compatibility
            'clsmap': clsmap,
        }

        if return_features:
            # Create feature magnitude map for visualization
            _, N, D = patch_features.shape
            H = W = int(N ** 0.5)

            feature_magnitude = torch.norm(patch_features, dim=2)
            feature_magnitude = feature_magnitude.view(B, 1, H, W)
            feat_min = feature_magnitude.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            feat_max = feature_magnitude.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            feature_magnitude_norm = (feature_magnitude - feat_min) / (feat_max - feat_min + 1e-8)
            feature_map_vis = F.interpolate(
                feature_magnitude_norm,
                size=(self.target_heatmap_size, self.target_heatmap_size),
                mode='bilinear',
                align_corners=False
            )

            # Attention map
            if attention_maps and len(attention_maps) > 0:
                first_attn_key = list(attention_maps.keys())[0]
                attn = attention_maps[first_attn_key]
                attn_map = attn.view(B, 1, H, W)
                attn_map = F.interpolate(
                    attn_map,
                    size=(self.target_heatmap_size, self.target_heatmap_size),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                attn_map = feature_map_vis

            output['feature_map'] = feature_map_vis
            output['attention_map'] = attn_map
            output['fused_features'] = fused_features

        return output

    def unfreeze_backbone(self, n_blocks: Optional[int] = None):
        """Unfreeze backbone parameters."""
        dinov2 = self.attention_extractor.dinov2

        if n_blocks is None:
            for p in dinov2.parameters():
                p.requires_grad = True
            logger.info("Unfroze entire backbone")
        else:
            if hasattr(dinov2, 'blocks'):
                total = len(dinov2.blocks)
                for i, block in enumerate(dinov2.blocks):
                    if i >= total - n_blocks:
                        for p in block.parameters():
                            p.requires_grad = True
                logger.info(f"Unfroze last {n_blocks} of {total} blocks")

    def __del__(self):
        if hasattr(self, 'attention_extractor'):
            self.attention_extractor.remove_hooks()


# =============================================================================
# DATASET
# =============================================================================

class HerdNetDataset(Dataset):
    """Dataset for HerdNet training with FIDT targets."""

    def __init__(
            self,
            csv_path: str,
            image_dir: str,
            crop_size: int = 518,
            overlap: int = 0,
            down_ratio: int = 4,
            target_type: str = 'fidt',
            fidt_alpha: float = 0.02,
            fidt_beta: float = 0.75,
            fidt_c: float = 1.0,
            gaussian_sigma: float = 2.0,
            augment: bool = False,
            positive_only: bool = False,  # Only include tiles with iguanas
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.overlap = overlap
        self.stride = crop_size - overlap
        self.down_ratio = down_ratio
        self.target_type = target_type
        self.augment = augment
        self.positive_only = positive_only

        if target_type == 'fidt':
            self.target_generator = FIDT(
                alpha=fidt_alpha,
                beta=fidt_beta,
                c=fidt_c,
                down_ratio=down_ratio,
            )
        else:
            self.target_generator = GaussianMap(
                sigma=gaussian_sigma,
                down_ratio=down_ratio,
            )

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        self.annotations = {}
        for name, group in self.df.groupby('images'):
            self.annotations[name] = group[['x', 'y']].values.astype(np.float32)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.photometric_transform = self._build_augmentation() if augment else None
        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])

        # Pre-compute tiles
        self.tiles = []
        self._image_sizes = {}
        n_total_tiles = 0
        n_positive_tiles = 0

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
                    n_total_tiles += 1

                    # Check if tile has iguanas
                    has_points = self._has_points_in_tile(name, crop_x, crop_y)

                    if positive_only:
                        if has_points:
                            self.tiles.append((name, crop_x, crop_y))
                            n_positive_tiles += 1
                    else:
                        self.tiles.append((name, crop_x, crop_y))
                        if has_points:
                            n_positive_tiles += 1

        total_points = sum(len(pts) for pts in self.annotations.values())
        logger.info(f"HerdNetDataset: {len(self.image_names)} images, {total_points} points")
        logger.info(f"  Crop size: {crop_size}, Stride: {self.stride}, Down ratio: {down_ratio}")
        logger.info(f"  Total tiles: {n_total_tiles}, Positive tiles: {n_positive_tiles}")
        if positive_only:
            logger.info(f"  positive_only=True: Using {len(self.tiles)} positive tiles only")
        else:
            logger.info(f"  positive_only=False: Using all {len(self.tiles)} tiles")
        logger.info(f"  Target type: {target_type}")

    def _build_augmentation(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
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

    def _get_points_in_tile(self, name: str, crop_x: int, crop_y: int) -> np.ndarray:
        """Get points within tile in tile coordinates."""
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

        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']

        crop = self.normalize_transform(image=crop)['image']
        target_heatmap = self.target_generator(points_in_crop, self.crop_size, self.crop_size)

        count = float(len(points_in_crop))
        label = 1.0 if len(points_in_crop) > 0 else 0.0

        return crop, {
            'heatmap': target_heatmap,
            'label': torch.tensor(label, dtype=torch.float32),
            'count': torch.tensor(count, dtype=torch.float32),
            'points': torch.from_numpy(points_in_crop).float(),
            'name': name,
            'crop_x': crop_x,
            'crop_y': crop_y,
        }


def herdnet_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {
        'heatmap': torch.stack([b[1]['heatmap'] for b in batch]),
        'label': torch.stack([b[1]['label'] for b in batch]),
        'count': torch.stack([b[1]['count'] for b in batch]),
        'points': [b[1]['points'] for b in batch],
        'name': [b[1]['name'] for b in batch],
        'crop_x': [b[1]['crop_x'] for b in batch],
        'crop_y': [b[1]['crop_y'] for b in batch],
    }
    return images, targets


# =============================================================================
# LOSSES
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for dense prediction."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_mask = (target >= 0.01).float()
        neg_mask = (target < 0.01).float()

        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred + 1e-8) * pos_mask
        neg_loss = -torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta) * torch.log(
            1 - pred + 1e-8) * neg_mask

        num_pos = pos_mask.sum()

        if num_pos == 0:
            return neg_loss.sum()

        return (pos_loss.sum() + neg_loss.sum()) / num_pos


class HerdNetLoss(nn.Module):
    """Combined loss for HerdNet with auxiliary heads."""

    def __init__(
            self,
            loc_weight: float = 1.0,
            cls_weight: float = 0.5,
            count_weight: float = 0.5,
            use_focal: bool = False,  # Changed default: MSE works better for density maps
            pos_weight: float = 3.0,
    ):
        super().__init__()
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight
        self.count_weight = count_weight
        self.pos_weight = pos_weight
        self.use_focal = use_focal

        if use_focal:
            self.loc_loss_fn = FocalLoss()
        else:
            # MSE loss - better for density map regression
            self.loc_loss_fn = nn.MSELoss()

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            target_heatmap: torch.Tensor,
            target_labels: torch.Tensor,
            target_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            outputs: Dict from model forward pass
            target_heatmap: [B, 1, H, W] target heatmap
            target_labels: [B] binary presence labels
            target_counts: [B] target counts
        """
        device = target_heatmap.device

        # 1. Localization loss (heatmap)
        loc_loss = self.loc_loss_fn(outputs['heatmap'], target_heatmap)

        # 2. Binary classification loss
        pos_weight_tensor = torch.tensor([self.pos_weight], device=device)
        cls_loss = F.binary_cross_entropy_with_logits(
            outputs['cls_logit'], target_labels,
            pos_weight=pos_weight_tensor.expand_as(target_labels)
        )

        # 3. Count regression loss
        count_loss = F.smooth_l1_loss(outputs['count_pred'], target_counts)

        # 4. Also supervise heatmap sum to approximate count
        heatmap_sum = outputs['heatmap'].sum(dim=(1, 2, 3))
        sum_loss = F.smooth_l1_loss(heatmap_sum, target_counts)

        # Total loss
        total_loss = (
                self.loc_weight * loc_loss +
                self.cls_weight * cls_loss +
                self.count_weight * (count_loss + 0.1 * sum_loss)
        )

        return {
            'total': total_loss,
            'loc': loc_loss,
            'cls': cls_loss,
            'count': count_loss,
            'sum': sum_loss,
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_loc_loss = 0
    total_cls_loss = 0
    total_count_loss = 0
    total_samples = 0
    total_count_mae = 0
    total_correct = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        target_heatmap = targets['heatmap'].to(device)
        target_labels = targets['label'].to(device)
        target_counts = targets['count'].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        losses = criterion(outputs, target_heatmap, target_labels, target_counts)

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = len(images)
        total_loss += losses['total'].item() * batch_size
        total_loc_loss += losses['loc'].item() * batch_size
        total_cls_loss += losses['cls'].item() * batch_size
        total_count_loss += losses['count'].item() * batch_size
        total_samples += batch_size

        # Count MAE from count_pred head
        total_count_mae += torch.abs(outputs['count_pred'] - target_counts).sum().item()

        # Classification accuracy
        preds = (torch.sigmoid(outputs['cls_logit']) > 0.5).float()
        total_correct += (preds == target_labels).sum().item()

    n = total_samples
    return {
        'loss': total_loss / n,
        'loc_loss': total_loc_loss / n,
        'cls_loss': total_cls_loss / n,
        'count_loss': total_count_loss / n,
        'count_mae': total_count_mae / n,
        'cls_acc': total_correct / n,
    }


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        lmds: LMDS,
        threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_loss = 0
    total_samples = 0

    all_heatmap_sums = []
    all_heatmap_maxs = []  # Track max values
    all_count_preds = []
    all_gt_counts = []
    all_detected_counts = []
    all_cls_probs = []
    all_cls_labels = []

    for images, targets in loader:
        images = images.to(device)
        target_heatmap = targets['heatmap'].to(device)
        target_labels = targets['label'].to(device)
        target_counts = targets['count'].to(device)

        outputs = model(images)
        losses = criterion(outputs, target_heatmap, target_labels, target_counts)

        batch_size = len(images)
        total_loss += losses['total'].item() * batch_size
        total_samples += batch_size

        # Collect predictions
        heatmaps = outputs['heatmap']
        all_heatmap_sums.extend(heatmaps.sum(dim=(1, 2, 3)).cpu().numpy())
        all_heatmap_maxs.extend(heatmaps.amax(dim=(1, 2, 3)).cpu().numpy())  # Max per sample
        all_count_preds.extend(outputs['count_pred'].cpu().numpy())
        all_gt_counts.extend(target_counts.cpu().numpy())

        # Classification
        cls_probs = torch.sigmoid(outputs['cls_logit']).cpu().numpy()
        all_cls_probs.extend(cls_probs)
        all_cls_labels.extend(target_labels.cpu().numpy())

        # LMDS detection
        counts, _, _ = lmds(heatmaps, scale_factor=model.down_ratio)
        all_detected_counts.extend(counts)

    all_heatmap_sums = np.array(all_heatmap_sums)
    all_heatmap_maxs = np.array(all_heatmap_maxs)
    all_count_preds = np.array(all_count_preds)
    all_gt_counts = np.array(all_gt_counts)
    all_detected_counts = np.array(all_detected_counts)
    all_cls_probs = np.array(all_cls_probs)
    all_cls_labels = np.array(all_cls_labels)

    # Count metrics
    count_mae_sum = np.abs(all_heatmap_sums - all_gt_counts).mean()
    count_mae_pred = np.abs(all_count_preds - all_gt_counts).mean()
    count_mae_lmds = np.abs(all_detected_counts - all_gt_counts).mean()

    # Count accuracy
    tolerance = 1
    count_acc = np.mean(np.abs(all_detected_counts - all_gt_counts) <= tolerance)

    # Classification metrics
    cls_preds = (all_cls_probs > threshold).astype(float)
    cls_acc = (cls_preds == all_cls_labels).mean()

    tp = ((cls_preds == 1) & (all_cls_labels == 1)).sum()
    fp = ((cls_preds == 1) & (all_cls_labels == 0)).sum()
    fn = ((cls_preds == 0) & (all_cls_labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    f3 = 10 * precision * recall / max(9 * precision + recall, 1e-6)

    # Heatmap statistics for debugging
    hm_max_mean = all_heatmap_maxs.mean()
    hm_max_std = all_heatmap_maxs.std()
    hm_sum_mean = all_heatmap_sums.mean()

    # Filtered LMDS counts: set count to 0 if classifier says "no iguana"
    # This shows the benefit of using classifier for filtering
    filtered_counts = np.where(all_cls_probs > threshold, all_detected_counts, 0)
    count_mae_filtered = np.abs(filtered_counts - all_gt_counts).mean()

    return {
        'loss': total_loss / total_samples,
        'count_mae_sum': count_mae_sum,
        'count_mae_pred': count_mae_pred,
        'count_mae_lmds': count_mae_lmds,
        'count_mae_filtered': count_mae_filtered,  # LMDS counts filtered by classifier
        'count_acc': count_acc,
        'cls_acc': cls_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f3': f3,
        'hm_max_mean': hm_max_mean,
        'hm_max_std': hm_max_std,
        'hm_sum_mean': hm_sum_mean,
        'lmds_total': all_detected_counts.sum(),
        'filtered_total': filtered_counts.sum(),  # Total detections after filtering
    }


def visualize_predictions(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        lmds: LMDS,
        output_dir: Path,
        epoch: int,
        max_samples: int = 16,
):
    """Visualize model predictions including feature maps."""
    if not HAS_MATPLOTLIB:
        return

    model.eval()
    all_results = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            outputs = model(images, return_features=True)
            pred_heatmap = outputs['heatmap']
            feature_map = outputs['feature_map']
            attention_map = outputs['attention_map']

            counts, locs, scores = lmds(pred_heatmap, scale_factor=model.down_ratio)

            for i in range(len(images)):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

                all_results.append({
                    'image': img,
                    'heatmap': pred_heatmap[i, 0].cpu().numpy(),
                    'feature_map': feature_map[i, 0].cpu().numpy(),
                    'attention_map': attention_map[i, 0].cpu().numpy(),
                    'gt_heatmap': targets['heatmap'][i, 0].numpy(),
                    'gt_points': targets['points'][i].numpy(),
                    'pred_locs': locs[i],
                    'pred_scores': scores[i],
                    'pred_count': counts[i],
                    'gt_count': int(targets['count'][i]),
                    'cls_prob': torch.sigmoid(outputs['cls_logit'][i]).item(),
                    'count_pred': outputs['count_pred'][i].item(),
                })

            if len(all_results) >= max_samples:
                break

    epoch_dir = output_dir / f'epoch_{epoch:03d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    n_show = min(max_samples, len(all_results))
    fig, axes = plt.subplots(n_show, 6, figsize=(24, 4 * n_show))

    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(all_results[:n_show]):
        # Column 0: Original image with GT points
        axes[i, 0].imshow(r['image'])
        for pt in r['gt_points']:
            axes[i, 0].plot(pt[0], pt[1], 'g+', markersize=10, markeredgewidth=2)
        axes[i, 0].set_title(f"GT: {r['gt_count']} points")
        axes[i, 0].axis('off')

        # Column 1: GT heatmap
        axes[i, 1].imshow(r['gt_heatmap'], cmap='hot', vmin=0, vmax=1)
        axes[i, 1].set_title("GT Heatmap")
        axes[i, 1].axis('off')

        # Column 2: Feature magnitude map
        feat_map = r['feature_map']
        axes[i, 2].imshow(feat_map, cmap='viridis')
        axes[i, 2].set_title(f"Feature Map (max={feat_map.max():.2f})")
        axes[i, 2].axis('off')

        # Column 3: Attention map
        attn_map = r['attention_map']
        axes[i, 3].imshow(attn_map, cmap='plasma')
        axes[i, 3].set_title(f"Attention (max={attn_map.max():.2f})")
        axes[i, 3].axis('off')

        # Column 4: Predicted heatmap
        hm = r['heatmap']
        axes[i, 4].imshow(hm, cmap='hot', vmin=0, vmax=max(0.1, hm.max()))
        axes[i, 4].set_title(f"Pred Heatmap (max={hm.max():.3f}, sum={hm.sum():.1f})")
        axes[i, 4].axis('off')

        # Column 5: Predictions overlay
        axes[i, 5].imshow(r['image'])
        for (y, x), score in zip(r['pred_locs'], r['pred_scores']):
            axes[i, 5].plot(x, y, 'r+', markersize=12, markeredgewidth=2)
        for pt in r['gt_points']:
            circle = plt.Circle((pt[0], pt[1]), 15, color='lime', fill=False, linewidth=2)
            axes[i, 5].add_patch(circle)
        axes[i, 5].set_title(
            f"LMDS:{r['pred_count']} Head:{r['count_pred']:.1f} (GT:{r['gt_count']}) P={r['cls_prob']:.2f}")
        axes[i, 5].axis('off')

    plt.tight_layout()
    plt.savefig(epoch_dir / 'predictions.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Save positive tiles visualization
    positive_results = [r for r in all_results if r['gt_count'] > 0]
    if len(positive_results) > 0:
        n_pos = min(8, len(positive_results))
        fig, axes = plt.subplots(n_pos, 5, figsize=(20, 4 * n_pos))

        if n_pos == 1:
            axes = axes.reshape(1, -1)

        for i, r in enumerate(positive_results[:n_pos]):
            axes[i, 0].imshow(r['image'])
            for pt in r['gt_points']:
                circle = plt.Circle((pt[0], pt[1]), 12, color='lime', fill=False, linewidth=2)
                axes[i, 0].add_patch(circle)
            axes[i, 0].set_title(f"Image (GT: {r['gt_count']})")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(r['image'])
            axes[i, 1].imshow(r['feature_map'], cmap='hot', alpha=0.6)
            for pt in r['gt_points']:
                axes[i, 1].plot(pt[0], pt[1], 'c+', markersize=12, markeredgewidth=2)
            axes[i, 1].set_title("Feature Map Overlay")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(r['image'])
            axes[i, 2].imshow(r['attention_map'], cmap='hot', alpha=0.6)
            for pt in r['gt_points']:
                axes[i, 2].plot(pt[0], pt[1], 'c+', markersize=12, markeredgewidth=2)
            axes[i, 2].set_title("Attention Overlay")
            axes[i, 2].axis('off')

            axes[i, 3].imshow(r['image'])
            axes[i, 3].imshow(r['heatmap'], cmap='hot', alpha=0.6)
            for pt in r['gt_points']:
                axes[i, 3].plot(pt[0], pt[1], 'c+', markersize=12, markeredgewidth=2)
            axes[i, 3].set_title(f"Heatmap Overlay (max={r['heatmap'].max():.3f})")
            axes[i, 3].axis('off')

            axes[i, 4].imshow(r['image'])
            for (y, x), score in zip(r['pred_locs'], r['pred_scores']):
                axes[i, 4].plot(x, y, 'r+', markersize=12, markeredgewidth=2)
            for pt in r['gt_points']:
                circle = plt.Circle((pt[0], pt[1]), 12, color='lime', fill=False, linewidth=2)
                axes[i, 4].add_patch(circle)
            axes[i, 4].set_title(f"Pred: {r['pred_count']}")
            axes[i, 4].axis('off')

        plt.tight_layout()
        plt.savefig(epoch_dir / 'positive_tiles.png', dpi=120, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved visualization to {epoch_dir / 'predictions.png'}")


# =============================================================================
# CHECKPOINT LOADING UTILITIES
# =============================================================================

def load_backbone_from_classifier(
        model: nn.Module,
        classifier_checkpoint: str,
        device: torch.device = None,
        load_classifier_heads: bool = True,
) -> Dict[str, Any]:
    """
    Load DINOv2 backbone weights from a trained IguanaClassifierWithCount checkpoint.

    Optionally also loads the classification heads (cls_head, count_head_cls, register_attention)
    which allows using the pretrained classifier for filtering before LMDS detection.

    Args:
        model: HerdNetDINOv2 model
        classifier_checkpoint: Path to classifier checkpoint
        device: Device to load weights to
        load_classifier_heads: If True, also load cls_head, count_head_cls, register_attention

    Returns:
        Info dict with source, epoch, metrics
    """
    if device is None:
        device = next(model.parameters()).device

    logger.info(f"Loading from classifier checkpoint: {classifier_checkpoint}")

    ckpt = torch.load(classifier_checkpoint, map_location=device, weights_only=False)

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Map classifier backbone keys to HerdNet keys
    backbone_state_dict = {}
    classifier_head_dict = {}

    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_key = 'attention_extractor.dinov2.' + key[len('backbone.'):]
            backbone_state_dict[new_key] = value
        elif load_classifier_heads:
            # Map classifier head keys directly (same names in HerdNet)
            if key.startswith('cls_head.') or key.startswith('count_head_cls.') or key.startswith(
                    'register_attention.'):
                classifier_head_dict[key] = value

    if len(backbone_state_dict) == 0:
        raise ValueError(f"No backbone weights found in checkpoint")

    logger.info(f"  Found {len(backbone_state_dict)} backbone parameters")

    # Load backbone
    model.load_state_dict(backbone_state_dict, strict=False)

    # Load classifier heads if requested
    if load_classifier_heads and len(classifier_head_dict) > 0:
        logger.info(f"  Found {len(classifier_head_dict)} classifier head parameters")

        # Check compatibility
        missing, unexpected = [], []
        model_state = model.state_dict()

        for key in classifier_head_dict:
            if key in model_state:
                if model_state[key].shape == classifier_head_dict[key].shape:
                    pass  # Compatible
                else:
                    logger.warning(
                        f"  Shape mismatch for {key}: model={model_state[key].shape}, ckpt={classifier_head_dict[key].shape}")
                    del classifier_head_dict[key]
            else:
                unexpected.append(key)

        if unexpected:
            logger.warning(f"  Unexpected keys (not loaded): {unexpected}")
            for key in unexpected:
                del classifier_head_dict[key]

        model.load_state_dict(classifier_head_dict, strict=False)
        logger.info(f"  Loaded {len(classifier_head_dict)} classifier head parameters")

    info = {
        'source': classifier_checkpoint,
        'epoch': ckpt.get('epoch', 'unknown'),
        'best_f3': ckpt.get('best_f3', 'unknown'),
        'best_count_mae': ckpt.get('best_count_mae', 'unknown'),
        'optimal_threshold_f3': ckpt.get('optimal_threshold_f3', 0.3),
        'backbone_loaded': len(backbone_state_dict),
        'heads_loaded': len(classifier_head_dict) if load_classifier_heads else 0,
    }

    logger.info(f"  Loaded from epoch {info['epoch']}, F3: {info['best_f3']}, MAE: {info['best_count_mae']}")
    logger.info(f"  Optimal classifier threshold: {info['optimal_threshold_f3']}")

    return info


def freeze_classifier_heads(model: nn.Module):
    """Freeze the classification heads (for training localization only)."""
    frozen = 0
    for name, param in model.named_parameters():
        if 'cls_head' in name or 'count_head_cls' in name or 'register_attention' in name:
            param.requires_grad = False
            frozen += 1
    logger.info(f"Froze {frozen} classifier head parameters")
    return frozen


def unfreeze_classifier_heads(model: nn.Module):
    """Unfreeze the classification heads."""
    unfrozen = 0
    for name, param in model.named_parameters():
        if 'cls_head' in name or 'count_head_cls' in name or 'register_attention' in name:
            param.requires_grad = True
            unfrozen += 1
    logger.info(f"Unfroze {unfrozen} classifier head parameters")
    return unfrozen


@torch.no_grad()
def detect_with_filtering(
        model: nn.Module,
        images: torch.Tensor,
        lmds: LMDS,
        cls_threshold: float = 0.3,
        return_all: bool = False,
) -> Dict[str, Any]:
    """
    Run detection with classifier-based filtering.

    Uses the classification head to filter tiles before running LMDS,
    which reduces false positives.

    Args:
        model: HerdNetDINOv2 model
        images: [B, 3, H, W] batch of images
        lmds: LMDS detector
        cls_threshold: Classification threshold (tiles below this are marked empty)
        return_all: If True, return all detections even from filtered tiles

    Returns:
        Dict with:
            - 'counts': [B] detected counts (0 for filtered tiles)
            - 'locations': List[np.ndarray] of (x, y) points per image
            - 'scores': List[np.ndarray] of confidence scores per detection
            - 'cls_probs': [B] classification probabilities
            - 'count_pred': [B] count predictions from count head
            - 'filtered': [B] boolean mask of which tiles were filtered out
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)

    # Forward pass
    outputs = model(images)

    # Get classification probabilities
    cls_probs = torch.sigmoid(outputs['cls_logit']).cpu().numpy()
    count_pred = outputs['count_pred'].cpu().numpy()

    # Run LMDS detection
    counts, locations, scores = lmds(outputs['heatmap'], scale_factor=model.down_ratio)

    # Apply filtering: if classifier says "no iguana", zero out detections
    filtered_mask = cls_probs < cls_threshold

    if not return_all:
        # Zero out counts and clear locations for filtered tiles
        filtered_counts = []
        filtered_locations = []
        filtered_scores = []

        for i in range(len(counts)):
            if filtered_mask[i]:
                filtered_counts.append(0)
                filtered_locations.append(np.zeros((0, 2)))
                filtered_scores.append(np.zeros((0,)))
            else:
                filtered_counts.append(counts[i])
                filtered_locations.append(locations[i])
                filtered_scores.append(scores[i])

        counts = filtered_counts
        locations = filtered_locations
        scores = filtered_scores

    return {
        'counts': counts,
        'locations': locations,
        'scores': scores,
        'cls_probs': cls_probs,
        'count_pred': count_pred,
        'filtered': filtered_mask,
        'heatmap': outputs['heatmap'].cpu(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HerdNet Iguana Detector v2")

    # Data
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')

    # Model
    parser.add_argument('--backbone', default='vit_large_patch14_dinov2.lvd142m')
    parser.add_argument('--down_ratio', type=int, default=4)
    parser.add_argument('--head_conv', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dim for classification heads (must match classifier)')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--use_registers', action='store_true', default=True,
                        help='Use register tokens for global features (for reg4 backbones)')
    parser.add_argument('--no_registers', dest='use_registers', action='store_false')

    # Dataset
    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--tile_overlap', type=int, default=120)
    parser.add_argument('--target_type', default='fidt', choices=['fidt', 'gaussian'])
    parser.add_argument('--fidt_alpha', type=float, default=0.02)
    parser.add_argument('--fidt_beta', type=float, default=0.75)
    parser.add_argument('--gaussian_sigma', type=float, default=2.0)
    parser.add_argument('--positive_only', action='store_true', default=True,
                        help='Train only on tiles containing iguanas (default: True)')
    parser.add_argument('--all_tiles', dest='positive_only', action='store_false',
                        help='Train on all tiles including empty ones')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--unfreeze_epoch', type=int, default=10)
    parser.add_argument('--unfreeze_blocks', type=int, default=4)

    # Loss weights
    parser.add_argument('--loc_weight', type=float, default=1.0)
    parser.add_argument('--cls_weight', type=float, default=0.5)
    parser.add_argument('--count_weight', type=float, default=0.5)
    parser.add_argument('--pos_weight', type=float, default=3.0)
    parser.add_argument('--use_focal', action='store_true',
                        help='Use FocalLoss for heatmap (default: MSE)')

    # LMDS - lower thresholds to detect more points
    parser.add_argument('--lmds_adapt_ts', type=float, default=0.2)
    parser.add_argument('--lmds_neg_ts', type=float, default=0.05)
    parser.add_argument('--lmds_score_ts', type=float, default=0.1)
    parser.add_argument('--lmds_kernel_size', type=int, default=3)

    # Output
    parser.add_argument('--output_dir', default='./outputs_herdnet')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visualize_every', type=int, default=5)

    # Resume/Load
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--load_classifier', type=str, default=None,
                        help='Path to classifier checkpoint to load backbone and optionally heads')
    parser.add_argument('--load_classifier_heads', action='store_true', default=True,
                        help='Also load classification heads from classifier (default: True)')
    parser.add_argument('--no_load_classifier_heads', dest='load_classifier_heads', action='store_false',
                        help='Only load backbone, not classification heads')
    parser.add_argument('--freeze_loaded_backbone', action='store_true', default=True)
    parser.add_argument('--no_freeze_loaded_backbone', dest='freeze_loaded_backbone', action='store_false')
    parser.add_argument('--freeze_classifier_heads', action='store_true', default=False,
                        help='Freeze classification heads during training (use pretrained for filtering)')
    parser.add_argument('--cls_threshold', type=float, default=0.3,
                        help='Classification threshold for filtering (used in inference)')
    parser.add_argument('--early_stopping', type=int, default=15)

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_ds = HerdNetDataset(
        args.train_csv, args.train_image_dir,
        crop_size=args.crop_size,
        overlap=args.tile_overlap,
        down_ratio=args.down_ratio,
        target_type=args.target_type,
        fidt_alpha=args.fidt_alpha,
        fidt_beta=args.fidt_beta,
        gaussian_sigma=args.gaussian_sigma,
        augment=True,
        positive_only=args.positive_only,  # Train only on tiles with iguanas
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=herdnet_collate_fn,
    )

    val_loader = None
    if args.val_csv and args.val_image_dir:
        val_ds = HerdNetDataset(
            args.val_csv, args.val_image_dir,
            crop_size=args.crop_size,
            overlap=args.tile_overlap,
            down_ratio=args.down_ratio,
            target_type=args.target_type,
            fidt_alpha=args.fidt_alpha,
            fidt_beta=args.fidt_beta,
            gaussian_sigma=args.gaussian_sigma,
            augment=False,
            positive_only=True,  # Validate on ALL tiles to measure false positives
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=herdnet_collate_fn,
        )

    # Model
    model = HerdNetDINOv2(
        backbone=args.backbone,
        num_classes=2,
        pretrained=True,
        down_ratio=args.down_ratio,
        head_conv=args.head_conv,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        use_registers=args.use_registers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}, Trainable: {n_train:,}")

    # Adjust loss weights for positive_only training
    cls_weight = args.cls_weight
    if args.positive_only:
        logger.info("Training positive_only: disabling classification loss (all labels are 1)")
        cls_weight = 0.0  # No point in binary classification when all samples are positive

    # Loss and optimizer
    criterion = HerdNetLoss(
        loc_weight=args.loc_weight,
        cls_weight=cls_weight,
        count_weight=args.count_weight,
        pos_weight=args.pos_weight,
        use_focal=args.use_focal,
    )

    logger.info(f"Loss: {'FocalLoss' if args.use_focal else 'MSELoss'} for heatmap")
    logger.info(f"Loss weights: loc={args.loc_weight}, cls={cls_weight}, count={args.count_weight}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # LMDS for evaluation
    lmds = LMDS(
        kernel_size=(args.lmds_kernel_size, args.lmds_kernel_size),
        adapt_ts=args.lmds_adapt_ts,
        neg_ts=args.lmds_neg_ts,
        score_threshold=args.lmds_score_ts,
    )

    # Store classifier threshold for inference
    cls_threshold = args.cls_threshold

    # Load from classifier checkpoint
    if args.load_classifier:
        classifier_path = Path(args.load_classifier)
        if classifier_path.exists():
            classifier_info = load_backbone_from_classifier(
                model, str(classifier_path), device,
                load_classifier_heads=args.load_classifier_heads
            )

            # Use optimal threshold from classifier if available
            if 'optimal_threshold_f3' in classifier_info:
                cls_threshold = classifier_info['optimal_threshold_f3']
                logger.info(f"Using classifier threshold: {cls_threshold:.3f}")

            if args.freeze_loaded_backbone:
                logger.info("Freezing loaded backbone")
                for param in model.attention_extractor.dinov2.parameters():
                    param.requires_grad = False

            if args.freeze_classifier_heads and args.load_classifier_heads:
                freeze_classifier_heads(model)
                logger.info("Classifier heads frozen (will use pretrained for filtering)")

            # Recreate optimizer with only trainable params
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6
            )

            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters after freezing: {n_train:,}")

    # Resume
    start_epoch = 0
    best_mae = float('inf')
    backbone_unfrozen = False
    epochs_without_improvement = 0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_mae = ckpt.get('best_mae', float('inf'))
            backbone_unfrozen = ckpt.get('backbone_unfrozen', False)

            if backbone_unfrozen:
                model.unfreeze_backbone(args.unfreeze_blocks)

    # Training loop
    logger.info("=" * 80)
    logger.info("TRAINING HerdNet v2")
    logger.info("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze backbone
        if epoch == args.unfreeze_epoch and args.unfreeze_blocks > 0 and not backbone_unfrozen:
            logger.info(f"Unfreezing last {args.unfreeze_blocks} backbone blocks")
            model.unfreeze_backbone(args.unfreeze_blocks)
            backbone_unfrozen = True

            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.named_parameters()
                            if 'attention_extractor.dinov2' in n and p.requires_grad], 'lr': args.lr * 0.01},
                {'params': [p for n, p in model.named_parameters()
                            if 'attention_extractor.dinov2' not in n and p.requires_grad], 'lr': args.lr * 0.1},
            ], weight_decay=args.weight_decay)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-7
            )

        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()

        log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | "
        log += f"loss={train_m['loss']:.4f} loc={train_m['loc_loss']:.4f} "
        if not args.positive_only:
            log += f"cls={train_m['cls_loss']:.4f} "
        log += f"MAE={train_m['count_mae']:.2f}"

        improved = False
        if val_loader:
            val_m = evaluate(model, val_loader, criterion, device, lmds, threshold=cls_threshold)
            log += f" | val: MAE_lmds={val_m['count_mae_lmds']:.2f} MAE_filt={val_m['count_mae_filtered']:.2f} hm_max={val_m['hm_max_mean']:.3f} F3={val_m['f3']:.3f}"

            if val_m['count_mae_lmds'] < best_mae:
                best_mae = val_m['count_mae_lmds']
                epochs_without_improvement = 0
                improved = True
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mae': best_mae,
                    'best_mae_filtered': val_m['count_mae_filtered'],
                    'best_f3': val_m['f3'],
                    'cls_threshold': cls_threshold,
                    'backbone': args.backbone,
                    'hidden_dim': args.hidden_dim,
                    'down_ratio': args.down_ratio,
                    'use_registers': args.use_registers,
                    'backbone_unfrozen': backbone_unfrozen,
                }, output_dir / 'best.pth')
                log += " ★"
            else:
                epochs_without_improvement += 1
                log += f" ({epochs_without_improvement}/{args.early_stopping})"

        logger.info(log)

        # Visualize
        if val_loader and args.visualize_every > 0 and (epoch % args.visualize_every == 0 or improved):
            vis_dir = output_dir / 'visualizations'
            visualize_predictions(model, val_loader, device, lmds, vis_dir, epoch)

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mae': best_mae,
            'cls_threshold': cls_threshold,
            'backbone': args.backbone,
            'hidden_dim': args.hidden_dim,
            'down_ratio': args.down_ratio,
            'use_registers': args.use_registers,
            'backbone_unfrozen': backbone_unfrozen,
            'epochs_without_improvement': epochs_without_improvement,
        }, output_dir / 'latest.pth')

        # Early stopping
        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            logger.info(f"Early stopping after {epochs_without_improvement} epochs")
            break

    logger.info("=" * 80)
    logger.info(f"Training complete! Best MAE: {best_mae:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()