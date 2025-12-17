"""
HerdNetP2P v4 - Direct Feature Sampling

Key insight: Transformer decoder causes query collapse with ViT backbones.
Solution: Sample features directly at reference point locations.

For each query:
1. Sample ViT/CNN features at the reference point via bilinear interpolation
2. Each query is guaranteed to see different features based on its position
3. Simple MLP heads predict class + offset

This approach:
- Avoids mode collapse completely
- Works with DINOv2, DINOv3, and CNN backbones
- Simpler and faster than transformer decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, List
import numpy as np


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal 2D positional encoding."""

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding for feature map."""
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)

        y_embed = y_embed / H
        x_embed = x_embed / W

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # [C, H, W]
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)

        return pos


class HerdNetP2P(nn.Module):
    """
    P2PNet with direct feature sampling - avoids transformer decoder collapse.

    For each reference point:
    1. Sample features from backbone feature map via bilinear interpolation
    2. Concatenate with learnable position embedding
    3. MLP predicts classification + coordinate offset

    This guarantees spatial diversity - each query MUST see different features.
    """

    def __init__(
        self,
        backbone: str = 'vit_large_patch16_dinov3.sat493m',
        num_classes: int = 2,
        num_queries: int = 100,
        hidden_dim: int = 256,
        max_offset: float = 0.5,
        nhead: int = 8,  # Kept for API compatibility, not used
        num_decoder_layers: int = 6,  # Kept for API compatibility, not used
        dim_feedforward: int = 1024,  # Kept for API compatibility, not used
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.max_offset = max_offset

        # Build backbone
        self._build_backbone(backbone, pretrained, freeze_backbone)

        # Feature projection
        self.input_proj = nn.Linear(self.backbone_channels, hidden_dim)

        # Reference points (fixed grid)
        self.register_buffer('reference_points', self._create_grid(num_queries))

        # Learnable position embeddings for each query
        self.pos_embed = nn.Embedding(num_queries, hidden_dim)

        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # features + position
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

        # Coordinate head - predicts offset from reference point
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        self._init_weights()
        self.criterion = criterion

    def _build_backbone(self, backbone: str, pretrained: bool, freeze: bool):
        """Build and configure backbone network."""
        if 'vit' in backbone.lower() or 'dino' in backbone.lower():
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.backbone_channels = self.backbone.embed_dim
            self.backbone_type = 'vit'

            # Get patch size
            ps = getattr(self.backbone.patch_embed, 'patch_size', (16, 16))
            self.patch_size = ps[0] if isinstance(ps, tuple) else ps

            # Check for prefix tokens (CLS, registers)
            if hasattr(self.backbone, 'num_prefix_tokens'):
                self.num_prefix_tokens = self.backbone.num_prefix_tokens
            else:
                self.num_prefix_tokens = 1  # Assume CLS token

            print(f"ViT backbone: {backbone}")
            print(f"  Embed dim: {self.backbone_channels}")
            print(f"  Patch size: {self.patch_size}")
            print(f"  Prefix tokens: {self.num_prefix_tokens}")
        else:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=(4,)
            )
            self.backbone_channels = self.backbone.feature_info.channels()[-1]
            self.backbone_type = 'cnn'
            self.patch_size = 16
            self.num_prefix_tokens = 0

            print(f"CNN backbone: {backbone}")
            print(f"  Feature channels: {self.backbone_channels}")

        if freeze:
            print("Freezing backbone!")
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _create_grid(self, n: int) -> torch.Tensor:
        """Create fixed grid of reference points in [0.05, 0.95]."""
        grid_size = int(np.ceil(np.sqrt(n)))
        coords = torch.linspace(0.05, 0.95, grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [N, 2] as (x, y)
        return grid[:n]

    def _init_weights(self):
        """Initialize weights."""
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Position embeddings - diverse initialization
        nn.init.normal_(self.pos_embed.weight, std=1.0)

        # Classification head - slight bias toward background
        for layer in self.class_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Bias final layer
        nn.init.constant_(self.class_head[-1].bias[0], 1.0)  # Background
        if self.num_classes > 1:
            nn.init.constant_(self.class_head[-1].bias[1:], -1.0)  # Foreground

        # Coordinate head - initialize to predict near-zero offsets
        for layer in self.coord_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.coord_head[-1].weight)
        nn.init.zeros_(self.coord_head[-1].bias)

    def _sample_features(
        self,
        feature_map: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample features at given points using bilinear interpolation.

        Args:
            feature_map: [B, H, W, C] - spatial feature map
            points: [N, 2] - normalized coordinates (x, y) in [0, 1]

        Returns:
            sampled: [B, N, C] - features at each point
        """
        B, H, W, C = feature_map.shape
        N = points.shape[0]

        # Convert [0, 1] to [-1, 1] for grid_sample (x, y order)
        grid = points * 2 - 1  # [N, 2]
        grid = grid.view(1, 1, N, 2).expand(B, 1, N, 2)  # [B, 1, N, 2]

        # Reshape feature map for grid_sample: [B, C, H, W]
        feature_map = feature_map.permute(0, 3, 1, 2)

        # Sample features
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [B, C, 1, N]

        sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, N, C]
        return sampled

    def set_criterion(self, criterion: nn.Module):
        """Set the loss criterion."""
        self.criterion = criterion

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Forward pass.

        Args:
            x: [B, 3, H, W] input images
            targets: Optional list of target dicts for training

        Returns:
            Dict with predictions and optionally loss
        """
        B, _, img_h, img_w = x.shape
        device = x.device

        # Extract features
        if self.backbone_type == 'vit':
            feat = self.backbone.forward_features(x)  # [B, num_tokens, embed_dim]

            grid_h = img_h // self.patch_size
            grid_w = img_w // self.patch_size

            if feat.dim() == 3:
                # Remove prefix tokens (CLS, registers)
                if self.num_prefix_tokens > 0:
                    feat = feat[:, self.num_prefix_tokens:, :]

                # Reshape to spatial grid
                feat = feat.view(B, grid_h, grid_w, self.backbone_channels)  # [B, H, W, C]
        else:
            feat = self.backbone(x)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            grid_h, grid_w = feat.shape[2], feat.shape[3]
            feat = feat.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Sample features at reference points
        sampled_features = self._sample_features(feat, self.reference_points)  # [B, N, C]

        # Project features
        sampled_features = self.input_proj(sampled_features)  # [B, N, hidden_dim]

        # Get position embeddings
        pos_indices = torch.arange(self.num_queries, device=device)
        pos_embed = self.pos_embed(pos_indices)  # [N, hidden_dim]
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N, hidden_dim]

        # Concatenate features and position
        combined = torch.cat([sampled_features, pos_embed], dim=-1)  # [B, N, hidden_dim*2]

        # Predict classification
        pred_logits = self.class_head(combined)  # [B, N, num_classes]

        # Predict coordinate offsets
        offsets = self.coord_head(combined)  # [B, N, 2]

        # Apply bounded offsets to reference points
        ref_pts = self.reference_points.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

        if self.max_offset is not None:
            offsets = torch.tanh(offsets) * self.max_offset
            pred_points_norm = (ref_pts + offsets).clamp(0, 1)  # [B, N, 2] in [0, 1]
        else:
            pred_points_norm = torch.sigmoid(offsets)  # Direct prediction

        # Scale to pixel coordinates
        pred_points = pred_points_norm.clone()
        pred_points[..., 0] *= img_w  # x
        pred_points[..., 1] *= img_h  # y

        outputs = {
            'pred_logits': pred_logits,
            'pred_points': pred_points,
            'pred_points_normalized': pred_points_norm,
            'reference_points': ref_pts,
            'offsets': offsets,
            'image_size': (img_h, img_w),
            # Aliases for compatibility
            'logits': pred_logits,
            'points': pred_points,
            'points_normalized': pred_points_norm,
        }

        if targets is not None and self.criterion is not None:
            outputs['loss_p2p'] = self.criterion(outputs, targets)

        return outputs

    def get_predictions(
        self,
        outputs: Dict,
        confidence_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Get predictions as list of dicts with points and scores.

        Args:
            outputs: Model output dict
            confidence_threshold: Minimum confidence to keep prediction

        Returns:
            List of dicts with 'points', 'scores', 'labels' for each image
        """
        logits = outputs['logits']
        points = outputs['points']

        B = logits.shape[0]
        results = []

        for b in range(B):
            probs = logits[b].softmax(-1)  # [N, num_classes]

            if self.num_classes == 2:
                # Binary: foreground is class 1
                scores = probs[:, 1]
                labels = torch.ones(len(scores), dtype=torch.long, device=scores.device)
            else:
                # Multi-class: take max (excluding background)
                scores, labels = probs[:, 1:].max(dim=-1)
                labels = labels + 1  # Shift since we excluded class 0

            # Filter by threshold
            mask = scores >= confidence_threshold

            results.append({
                'points': points[b, mask],
                'scores': scores[mask],
                'labels': labels[mask],
            })

        return results


class P2PNetLite(nn.Module):
    """
    Dense grid P2PNet - naturally avoids mode collapse.

    Uses convolutional heads on the feature map directly,
    treating each spatial location as a potential detection.
    """

    def __init__(
        self,
        backbone: str = 'vit_large_patch16_dinov3.sat493m',
        num_classes: int = 2,
        hidden_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self._build_backbone(backbone, pretrained, freeze_backbone)

        # Adapter layers
        self.adapter = nn.Sequential(
            nn.Conv2d(self.backbone_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Classification head (per-pixel)
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

        # Regression head (per-pixel offset)
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1),
        )

        self._init_weights()
        self.criterion = criterion

    def _build_backbone(self, backbone: str, pretrained: bool, freeze: bool):
        """Build and configure backbone network."""
        if 'vit' in backbone.lower() or 'dino' in backbone.lower():
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.backbone_channels = self.backbone.embed_dim
            self.backbone_type = 'vit'
            ps = getattr(self.backbone.patch_embed, 'patch_size', (16, 16))
            self.patch_size = ps[0] if isinstance(ps, tuple) else ps

            if hasattr(self.backbone, 'num_prefix_tokens'):
                self.num_prefix_tokens = self.backbone.num_prefix_tokens
            else:
                self.num_prefix_tokens = 1
        else:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=(4,)
            )
            self.backbone_channels = self.backbone.feature_info.channels()[-1]
            self.backbone_type = 'cnn'
            self.patch_size = 16
            self.num_prefix_tokens = 0

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_weights(self):
        """Initialize weights."""
        # Bias toward background
        nn.init.constant_(self.cls_head[-1].bias[0], 2.0)
        if self.num_classes > 1:
            nn.init.constant_(self.cls_head[-1].bias[1:], -4.0)

        # Zero-init regression
        nn.init.zeros_(self.reg_head[-1].weight)
        nn.init.zeros_(self.reg_head[-1].bias)

    def set_criterion(self, criterion: nn.Module):
        """Set the loss criterion."""
        self.criterion = criterion

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict:
        """Forward pass."""
        B, _, img_h, img_w = x.shape

        # Extract features
        if self.backbone_type == 'vit':
            feat = self.backbone.forward_features(x)
            grid_h = img_h // self.patch_size
            grid_w = img_w // self.patch_size

            if feat.dim() == 3:
                if self.num_prefix_tokens > 0:
                    feat = feat[:, self.num_prefix_tokens:, :]
                feat = feat.transpose(1, 2).reshape(B, -1, grid_h, grid_w)
        else:
            feat = self.backbone(x)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            grid_h, grid_w = feat.shape[2], feat.shape[3]

        # Process features
        feat = self.adapter(feat)

        # Predictions
        logits = self.cls_head(feat)  # [B, C, H, W]
        raw_offsets = self.reg_head(feat)  # [B, 2, H, W]
        offsets = torch.tanh(raw_offsets) * 0.5

        # Decode points
        points_normalized = self._decode_points(offsets, grid_h, grid_w)

        # Scale to pixels
        points_pixel = points_normalized.clone()
        points_pixel[..., 0] *= img_w
        points_pixel[..., 1] *= img_h

        outputs = {
            'logits': logits,
            'points': points_pixel,
            'points_normalized': points_normalized,
            'offsets': offsets,
            'image_size': (img_h, img_w),
            'grid_size': (grid_h, grid_w),
        }

        if targets is not None and self.criterion is not None:
            outputs['loss_p2p'] = self.criterion(outputs, targets)

        return outputs

    def _decode_points(self, offsets: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Decode grid centers + offsets to normalized coordinates."""
        device, dtype = offsets.device, offsets.dtype

        y_centers = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        x_centers = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing='ij')
        grid = torch.stack([x_grid, y_grid], dim=0)  # [2, H, W]

        # Scale offsets by grid cell size
        offset_scaled = torch.stack([
            offsets[:, 0] / w,
            offsets[:, 1] / h
        ], dim=1)

        points = grid.unsqueeze(0) + offset_scaled  # [B, 2, H, W]
        points = points.flatten(2).transpose(1, 2)  # [B, H*W, 2]

        return points