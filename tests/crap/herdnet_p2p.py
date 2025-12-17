"""
HerdNetP2P v4 - Fixed reference points with learned offsets

Key insight: The coordinate head outputs similar values for all queries
because query embeddings don't encode enough positional information.

Solution: Each query has a FIXED reference point (not learned), and the
coordinate head predicts OFFSETS from these fixed anchors. This guarantees
diverse initial predictions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, List


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal positional encoding for 2D feature maps."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype

        y_embed = torch.arange(H, device=device, dtype=dtype).view(-1, 1).expand(H, W)
        x_embed = torch.arange(W, device=device, dtype=dtype).view(1, -1).expand(H, W)

        y_embed = y_embed / (H - 1 + 1e-6) * self.scale
        x_embed = x_embed / (W - 1 + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)
        return pos.unsqueeze(0).expand(B, -1, -1, -1)


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, memory, query_pos, memory_pos):
        q = k = query + query_pos
        query = query + self.dropout1(self.self_attn(q, k, query)[0])
        query = self.norm1(query)

        query = query + self.dropout2(self.cross_attn(query + query_pos, memory + memory_pos, memory)[0])
        query = self.norm2(query)

        query = query + self.ffn(query)
        query = self.norm3(query)
        return query


class HerdNetP2P(nn.Module):
    """
    P2PNet with FIXED reference points and learned offsets.

    Each of the num_queries has a fixed anchor position on a grid.
    The coordinate head predicts offsets from these anchors.
    Final position = anchor + tanh(offset) * max_offset

    This guarantees diverse predictions from initialization.
    """

    def __init__(
        self,
        backbone: str = 'vit_large_patch16_dinov3.sat493m',
        num_classes: int = 2,
        num_queries: int = 25,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        max_offset: float = 0.2,  # Max offset from reference point (as fraction of image)
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.max_offset = max_offset

        # Build backbone
        self._build_backbone(backbone, pretrained, freeze_backbone)

        # Projection
        self.input_proj = nn.Conv2d(self.backbone_channels, hidden_dim, 1)
        self.pos_encoder = PositionEmbeddingSine(hidden_dim // 2)

        # Learnable query content embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Learnable query positional embeddings
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        # FIXED reference points on a grid (not learned!)
        # These are registered as a buffer, not a parameter
        self.register_buffer('reference_points', self._create_reference_grid())

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

        # Offset head - predicts (dx, dy) offset from reference point
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        self._init_weights()
        self.criterion = criterion

    def _create_reference_grid(self) -> torch.Tensor:
        """Create fixed reference points on a grid."""
        grid_size = int(math.ceil(math.sqrt(self.num_queries)))

        coords = []
        for i in range(self.num_queries):
            row = i // grid_size
            col = i % grid_size
            # Position in [0, 1], offset from edges
            x = (col + 0.5) / grid_size
            y = (row + 0.5) / grid_size
            coords.append([x, y])

        return torch.tensor(coords, dtype=torch.float32)

    def _build_backbone(self, backbone: str, pretrained: bool, freeze: bool):
        if 'vit' in backbone.lower() or 'dino' in backbone.lower():
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.backbone_channels = self.backbone.embed_dim
            self.backbone_type = 'vit'
            ps = getattr(self.backbone.patch_embed, 'patch_size', (16, 16))
            self.patch_size = ps[0] if isinstance(ps, tuple) else ps
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(4,))
            self.backbone_channels = self.backbone.feature_info.channels()[-1]
            self.backbone_type = 'cnn'
            self.patch_size = 16

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_weights(self):
        # Classification head
        for layer in self.class_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Slight bias toward background
        nn.init.constant_(self.class_head[-1].bias[0], 0.5)
        if self.num_classes > 1:
            nn.init.constant_(self.class_head[-1].bias[1:], -0.5)

        # Offset head - initialize to output near-zero offsets
        for layer in self.offset_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        # Last layer outputs ~0 initially
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def set_criterion(self, criterion: nn.Module):
        self.criterion = criterion

    def forward(self, x: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict:
        B, _, img_h, img_w = x.shape

        # Extract features
        if self.backbone_type == 'vit':
            feat = self.backbone.forward_features(x)
            grid_h, grid_w = img_h // self.patch_size, img_w // self.patch_size
            if feat.dim() == 3:
                expected = grid_h * grid_w
                if feat.shape[1] > expected:
                    feat = feat[:, -expected:]
                feat = feat.transpose(1, 2).reshape(B, -1, grid_h, grid_w)
        else:
            feat = self.backbone(x)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            grid_h, grid_w = feat.shape[2], feat.shape[3]

        # Project + positional encoding
        feat = self.input_proj(feat)
        pos = self.pos_encoder(feat)

        # Flatten for transformer
        memory = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        memory_pos = pos.flatten(2).transpose(1, 2)

        # Queries
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Decode
        for layer in self.decoder_layers:
            query = layer(query, memory, query_pos, memory_pos)

        # Classification
        pred_logits = self.class_head(query)  # [B, num_queries, num_classes]

        # Coordinate prediction: reference + offset
        offsets = self.offset_head(query)  # [B, num_queries, 2]
        offsets = torch.tanh(offsets) * self.max_offset  # Bounded offsets

        # Add offsets to reference points
        ref_pts = self.reference_points.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, 2]
        pred_points_norm = (ref_pts + offsets).clamp(0, 1)  # [B, num_queries, 2] in [0, 1]

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
            # Aliases
            'logits': pred_logits,
            'points': pred_points,
            'points_normalized': pred_points_norm,
        }

        if targets is not None and self.criterion is not None:
            outputs['loss_p2p'] = self.criterion(outputs, targets)

        return outputs


class P2PNetLite(nn.Module):
    """Dense grid P2PNet - naturally has spatial diversity."""

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

        self.adapter = nn.Sequential(
            nn.Conv2d(self.backbone_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1),
        )

        self._init_weights()
        self.criterion = criterion

    def _build_backbone(self, backbone: str, pretrained: bool, freeze: bool):
        if 'vit' in backbone.lower() or 'dino' in backbone.lower():
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.backbone_channels = self.backbone.embed_dim
            self.backbone_type = 'vit'
            ps = getattr(self.backbone.patch_embed, 'patch_size', (16, 16))
            self.patch_size = ps[0] if isinstance(ps, tuple) else ps
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(4,))
            self.backbone_channels = self.backbone.feature_info.channels()[-1]
            self.backbone_type = 'cnn'
            self.patch_size = 16

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_weights(self):
        nn.init.constant_(self.cls_head[-1].bias[0], 2.0)
        if self.num_classes > 1:
            nn.init.constant_(self.cls_head[-1].bias[1:], -4.0)
        nn.init.zeros_(self.reg_head[-1].weight)
        nn.init.zeros_(self.reg_head[-1].bias)

    def set_criterion(self, criterion: nn.Module):
        self.criterion = criterion

    def forward(self, x: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict:
        B, _, img_h, img_w = x.shape

        if self.backbone_type == 'vit':
            feat = self.backbone.forward_features(x)
            grid_h, grid_w = img_h // self.patch_size, img_w // self.patch_size
            if feat.dim() == 3:
                expected = grid_h * grid_w
                if feat.shape[1] > expected:
                    feat = feat[:, -expected:]
                feat = feat.transpose(1, 2).reshape(B, -1, grid_h, grid_w)
        else:
            feat = self.backbone(x)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            grid_h, grid_w = feat.shape[2], feat.shape[3]

        feat = self.adapter(feat)
        logits = self.cls_head(feat)
        raw_offsets = self.reg_head(feat)
        offsets = torch.tanh(raw_offsets) * 0.5

        points_normalized = self._decode_points(offsets, grid_h, grid_w)

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
        device, dtype = offsets.device, offsets.dtype

        y_centers = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        x_centers = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing='ij')
        grid = torch.stack([x_grid, y_grid], dim=0)

        offset_scaled = torch.stack([offsets[:, 0] / w, offsets[:, 1] / h], dim=1)
        points = grid.unsqueeze(0) + offset_scaled
        points = points.flatten(2).transpose(1, 2)

        return points