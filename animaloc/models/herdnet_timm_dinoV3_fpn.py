from typing import Optional, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .register import MODELS


class DINOv3MultiScaleProcessor(nn.Module):
    """
    Projects different ViT layers to specific channel dimensions and
    applies spatial convolutions.
    """

    def __init__(self,
                 in_channels_list: List[int],
                 output_channels: List[int]):
        super().__init__()

        # This is where your error happened previously.
        # We now ensure these lists are synced before calling this class.
        assert len(in_channels_list) == len(output_channels), \
            f"Input channels ({len(in_channels_list)}) and output channels ({len(output_channels)}) must match."

        self.num_layers = len(in_channels_list)

        self.layer_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ) for in_ch, out_ch in zip(in_channels_list, output_channels)
        ])

        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for out_ch in output_channels
        ])

    def forward(self, layer_features: List[torch.Tensor]) -> List[torch.Tensor]:
        processed_features = []
        for feat, projector, conv in zip(layer_features, self.layer_projectors, self.spatial_convs):
            x = projector(feat)
            x = conv(x)
            processed_features.append(x)
        return processed_features


class FeatureFusionHead(nn.Module):
    """Fuses the processed features into a single feature map."""

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1)
            if ch != out_channels else nn.Identity()
            for ch in in_channels_list
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        target_size = projected[0].shape[2:]
        fused = projected[0]
        for i in range(1, len(projected)):
            p = projected[i]
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            fused = fused + p
        return self.fusion_conv(fused)


import torch
import torch.nn as nn
import math


class UpsampleBlock(nn.Module):
    """
    A single block that doubles resolution:
    Upsample (2x) -> Conv -> BN -> ReLU
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DynamicHeatmapDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 down_ratio: int = 4,
                 backbone_stride: int = 16,
                 hidden_dim: int = 64,
                 out_channels: int = 1):
        super().__init__()

        # 1. Calculate required upsampling factor
        # Example: Backbone stride 16 (32px), Target stride 4 (128px) -> Factor = 4x
        total_upscale_factor = backbone_stride // down_ratio

        # Calculate how many 2x stages we need (log2)
        # 8x upsample -> 3 stages
        # 4x upsample -> 2 stages
        # 2x upsample -> 1 stage
        num_stages = int(math.log2(total_upscale_factor))

        assert num_stages >= 0, f"Down ratio {down_ratio} is too large for backbone stride {backbone_stride}"

        # 2. Build Layers Dynamically
        layers = []
        current_ch = in_channels

        for i in range(num_stages):
            # Gradual channel reduction or keep constant?
            # Strategy: Drop to hidden_dim immediately, then maintain
            next_ch = hidden_dim if i == 0 else hidden_dim

            # Or Strategy B: Halve channels every step until hidden_dim
            # next_ch = max(current_ch // 2, hidden_dim)

            layers.append(UpsampleBlock(current_ch, next_ch))
            current_ch = next_ch

        self.decoder = nn.Sequential(*layers)

        # 3. Final Prediction Head
        self.head = nn.Sequential(
            nn.Conv2d(current_ch, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.head[0].bias.data.fill_(-4.6)  # Standard init for Focal Loss

    def forward(self, x):
        x = self.decoder(x)
        x = self.head(x)
        return x



@MODELS.register()
class HerdNetDINOv3Fusion(nn.Module):
    def __init__(
            self,
            # backbone='vit_large_patch16_dinov3.sat493m',
            backbone='vit_base_patch16_dinov3.sat493m',
            num_classes: int = 2,
            pretrained: bool = True,
            freeze_backbone: bool = True,
            head_conv: int = 64,
            out_indices: Optional[List[int]] = None,
            hidden_dims: List[int] = [512, 768, 1024],
            input_resolution: Tuple[int, int] = (512, 512),
            down_ratio=4,
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_resolution = tuple(input_resolution)

        # 1. Determine Indices
        # Load temp model just to check depth
        logger.info(f"Loading backbone structure: {backbone}")
        raw_model = timm.create_model(backbone, pretrained=False, num_classes=0)
        total_blocks = len(raw_model.blocks)
        del raw_model

        if out_indices is None:
            # Auto-select: Middle, Late-Middle, Final
            if total_blocks >= 24:
                self.out_indices = [11, 17, 23]
            else:
                self.out_indices = [5, 8, 11]
        else:
            self.out_indices = [i if i < total_blocks else total_blocks - 1 for i in out_indices]

        # Ensure hidden_dims matches out_indices length
        if len(hidden_dims) != len(self.out_indices):
            logger.warning(f"Adjusting hidden_dims ({len(hidden_dims)}) to match indices ({len(self.out_indices)})")
            hidden_dims = [hidden_dims[-1]] * len(self.out_indices)

        # 2. Load Real Backbone
        self.backbone = timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen.")
        # TODO check for trainable param
        # 3. FIX: Robust Channel Extraction
        feature_info = self.backbone.feature_info

        # If feature_info has more entries than we asked for (e.g., 12 vs 3),
        # we manually select the channels corresponding to our indices.
        if len(feature_info) > len(self.out_indices):
            # We assume feature_info maps 1:1 to blocks when it returns all blocks
            in_channels_list = [feature_info[i]['num_chs'] for i in self.out_indices]
        else:
            # It matches exactly
            in_channels_list = [info['num_chs'] for info in feature_info]

        logger.info(f"Indices: {self.out_indices}")
        logger.info(f"Channels: {in_channels_list}")

        # 4. Processors
        self.processor = DINOv3MultiScaleProcessor(
            in_channels_list=in_channels_list,
            output_channels=hidden_dims
        )

        self.fusion_dim = hidden_dims[0]
        self.fusion = FeatureFusionHead(
            in_channels_list=hidden_dims,
            out_channels=self.fusion_dim
        )

        # # 5. Heads
        # self.loc_head = nn.Sequential(
        #     nn.Conv2d(self.fusion_dim, head_conv, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(head_conv, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.loc_head[-2].bias.data.fill_(0.0)
        self.loc_head = DynamicHeatmapDecoder(
            in_channels=self.fusion_dim,
            down_ratio=down_ratio,  # e.g., 2 or 4
            backbone_stride=16,  # DINO standard
            hidden_dim=head_conv,
            out_channels=1
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(self.fusion_dim, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(head_conv, num_classes)
        )
        self.cls_head[-1].bias.data.fill_(0.0)

    def forward(self, x, debug=False):

        if x.shape[2:] != self.input_resolution:
            x = F.interpolate(x, size=self.input_resolution, mode='bicubic', align_corners=False)

        # 1. Extract (Returns list of tensors)
        raw_features = self.backbone(x)

        # 2. Process
        processed = self.processor(raw_features)

        # 3. Fuse
        fused = self.fusion(processed)

        # 4. Heads
        heatmap = self.loc_head(fused)
        # Ensure it matches 128x128 exactly (sanity check)
        if heatmap.shape[2:] != (128, 128):
            heatmap = F.interpolate(heatmap, size=(128, 128), mode='bilinear', align_corners=False)

        cls_logits = self.cls_head(fused)  # [B, num_classes]

        # --- DEBUG RETURN ---
        if debug:
            # Helper to resize for visualization consistency
            def _resize(t):
                return F.interpolate(t, size=(128, 128), mode='bilinear', align_corners=False)

            return {
                'prediction': heatmap,
                # Store dictionary of raw backbone layers
                'backbone': {f'layer_{k}': _resize(v) for k, v in zip(self.out_indices, raw_features)},
                # Store processed layers (useful to see if 1x1 conv killed the signal)
                'processed': {f'layer_{k}': _resize(v) for k, v in zip(self.out_indices, processed)},
                # Store the fused map (what the head actually sees)
                'fused': _resize(fused)
            }

        # Legacy reshape
        cls_out = cls_logits.view(cls_logits.size(0), self.num_classes, 1, 1)
        cls_out_16x16 = F.interpolate(cls_out, size=(16, 16), mode='nearest')

        return heatmap, cls_out_16x16

    def reshape_classes(self, num_classes: int):
        self.num_classes = num_classes
        in_features = self.cls_head[-1].in_features
        self.cls_head[-1] = nn.Linear(in_features, num_classes)
        self.cls_head[-1].bias.data.fill_(0.0)