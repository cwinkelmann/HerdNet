from typing import Optional, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .register import MODELS


class DINOv3MultiScaleProcessor(nn.Module):
    """Process DINOv3 multi-layer features into multi-scale representations."""

    def __init__(self,
                 feature_dim: int = 1024,
                 output_channels: List[int] = [256, 512, 1024],
                 num_layers: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_channels = output_channels
        self.num_layers = num_layers

        # Projection layers for each extracted layer
        self.layer_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ) for out_ch in output_channels
        ])

        # Spatial processing convolutions
        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for out_ch in output_channels
        ])

    def forward(self, layer_features: List[torch.Tensor]):
        """
        Args:
            layer_features: List of [B, C, H, W] features from different layers
        Returns:
            multi_scale_features: List of feature maps at different scales
        """
        multi_scale_features = []

        for i, (layer_feat, projector, conv) in enumerate(
                zip(layer_features, self.layer_projectors, self.spatial_convs)
        ):
            # Project features
            projected = projector(layer_feat)  # [B, out_ch, H, W]

            # Apply spatial convolution
            processed = conv(projected)

            # Create different scales
            if i == 0:  # Finest scale (upsample)
                scale_feat = F.interpolate(processed, scale_factor=2,
                                           mode='bilinear', align_corners=False)
            elif i == 1:  # Original scale
                scale_feat = processed
            else:  # Coarser scale (downsample)
                scale_feat = F.avg_pool2d(processed, kernel_size=2, stride=2)

            multi_scale_features.append(scale_feat)

        return multi_scale_features


class FeatureUpsampler(nn.Module):
    """Upsample and fuse multi-scale features for dense prediction."""

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: List[torch.Tensor]):
        """Fuse multi-scale features."""
        # Project all features to same channel dimension
        target_size = features[0].shape[2:]  # Use finest scale as target

        projected_features = []
        for feat, proj in zip(features, self.projects):
            projected = proj(feat)
            if projected.shape[2:] != target_size:
                projected = F.interpolate(projected, size=target_size,
                                          mode='bilinear', align_corners=False)
            projected_features.append(projected)

        # Fuse features
        fused = sum(projected_features)
        fused = self.fusion(fused)

        return fused


@MODELS.register()
class HerdNetDINOv3(nn.Module):
    def __init__(
            self,
            backbone='vit_large_patch16_dinov3.sat493m',
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: Optional[int] = 2,
            freeze_backbone: bool = True,
            head_conv: int = 64,
            pretrained_path=None,
            debug=True,
            out_indices: List[int] = [11],  # Which feature stages to extract
            output_channels: List[int] = [1024],
            input_resolution: tuple = (512, 512)
    ):
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], f"Invalid down_ratio: {down_ratio}"

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.out_indices = out_indices
        self.input_resolution = input_resolution

        # Load DINOv3 model from timm with features_only=True
        self.backbone = timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices  # Specify which stages to extract
        )

        data_config = timm.data.resolve_model_data_config(self.backbone)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        self.backbone.eval()  # Keep in eval mode for stable features

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if debug:
                logger.info("  DINOv3 backbone frozen")

        # Get feature info from the backbone
        feature_info = self.backbone.feature_info

        if debug:
            logger.info(f"  Backbone: {backbone}")
            logger.info(f"  Feature stages: {out_indices}")
            logger.info(f"  Feature info:")
            for i, info in enumerate(feature_info):
                logger.info(f"    Stage {i}: channels={info['num_chs']}, reduction={info['reduction']}")
            logger.info(f"  Output channels: {output_channels}")

        # Get embed_dim from feature info (use the last stage)
        self.embed_dim = feature_info[-1]['num_chs']

        # Assuming patch size based on model name (patch14 or patch16)
        if 'patch14' in backbone:
            self.patch_size = 14
        elif 'patch16' in backbone:
            self.patch_size = 16
        else:
            self.patch_size = 16  # default

        # Multi-scale processor
        self.multi_scale_processor = DINOv3MultiScaleProcessor(
            feature_dim=self.embed_dim,
            output_channels=output_channels,
            num_layers=len(out_indices)
        )

        # Feature upsampler
        self.feature_upsampler = FeatureUpsampler(
            in_channels=output_channels,
            out_channels=output_channels[0]
        )

        # Bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            output_channels[0], output_channels[0],
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Localization head (heatmap prediction)
        self.loc_head = nn.Sequential(
            nn.Conv2d(output_channels[0], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(0.0)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(output_channels[-1], head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, self.num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(0.00)

        if debug:
            self._inspect_model()

    def freeze_backbone(self):
        """Freeze all parameters in the DINOv3 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def _inspect_model(self):
        """Debug function to inspect model outputs"""
        logger.info(f"\nInspecting DINOv3 model outputs:")
        dummy_input = torch.randn(1, 3, *self.input_resolution)

        with torch.no_grad():
            # Get features - just call forward with features_only=True
            layer_features = self.backbone(dummy_input)

            logger.info(f"Extracted {len(layer_features)} feature stages:")
            for i, feat in enumerate(layer_features):
                logger.info(f"  Stage {i}: {feat.shape}")

            # Process through multi-scale
            multi_scale = self.multi_scale_processor(layer_features)
            logger.info(f"Multi-scale features: {[f.shape for f in multi_scale]}")

    def forward(self, x):
        B, C, H, W = x.shape

        # Resize input if needed to match expected resolution
        if (H, W) != self.input_resolution:
            x = F.interpolate(
                x, size=self.input_resolution,
                mode='bilinear', align_corners=False
            )

        # Extract features from backbone - features_only=True returns list of feature maps
        layer_features = self.backbone(x)  # Returns list of [B, C, H, W] tensors

        # Process into multi-scale features
        multi_scale_features = self.multi_scale_processor(layer_features)

        # Fuse features
        fused_features = self.feature_upsampler(multi_scale_features)

        # Localization heatmap
        bottleneck_features = self.bottleneck_conv(fused_features)
        heatmap = self.loc_head(bottleneck_features)

        # Classification from deepest features
        cls_out = self.cls_head(multi_scale_features[-1])

        # Resize outputs to target sizes
        cls_out_16x16 = F.interpolate(
            cls_out, size=(16, 16),
            mode='bilinear', align_corners=False
        )

        heatmap_128x128 = F.interpolate(
            heatmap, size=(128, 128),
            mode='bilinear', align_corners=False
        )

        return heatmap_128x128, cls_out_16x16

    def freeze(self, layers: list) -> None:
        """Freeze specific layers."""
        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = False

    def reshape_classes(self, num_classes: int) -> None:
        """Reshape architecture for new number of classes."""
        self.cls_head[-1] = nn.Conv2d(
            self.head_conv, num_classes,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.cls_head[-1].bias.data.fill_(0.00)
        self.num_classes = num_classes