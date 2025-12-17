__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: December 16, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.2"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, List
from .register import MODELS

# Assuming you have timm installed: pip install timm
import timm


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder for multi-scale feature fusion"""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        """
        Args:
            in_channels: List of channel dimensions from Swin stages [C1, C2, C3, C4]
            out_channels: Output channels for FPN features
        """
        super(FPNDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lateral connections (1x1 conv to match channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels
        ])

        # Output convs (3x3 conv to reduce aliasing after upsampling)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from Swin stages [C1, C2, C3, C4]
                     from low-level to high-level
        Returns:
            List of FPN features at multiple scales
        """
        # Build laterals
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # Apply output convolutions
        outputs = [
            output_conv(lateral)
            for lateral, output_conv in zip(laterals, self.output_convs)
        ]

        return outputs


class UpsampleFusion(nn.Module):
    """Upsample and fuse FPN features for final prediction"""

    def __init__(self, in_channels: int, out_channels: int, num_stages: int):
        """
        Args:
            in_channels: Input channels from FPN
            out_channels: Output channels after fusion
            num_stages: Number of FPN stages to fuse
        """
        super(UpsampleFusion, self).__init__()

        self.num_stages = num_stages

        # Fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_stages, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, fpn_features: List[torch.Tensor], target_size: tuple) -> torch.Tensor:
        """
        Args:
            fpn_features: List of FPN features
            target_size: Target spatial size (H, W)
        Returns:
            Fused feature map
        """
        # Upsample all features to target size
        upsampled = []
        for feat in fpn_features[:self.num_stages]:
            upsampled.append(
                F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            )

        # Concatenate and fuse
        fused = torch.cat(upsampled, dim=1)
        output = self.fusion_conv(fused)

        return output


@MODELS.register()
class SwinHerdNet(nn.Module):
    """HerdNet architecture with Swin Transformer backbone"""

    def __init__(
            self,
            model_name: str = 'swinv2_tiny_window8_256',
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: Optional[int] = 2,
            head_conv: int = 64,
            fpn_channels: int = 256,
            freeze_backbone: bool = False,
            use_fpn_stages: int = 3  # Use top 3 FPN stages for fusion
    ):
        """
        Args:
            model_name (str): Swin model variant compatible with 512x512 images.
                IMPORTANT: Must use window_size=8 or 16 for 512x512 compatibility!
                Options:
                - 'swinv2_tiny_window8_256': 28M params, window=8, [96,192,384,768] channels ✓ RECOMMENDED
                - 'swinv2_small_window8_256': 50M params, window=8, [96,192,384,768] channels ✓
                - 'swinv2_base_window8_256': 88M params, window=8, [128,256,512,1024] channels ✓
                - 'swinv2_tiny_window16_256': 28M params, window=16, [96,192,384,768] channels ✓
                Note: Models are loaded with img_size=512 and dynamic_img_size=True
            num_classes (int): Number of output classes, background included
            pretrained (bool): Use ImageNet pretrained weights
            down_ratio (int): Output downsample ratio. Options: 1, 2, 4, 8
            head_conv (int): Channels in prediction heads
            fpn_channels (int): Channels in FPN features
            freeze_backbone (bool): Freeze Swin backbone weights
            use_fpn_stages (int): Number of FPN stages to fuse (2-4)
        """
        super(SwinHerdNet, self).__init__()

        assert down_ratio in [1, 2, 4, 8], \
            f'Downsample ratio possible values are 1, 2, 4, or 8, got {down_ratio}'

        assert use_fpn_stages in [2, 3, 4], \
            f'use_fpn_stages must be 2, 3, or 4, got {use_fpn_stages}'

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.fpn_channels = fpn_channels

        # Load Swin Transformer backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,  # Return intermediate features
            out_indices=(0, 1, 2, 3),  # Output from all 4 stages
            img_size=512,  # Set input size to 512x512
            dynamic_img_size=True  # Allow flexible input sizes
        )

        # Get feature dimensions from backbone
        # Swin stages output at different resolutions:
        # Stage 0: H/4, W/4
        # Stage 1: H/8, W/8
        # Stage 2: H/16, W/16
        # Stage 3: H/32, W/32
        feature_info = self.backbone.feature_info.channels()
        self.backbone_channels = feature_info  # e.g., [96, 192, 384, 768] for Swin-T

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Feature Pyramid Network
        self.fpn = FPNDecoder(
            in_channels=self.backbone_channels,
            out_channels=fpn_channels
        )

        # Calculate target spatial dimensions for heatmap
        # Input is 512x512, we want output based on down_ratio
        self.heatmap_size = (512 // down_ratio, 512 // down_ratio)

        # Upsample and fusion module for localization
        self.loc_fusion = UpsampleFusion(
            in_channels=fpn_channels,
            out_channels=head_conv,
            num_stages=use_fpn_stages
        )

        # Localization head (outputs heatmap)
        self.loc_head = nn.Sequential(
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(-2.00)

        # Classification head (operates on highest-level features)
        # Uses Stage 3 features (H/32, W/32) -> 16x16 for 512x512 input
        self.cls_reduce = nn.Sequential(
            nn.Conv2d(self.backbone_channels[-1], fpn_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(fpn_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, self.num_classes, kernel_size=1, stride=1,
                      padding=0, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(0.00)

    def forward(self, input: torch.Tensor) -> tuple:
        """
        Args:
            input: Input tensor of shape (B, 3, H, W)

        Returns:
            heatmap: Localization heatmap (B, 1, H/down_ratio, W/down_ratio)
            clsmap: Classification map (B, num_classes, H/32, W/32)
        """
        # Extract multi-scale features from Swin backbone
        # features: [Stage0, Stage1, Stage2, Stage3]
        # Swin outputs in (B, H, W, C) format, need to convert to (B, C, H, W)
        # Shapes for 512x512 input AFTER permutation:
        #   Stage0: (B, 96, 128, 128)   - H/4, W/4
        #   Stage1: (B, 192, 64, 64)    - H/8, W/8
        #   Stage2: (B, 384, 32, 32)    - H/16, W/16
        #   Stage3: (B, 768, 16, 16)    - H/32, W/32
        features = self.backbone(input)

        # Convert from (B, H, W, C) to (B, C, H, W) format
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        # Build Feature Pyramid
        fpn_features = self.fpn(features)

        # Localization branch: fuse multi-scale FPN features
        loc_features = self.loc_fusion(fpn_features, self.heatmap_size)
        heatmap = self.loc_head(loc_features)

        # Classification branch: use highest-level semantic features
        cls_features = self.cls_reduce(features[-1])
        clsmap = self.cls_head(cls_features)

        # Assertions for expected output shapes (512x512 input)
        if self.down_ratio == 1:
            assert heatmap.shape[-2:] == (512, 512), \
                f"Expected heatmap shape (512, 512), got {heatmap.shape[-2:]}"
        elif self.down_ratio == 2:
            assert heatmap.shape[-2:] == (256, 256), \
                f"Expected heatmap shape (256, 256), got {heatmap.shape[-2:]}"
        elif self.down_ratio == 4:
            assert heatmap.shape[-2:] == (128, 128), \
                f"Expected heatmap shape (128, 128), got {heatmap.shape[-2:]}"
        elif self.down_ratio == 8:
            assert heatmap.shape[-2:] == (64, 64), \
                f"Expected heatmap shape (64, 64), got {heatmap.shape[-2:]}"

        assert clsmap.shape[-2:] == (16, 16), \
            f"Expected clsmap shape (16, 16), got {clsmap.shape[-2:]}"

        return heatmap, clsmap

    def freeze(self, layers: list) -> None:
        """Freeze specific layers"""
        for layer in layers:
            self._freeze_layer(layer)

    def _freeze_layer(self, layer_name: str) -> None:
        """Freeze all parameters in a layer"""
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def reshape_classes(self, num_classes: int) -> None:
        """Reshape architecture according to a new number of classes

        Args:
            num_classes (int): New number of classes
        """
        self.cls_head[-1] = nn.Conv2d(
            self.head_conv, num_classes,
            kernel_size=1, stride=1,
            padding=0, bias=True
        )
        self.cls_head[-1].bias.data.fill_(0.00)
        self.num_classes = num_classes

    def get_parameter_groups(self,
                             lr_backbone: float = 1e-5,
                             lr_new: float = 1e-4) -> List[dict]:
        """Get parameter groups with different learning rates

        Args:
            lr_backbone: Learning rate for pretrained backbone
            lr_new: Learning rate for newly initialized layers

        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = []
        new_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                new_params.append(param)

        return [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': new_params, 'lr': lr_new}
        ]


# Convenience functions for common configurations (512x512 compatible!)
def swin_tiny_herdnet(num_classes: int = 2, pretrained: bool = True,
                      down_ratio: int = 2, **kwargs):
    """Swin-Tiny HerdNet (~28M params) - Compatible with 512x512 images"""
    return SwinHerdNet(
        model_name='swinv2_tiny_window8_256',
        num_classes=num_classes,
        pretrained=pretrained,
        down_ratio=down_ratio,
        **kwargs
    )


def swin_small_herdnet(num_classes: int = 2, pretrained: bool = True,
                       down_ratio: int = 2, **kwargs):
    """Swin-Small HerdNet (~50M params) - Compatible with 512x512 images"""
    return SwinHerdNet(
        model_name='swinv2_small_window8_256',
        num_classes=num_classes,
        pretrained=pretrained,
        down_ratio=down_ratio,
        **kwargs
    )


def swin_base_herdnet(num_classes: int = 2, pretrained: bool = True,
                      down_ratio: int = 2, **kwargs):
    """Swin-Base HerdNet (~88M params) - Compatible with 512x512 images"""
    return SwinHerdNet(
        model_name='swinv2_base_window8_256',
        num_classes=num_classes,
        pretrained=pretrained,
        down_ratio=down_ratio,
        **kwargs
    )


if __name__ == '__main__':
    # Test the model
    model = swin_tiny_herdnet(num_classes=2, pretrained=False, down_ratio=2)

    # Test input
    x = torch.randn(2, 3, 512, 512)

    # Forward pass
    heatmap, clsmap = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Classification map shape: {clsmap.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")