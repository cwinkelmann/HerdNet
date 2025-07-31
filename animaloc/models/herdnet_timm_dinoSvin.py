
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .register import MODELS
from loguru import logger

class SimpleDINOv2Extractor(nn.Module):
    """Simplified DINOv2 feature extractor without attention hooks."""

    def __init__(self, dinov2_model):
        super().__init__()
        self.dinov2 = dinov2_model

    def forward(self, x):
        # Forward through DINOv2 - returns tensor directly
        features = self.dinov2.forward_features(x)  # [B, N, D]

        # Handle the case where CLS token might be included
        if features.shape[1] % (int(features.shape[1] ** 0.5) ** 2) != 0:
            # Likely includes CLS token, remove it
            features = features[:, 1:]  # [B, N-1, D] - only patch tokens

        # Create simple attention from feature magnitude
        B, N, D = features.shape
        H = W = int(N ** 0.5)

        # Feature-based attention (simple but effective)
        feature_attention = torch.norm(features, dim=2)  # [B, N]
        feature_attention = (feature_attention - feature_attention.min(dim=1, keepdim=True)[0]) / \
                          (feature_attention.max(dim=1, keepdim=True)[0] - feature_attention.min(dim=1, keepdim=True)[0] + 1e-8)

        attention_maps = {0: feature_attention}  # Simple single-layer attention

        return features, attention_maps


"""
Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
All rights reserved.

This source code is under the MIT License.

Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

Last modification: March 18, 2024
"""
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"

from typing import Optional, List
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .register import MODELS


class DINOv2AttentionExtractor(nn.Module):
    """Extract spatial attention maps from DINOv2 transformer blocks."""

    def __init__(self, dinov2_model, layer_indices: List[int] = [-4, -3, -2, -1]):
        super().__init__()
        self.dinov2 = dinov2_model
        self.layer_indices = layer_indices
        self.attention_maps = {}
        self.hooks = []

        # Register hooks to extract attention weights from specified layers
        for i, layer_idx in enumerate(layer_indices):
            # Hook into the attention module's forward method
            target_layer = self.dinov2.blocks[layer_idx].attn
            hook = target_layer.register_forward_hook(
                lambda module, input, output, idx=i: self._save_attention(module, input, output, idx)
            )
            self.hooks.append(hook)

    def _save_attention(self, module, input, output, layer_idx):
        """Extract attention weights from the attention module."""
        # For timm ViT, we need to manually compute attention weights
        # Input[0] is the input tensor [B, N, D]
        x = input[0]
        B, N, C = x.shape

        # Get q, k, v from the attention module
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]

        # Compute attention weights
        attn_weights = (q @ k.transpose(-2, -1)) * module.scale  # [B, num_heads, N, N]
        attn_weights = attn_weights.softmax(dim=-1)

        # Extract CLS to patch attention (assuming CLS token is first)
        if N > 1:  # Make sure we have patch tokens
            cls_attention = attn_weights[:, :, 0, 1:].mean(dim=1)  # [B, N-1] (average across heads)
            self.attention_maps[layer_idx] = cls_attention.detach()

    def forward(self, x):
        # Clear previous attention maps
        self.attention_maps.clear()

        # Forward through DINOv2 - returns tensor directly
        patch_features = self.dinov2.forward_features(x)  # [B, N, D]

        # Handle the case where CLS token might be included
        if patch_features.shape[1] == 1370:  # 37*37 + 1 CLS token
            # Remove CLS token if present
            patch_features = patch_features[:, 1:]  # [B, N-1, D] - only patch tokens

        return patch_features, self.attention_maps

    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()


class DINOv2SpatialProcessor(nn.Module):
    """Process DINOv2 features to create multi-scale representations."""

    def __init__(self, feature_dim: int = 1024, output_channels: List[int] = [256, 512, 1024]):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_channels = output_channels

        # Projection layers for different scales
        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, out_ch),
                nn.LayerNorm(out_ch),
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

    def forward(self, patch_features, attention_maps):
        """
        Args:
            patch_features: [B, N, D] patch features from DINOv2
            attention_maps: dict of attention maps from different layers
        Returns:
            multi_scale_features: list of feature maps at different scales
            attention_heatmaps: list of attention-based heatmaps
        """
        B, N, D = patch_features.shape
        H = W = int(N ** 0.5)  # Assuming square patches

        multi_scale_features = []
        attention_heatmaps = []

        for i, (projector, conv) in enumerate(zip(self.scale_projectors, self.spatial_convs)):
            # Project features
            projected = projector(patch_features)  # [B, N, out_ch]

            # Reshape to spatial
            spatial_feat = projected.transpose(1, 2).reshape(B, -1, H, W)  # [B, out_ch, H, W]

            # Apply spatial convolution
            processed_feat = conv(spatial_feat)

            # Create different scales through pooling/upsampling
            if i == 0:  # Finest scale (upsample)
                scale_feat = F.interpolate(processed_feat, scale_factor=2, mode='bilinear', align_corners=False)
            elif i == 1:  # Original scale
                scale_feat = processed_feat
            else:  # Coarser scale (downsample)
                scale_feat = F.avg_pool2d(processed_feat, kernel_size=2, stride=2)

            multi_scale_features.append(scale_feat)

            # Create attention heatmap if available
            if attention_maps and i in attention_maps:
                attention = attention_maps[i]  # [B, N]
                attention_heatmap = attention.reshape(B, 1, H, W)  # [B, 1, H, W]
                attention_heatmap = F.interpolate(attention_heatmap, size=scale_feat.shape[2:],
                                                mode='bilinear', align_corners=False)
                attention_heatmaps.append(attention_heatmap)
            else:
                # Create uniform attention map as fallback
                attention_heatmaps.append(
                    torch.ones(B, 1, *scale_feat.shape[2:], device=scale_feat.device) * 0.5
                )

        return multi_scale_features, attention_heatmaps


class DINOv2FeatureUpsampler(nn.Module):
    """Upsample and fuse DINOv2 features for dense prediction."""

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])

        # Additional processing for attention integration
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(len(in_channels), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, attention_maps):
        """
        Args:
            features: list of multi-scale features
            attention_maps: list of attention heatmaps
        """
        # Project all features to same channel dimension
        projected_features = []
        target_size = features[0].shape[2:]  # Use finest scale as target

        for i, (feat, proj) in enumerate(zip(features, self.projects)):
            projected = proj(feat)
            # Resize to target size
            if projected.shape[2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode='bilinear', align_corners=False)
            projected_features.append(projected)

        # Fuse features
        fused_features = sum(projected_features)

        # Integrate attention maps
        attention_stack = []
        for attention in attention_maps:
            if attention.shape[2:] != target_size:
                attention = F.interpolate(attention, size=target_size, mode='bilinear', align_corners=False)
            attention_stack.append(attention)

        if attention_stack:
            attention_tensor = torch.cat(attention_stack, dim=1)  # [B, num_layers, H, W]
            attention_features = self.attention_fusion(attention_tensor)

            # Combine features with attention guidance
            fused_features = fused_features + attention_features

        return fused_features


def _load_backbone_checkpoint(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Optionally remove "module." if trained with DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model


@MODELS.register()
class HerdNetDINOv2(nn.Module):
    def __init__(
        self,
        num_layers: int = 34,  # Keep for compatibility, not used with DINOv2
        num_classes: int = 2,
        pretrained: bool = True,
        down_ratio: Optional[int] = 2,
        head_conv: int = 64,
        pretrained_path=None,
        debug=True,
        attention_layers: List[int] = [-4, -3, -2, -1],  # Which transformer layers to extract attention from
    ):
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], f"Invalid down_ratio: {down_ratio}"

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.attention_layers = attention_layers

        # Load DINOv2 model from timm
        dinov2_model = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        if pretrained_path:
            dinov2_model = _load_backbone_checkpoint(dinov2_model, pretrained_path)

        # Extract model info
        self.patch_size = dinov2_model.patch_embed.patch_size[0]
        self.embed_dim = dinov2_model.embed_dim  # 1024 for large model

        if debug:
            logger.info(f"\nDINOv2 Large model loaded:")
            logger.info(f"  Patch size: {self.patch_size}x{self.patch_size}")
            logger.info(f"  Embedding dim: {self.embed_dim}")
            logger.info(f"  Attention extraction layers: {attention_layers}")

        # Attention extractor - try hook-based first, fallback to simple
        try:
            self.attention_extractor = DINOv2AttentionExtractor(dinov2_model, attention_layers)
            self.use_hook_attention = True
            attention_channels = len(attention_layers)
            if debug:
                logger.info("Using hook-based attention extraction")
        except Exception as e:
            if debug:
                logger.error(f"Hook-based attention failed ({e}), using simple feature-based attention")
            self.attention_extractor = SimpleDINOv2Extractor(dinov2_model)
            self.use_hook_attention = False
            attention_channels = 1

        # Spatial processor for multi-scale features
        output_channels = [256, 512, 1024]
        self.spatial_processor = DINOv2SpatialProcessor(
            feature_dim=self.embed_dim,
            output_channels=output_channels
        )

        # Feature upsampler with attention integration
        self.feature_upsampler = DINOv2FeatureUpsampler(
            in_channels=output_channels,
            out_channels=output_channels[0]
        )

        # Bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            output_channels[0], output_channels[0],
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Localization head (for heatmap prediction)
        self.loc_head = nn.Sequential(
            nn.Conv2d(output_channels[0], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(0.0)

        # # Classification head (using deepest features)
        # self.cls_head = nn.Sequential(
        #     nn.Conv2d(output_channels[-1], head_conv, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(head_conv, num_classes, kernel_size=1)
        # )
        # self.cls_head[-1].bias.data.fill_(0.0)

        # classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(output_channels[-1], head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, self.num_classes,
                kernel_size=1, stride=1,
                padding=0, bias=True
                )
            )
        self.cls_head[-1].bias.data.fill_(0.00)

        # Attention heatmap head (coarse attention-based heatmap)
        self.attention_head = nn.Sequential(
            nn.Conv2d(attention_channels, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Fallback feature-based attention (if hook-based attention fails)
        self.feature_attention = nn.Sequential(
            nn.Conv2d(output_channels[0], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
            nn.Sigmoid()
        )

        if debug:
            self._inspect_model()

    def _inspect_model(self):
        """Debug function to inspect model outputs"""
        logger.info(f"\nInspecting DINOv2 model outputs:")
        dummy_input = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            patch_features, attention_maps = self.attention_extractor(dummy_input)

            logger.info(f"Patch features shape: {patch_features.shape}")
            logger.info(f"Number of patches: {patch_features.shape[1]}")
            logger.info(f"Feature dimension: {patch_features.shape[2]}")

            # Calculate spatial dimensions
            N = patch_features.shape[1]
            H = W = int(N ** 0.5)
            logger.info(f"Spatial grid: {H}x{W}")

            logger.info(f"Attention maps extracted from {len(attention_maps)} layers:")
            for layer_idx, attn in attention_maps.items():
                logger.info(f"  Layer {layer_idx}: {attn.shape}")

            if attention_maps:
                multi_scale_features, attention_heatmaps = self.spatial_processor(patch_features, attention_maps)
                logger.info(f"Multi-scale features: {[f.shape for f in multi_scale_features]}")
                logger.info(f"Attention heatmaps: {[h.shape for h in attention_heatmaps]}")
            else:
                raise ValueError("No attention maps extracted - check hook setup")

    def forward(self, x):
        # Extract DINOv2 features and attention maps
        patch_features, attention_maps = self.attention_extractor(x)

        # Process into multi-scale features and attention heatmaps
        multi_scale_features, attention_heatmaps = self.spatial_processor(patch_features, attention_maps)

        # Fuse features with attention guidance
        fused_features = self.feature_upsampler(multi_scale_features, attention_heatmaps)

        # Localization heatmap from fused features
        bottleneck_features = self.bottleneck_conv(fused_features)
        heatmap = self.loc_head(bottleneck_features)  # [B, 1, H, W]

        # Classification from deepest features
        cls_out = self.cls_head(multi_scale_features[-1])  # [B, num_classes]
        # Enforce classification output to be exactly 16x16
        cls_out_16x16 = F.interpolate(
            cls_out,
            size=(16, 16),
            mode='bilinear',
            align_corners=False
        )  # [B, num_classes, 16, 16]

        # # Coarse attention-based heatmap
        # if attention_maps and len(attention_heatmaps) > 0:
        #     # Use real attention maps if available
        #     # Resize all attention maps to the same size (use the largest one)
        #     target_size = max([h.shape[2:] for h in attention_heatmaps], key=lambda x: x[0] * x[1])
        #
        #     resized_attention_maps = []
        #     for attention_map in attention_heatmaps:
        #         if attention_map.shape[2:] != target_size:
        #             resized = F.interpolate(attention_map, size=target_size, mode='bilinear', align_corners=False)
        #             resized_attention_maps.append(resized)
        #         else:
        #             resized_attention_maps.append(attention_map)
        #
        #     attention_stack = torch.cat(resized_attention_maps, dim=1)  # [B, num_layers, H, W]
        #     attention_heatmap = self.attention_head(attention_stack)  # [B, 1, H, W]
        # else:
        #     # Fallback: use feature-based attention
        #     attention_heatmap = self.feature_attention(fused_features)  # [B, 1, H, W]
        #
        # return {
        #     'heatmap': heatmap,
        #     'classification': cls_out,
        #     'attention_heatmap': attention_heatmap,
        #     'features': fused_features
        # }

        heatmap_upscaled = F.interpolate(
            heatmap,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )  # [B, 1, 128, 128]

        # TODO this headtmap size depends on the downn_ratio
        return heatmap_upscaled, cls_out_16x16

    def freeze(self, layers: list) -> None:
        """Freeze all layers mentioned in the input list"""
        for layer in layers:
            self._freeze_layer(layer)

    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def reshape_classes(self, num_classes: int) -> None:
        """Reshape architecture according to a new number of classes."""
        self.cls_head[-1] = nn.Linear(self.head_conv, num_classes)
        self.cls_head[-1].bias.data.fill_(0.00)
        self.num_classes = num_classes

    def get_attention_maps(self, x):
        """Extract and return attention maps for visualization."""
        with torch.no_grad():
            _, attention_maps = self.attention_extractor(x)

            if not attention_maps:
                # Fallback: create attention from feature magnitude
                patch_features, _ = self.attention_extractor(x)
                B, N, D = patch_features.shape
                H = W = int(N ** 0.5)

                # Use feature magnitude as proxy for attention
                feature_magnitude = torch.norm(patch_features, dim=2)  # [B, N]
                feature_magnitude = (feature_magnitude - feature_magnitude.min(dim=1, keepdim=True)[0]) / \
                                  (feature_magnitude.max(dim=1, keepdim=True)[0] - feature_magnitude.min(dim=1, keepdim=True)[0] + 1e-8)

                return {0: feature_magnitude.reshape(B, 1, H, W)}

            # Convert to spatial format for visualization
            B = x.shape[0]
            spatial_attention = {}

            for layer_idx, attention in attention_maps.items():
                N = attention.shape[1]
                H = W = int(N ** 0.5)
                spatial_attention[layer_idx] = attention.reshape(B, 1, H, W)

            return spatial_attention

    def __del__(self):
        """Clean up hooks when model is deleted."""
        if hasattr(self, 'attention_extractor'):
            self.attention_extractor.remove_hooks()