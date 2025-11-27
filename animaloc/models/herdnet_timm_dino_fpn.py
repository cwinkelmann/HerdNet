from typing import Optional, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from . import HerdNetDINOv2
from .register import MODELS




class DINOv2AttentionExtractor(nn.Module):
    """
    Extract spatial attention maps from DINOv2 transformer blocks.
    """

    def __init__(self, dinov2_model,
                 layer_indices: List[int] = [-4, -3, -2, -1]):
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

    def __init__(self,
                 feature_dim: int = 1024,
                 output_channels: List[int] = [256, 512, 1024]):
        """
        Args:
            feature_dim: Dimension of DINOv2 patch features
            output_channels: List of output channels for different scales
                           Should be ordered from COARSE to FINE for FPN compatibility
        """
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
            multi_scale_features: list of feature maps at different scales (COARSE to FINE)
            attention_heatmaps: list of attention-based heatmaps (COARSE to FINE)
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
            # REVERSED ORDER: Now creating COARSE to FINE (matching FPN expectations)
            if i == 0:  # Coarsest scale (downsample)
                scale_feat = F.avg_pool2d(processed_feat, kernel_size=2, stride=2)
            elif i == 1:  # Original scale
                scale_feat = processed_feat
            else:  # Finest scale (upsample)
                scale_feat = F.interpolate(processed_feat, scale_factor=2, mode='bilinear', align_corners=False)

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


class FPNModule(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # Lateral connections (1x1 convs to unify channel dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])

        # Output convs (3x3 convs to refine features)
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: list of [B, C_i, H_i, W_i] from coarse to fine
                     e.g., [coarse_feat, mid_feat, fine_feat]
        Returns:
            fpn_features: list of [B, out_channels, H_i, W_i] at each scale
        """
        # Apply lateral convolutions
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]

        # Top-down pathway with lateral connections
        # Start from coarsest level (smallest spatial size)
        fpn_features = []

        # Process from coarse to fine (bottom-up in the list)
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # Coarsest level - no upsampling needed
                fpn_out = laterals[i]
            else:
                # Upsample previous level and add to current lateral
                upsampled = F.interpolate(
                    fpn_features[0],
                    size=laterals[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                fpn_out = laterals[i] + upsampled

            # Refine with 3x3 conv
            fpn_out = self.fpn_convs[i](fpn_out)
            fpn_features.insert(0, fpn_out)  # Insert at beginning to maintain order

        return fpn_features


class EnhancedAttentionFusion(nn.Module):
    """Learnable weighted fusion of multi-layer attention maps."""

    def __init__(self, num_attention_layers, out_channels, use_spatial_attention=True):
        super().__init__()
        self.num_attention_layers = num_attention_layers
        self.use_spatial_attention = use_spatial_attention

        # Learnable weights for each attention layer
        self.attention_weights = nn.Parameter(torch.ones(num_attention_layers) / num_attention_layers)

        # Fusion network - input channels should match num_attention_layers
        self.fusion = nn.Sequential(
            nn.Conv2d(num_attention_layers, out_channels // 2, 3, padding=1),
            # This expects num_attention_layers input channels
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Optional: Spatial attention mechanism
        if use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, attention_maps, target_size):
        """
        Args:
            attention_maps: list of [B, 1, H_i, W_i] attention maps
            target_size: (H, W) tuple for output size
        Returns:
            fused_attention: [B, out_channels, H, W]
        """
        # CRITICAL: Verify we have the right number of attention maps
        if len(attention_maps) != self.num_attention_layers:
            raise ValueError(
                f"Expected {self.num_attention_layers} attention maps, "
                f"but got {len(attention_maps)}. "
                f"Attention map shapes: {[a.shape for a in attention_maps]}"
            )

        # Normalize weights to sum to 1
        normalized_weights = F.softmax(self.attention_weights, dim=0)

        # Resize all attention maps to target size and apply learned weights
        weighted_attentions = []
        for i, (attn_map, weight) in enumerate(zip(attention_maps, normalized_weights)):
            # Each attn_map should be [B, 1, H, W]
            if attn_map.shape[2:] != target_size:
                attn_map = F.interpolate(attn_map, size=target_size, mode='bilinear', align_corners=False)
            # Apply weight but keep the channel dimension
            weighted_attentions.append(attn_map * weight)  # Still [B, 1, H, W]

        # Stack along channel dimension: [B, num_layers * 1, H, W] = [B, num_layers, H, W]
        attention_stack = torch.cat(weighted_attentions, dim=1)

        # Verify the shape before passing to fusion
        assert attention_stack.shape[1] == self.num_attention_layers, \
            f"Expected {self.num_attention_layers} channels but got {attention_stack.shape[1]}"

        fused = self.fusion(attention_stack)  # [B, out_channels, H, W]

        # Apply spatial attention if enabled
        if self.use_spatial_attention:
            spatial_weight = self.spatial_attention(fused)
            fused = fused * spatial_weight

        return fused

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
class HerdNetDINOv2FPN(nn.Module):
    def __init__(
            self,
            backbone='vit_large_patch14_dinov2.lvd142m',
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: Optional[int] = 2,
            head_conv: int = 64,
            pretrained_path=None,
            debug=True,
            attention_layers: List[int] = [-4, -3, -2, -1],
            output_channels=[256, 512, 1024],
            fpn_out_channels=256,  # NEW: FPN output channels
            use_fpn=True,  # NEW: Toggle FPN
            use_enhanced_attention=True,  # NEW: Toggle enhanced attention
            input_resolution=(512, 512)
    ):
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], f"Invalid down_ratio: {down_ratio}"

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.attention_layers = attention_layers

        # Load DINOv2 model from timm
        dinov2_model = timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        if pretrained_path:
            dinov2_model = _load_backbone_checkpoint(dinov2_model, pretrained_path)

        # self.backbone = dinov2_model

        # Extract model info
        self.patch_size = dinov2_model.patch_embed.patch_size[0]
        self.embed_dim = dinov2_model.embed_dim  # 1024 for large model

        if debug:
            logger.info(f"  Patch size: {self.patch_size}x{self.patch_size}")
            logger.info(f"  Embedding dim: {self.embed_dim}")
            logger.info(f"  Attention extraction layers: {attention_layers}")

        # Attention extractor
        self.attention_extractor = DINOv2AttentionExtractor(dinov2_model, attention_layers)
        self.use_hook_attention = True
        attention_channels = len(attention_layers)
        if debug:
            logger.info("Using hook-based attention extraction")


        # Spatial processor for multi-scale features
        # [256, 512, 1024]
        logger.warning(f"  Output channels: {output_channels}")

        # Spatial processor for multi-scale features
        self.spatial_processor = DINOv2SpatialProcessor(
            feature_dim=self.embed_dim,
            output_channels=output_channels
        )
        # TOO use the IDA and HDA methods
        # NEW: FPN Module (replaces or augments DINOv2FeatureUpsampler)
        self.use_fpn = use_fpn
        if use_fpn:
            self.fpn = FPNModule(
                in_channels_list=output_channels,
                out_channels=fpn_out_channels
            )
            feature_channels = fpn_out_channels
        else:
            # Original upsampler
            self.feature_upsampler = DINOv2FeatureUpsampler(
                in_channels=output_channels,
                out_channels=output_channels[0]
            )
            feature_channels = output_channels[0]

        # NEW: Enhanced Attention Fusion
        self.use_enhanced_attention = use_enhanced_attention
        if use_enhanced_attention:
            self.attention_fusion = EnhancedAttentionFusion(
                num_attention_layers=len(attention_layers),
                out_channels=feature_channels,
                use_spatial_attention=True
            )

        # Bottleneck conv (adjusted input channels)
        self.bottleneck_conv = nn.Conv2d(
            feature_channels * 2 if use_enhanced_attention else feature_channels,  # *2 if concatenating attention
            feature_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Localization head
        self.loc_head = nn.Sequential(
            nn.Conv2d(feature_channels, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(0.0)

        # Classification head (operates on coarsest FPN level or last multi-scale feature)
        cls_input_channels = fpn_out_channels if use_fpn else output_channels[-1]
        self.cls_head = nn.Sequential(
            nn.Conv2d(cls_input_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(0.00)

        if debug:
            self._inspect_model()

    def freeze_backbone_completely(self):
        """Freeze all parameters in the DINOv2 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def check_trainable_parameters(self):
        """Check which parameters are trainable."""
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        # Check specifically DINOv2 parameters
        dinov2_total = 0
        dinov2_trainable = 0

        for name, param in self.attention_extractor.dinov2.named_parameters():
            dinov2_total += param.numel()
            if param.requires_grad:
                dinov2_trainable += param.numel()

        logger.info(f"DINOv2 total parameters: {dinov2_total:,}")
        logger.info(f"DINOv2 trainable parameters: {dinov2_trainable:,}")
        logger.info(f"DINOv2 percentage trainable: {100 * dinov2_trainable / dinov2_total:.2f}%")

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dinov2_total': dinov2_total,
            'dinov2_trainable': dinov2_trainable
        }

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
        # Resize input if needed
        if x.shape[2:] != (self.patch_size * 37, self.patch_size * 37):
            x = F.interpolate(
                x,
                size=(self.patch_size * 37, self.patch_size * 37),
                mode='bilinear',
                align_corners=False
            )

        # Extract DINOv2 features and attention maps
        patch_features, attention_maps = self.attention_extractor(x)

        # Process into multi-scale features and attention heatmaps
        multi_scale_features, attention_heatmaps = self.spatial_processor(patch_features, attention_maps)
        # multi_scale_features: list of [B, C_i, H_i, W_i] NOW from COARSE to FINE
        # attention_heatmaps: list of [B, 1, H_i, W_i] NOW from COARSE to FINE

        # Apply FPN or original upsampler
        if self.use_fpn:
            # NO NEED TO REVERSE - already in correct order (coarse to fine)
            fpn_features = self.fpn(multi_scale_features)
            # fpn_features: list of [B, fpn_out_channels, H_i, W_i]

            # Use finest FPN level for localization
            finest_features = fpn_features[-1]  # Finest scale (last element)

            # Use coarsest FPN level for classification
            coarsest_features = fpn_features[0]  # Coarsest scale (first element)
        else:
            # Original approach - need to reverse back to fine->coarse for upsampler
            fused_features = self.feature_upsampler(multi_scale_features[::-1], attention_heatmaps[::-1])
            finest_features = fused_features
            coarsest_features = multi_scale_features[0]  # First is coarsest now

        # Enhanced attention fusion
        if self.use_enhanced_attention:
            target_size = finest_features.shape[2:]
            attention_features = self.attention_fusion(attention_heatmaps, target_size)

            # Concatenate or add attention features
            combined_features = torch.cat([finest_features, attention_features], dim=1)  # Concatenate
        else:
            combined_features = finest_features

        # Bottleneck
        bottleneck_features = self.bottleneck_conv(combined_features)

        # Localization heatmap
        heatmap = self.loc_head(bottleneck_features)  # [B, 1, H, W]

        # Classification from coarsest features
        cls_out = self.cls_head(coarsest_features)
        cls_out_16x16 = F.interpolate(cls_out, size=(16, 16), mode='bilinear', align_corners=False)

        # Upscale heatmap to final size
        heatmap_upscaled = F.interpolate(heatmap, size=(128, 128), mode='bilinear', align_corners=False)

        return heatmap_upscaled, cls_out_16x16