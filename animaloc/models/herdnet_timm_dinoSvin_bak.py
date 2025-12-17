from typing import Optional, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .register import MODELS


class SimpleDINOv2Extractor(nn.Module):
    """Simplified DINOv2 feature extractor without attention hooks."""

    def __init__(self, dinov2_model):
        super().__init__()
        self.dinov2 = dinov2_model
        self.num_prefix_tokens = getattr(dinov2_model, 'num_prefix_tokens', 1)

    def forward(self, x):
        features = self.dinov2.forward_features(x)  # [B, N, D]

        # Remove prefix tokens (CLS + registers for reg4 models)
        features = features[:, self.num_prefix_tokens:]  # [B, N-prefix, D]

        B, N, D = features.shape
        H = W = int(N ** 0.5)

        # Feature-based attention (simple but effective)
        feature_attention = torch.norm(features, dim=2)  # [B, N]
        feature_attention = (feature_attention - feature_attention.min(dim=1, keepdim=True)[0]) / \
                            (feature_attention.max(dim=1, keepdim=True)[0] -
                             feature_attention.min(dim=1, keepdim=True)[0] + 1e-8)

        attention_maps = {0: feature_attention}

        return features, attention_maps


class DINOv2AttentionExtractor(nn.Module):
    """Extract spatial attention maps from DINOv2 transformer blocks."""

    def __init__(self, dinov2_model, layer_indices: List[int] = [-4, -3, -2, -1]):
        super().__init__()
        self.dinov2 = dinov2_model
        self.layer_indices = layer_indices
        self.num_prefix_tokens = getattr(dinov2_model, 'num_prefix_tokens', 1)
        self.attention_maps = {}
        self.hooks = []

        for i, layer_idx in enumerate(layer_indices):
            target_layer = self.dinov2.blocks[layer_idx].attn
            hook = target_layer.register_forward_hook(
                lambda module, input, output, idx=i: self._save_attention(module, input, output, idx)
            )
            self.hooks.append(hook)

    def _save_attention(self, module, input, output, layer_idx):
        """Extract attention weights from the attention module."""
        x = input[0]
        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_weights = (q @ k.transpose(-2, -1)) * module.scale
        attn_weights = attn_weights.softmax(dim=-1)

        # Skip all prefix tokens (CLS + registers)
        if N > self.num_prefix_tokens:
            cls_attention = attn_weights[:, :, 0, self.num_prefix_tokens:].mean(dim=1)
            self.attention_maps[layer_idx] = cls_attention.detach()

    def forward(self, x):
        self.attention_maps.clear()

        patch_features = self.dinov2.forward_features(x)  # [B, N, D]

        # Remove prefix tokens (CLS + registers)
        patch_features = patch_features[:, self.num_prefix_tokens:]

        return patch_features, self.attention_maps

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class DINOv2SpatialProcessor(nn.Module):
    def __init__(
            self,
            feature_dim: int = 1024,
            output_channels: List[int] = [256, 512, 1024],
            num_attention_layers: int = 4,  # Add this
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_channels = output_channels
        self.num_attention_layers = num_attention_layers

        # ... existing projectors and convs ...

        # Fuse all attention layers into one per scale
        self.attention_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_attention_layers, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in output_channels
        ])

    def forward(self, patch_features, attention_maps):
        B, N, D = patch_features.shape
        H = W = int(N ** 0.5)

        # Normalize features
        patch_features = F.layer_norm(patch_features, [D])

        # Stack all attention maps [B, num_layers, H, W]
        if attention_maps:
            attn_stack = torch.stack(
                [attention_maps[i].reshape(B, 1, H, W) for i in sorted(attention_maps.keys())],
                dim=1
            ).squeeze(2)  # [B, num_layers, H, W]
        else:
            attn_stack = torch.ones(B, self.num_attention_layers, H, W,
                                    device=patch_features.device) * 0.5

        multi_scale_features = []
        attention_heatmaps = []

        for i, (projector, conv, attn_fuse) in enumerate(
                zip(self.scale_projectors, self.spatial_convs, self.attention_fusion)
        ):
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

            # Fuse all attention layers for this scale
            attn_resized = F.interpolate(attn_stack, size=scale_feat.shape[2:],
                                         mode='bilinear', align_corners=False)
            fused_attn = attn_fuse(attn_resized)
            attention_heatmaps.append(fused_attn)

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
            backbone='vit_large_patch14_dinov2.lvd142m',
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: Optional[int] = 2,
            head_conv: int = 64,
            pretrained_path=None,
            debug=True,
            attention_layers: List[int] = [-4, -3, -2, -1], # Which transformer layers to extract attention from
            output_channels=[256, 512, 1024],
            input_resolution=(512, 512),
            freeze_backbone=False
    ):
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], f"Invalid down_ratio: {down_ratio}"

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.attention_layers = attention_layers

        # Load DINOv2 model from timm
        self.backbone = timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )


        if pretrained_path:
            self.backbone = _load_backbone_checkpoint(self.backbone, pretrained_path)
        if freeze_backbone:
            self.freeze_backbone_completely(self.backbone)
        # self.backbone = dinov2_model

        # Extract model info
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.embed_dim = self.backbone.embed_dim  # 1024 for large model

        if debug:
            logger.info(f"  Patch size: {self.patch_size}x{self.patch_size}")
            logger.info(f"  Embedding dim: {self.embed_dim}")
            logger.info(f"  Attention extraction layers: {attention_layers}")

        # Attention extractor
        try:
            self.attention_extractor = DINOv2AttentionExtractor(self.backbone, attention_layers)
            self.use_hook_attention = True
            attention_channels = len(attention_layers)
            if debug:
                logger.info("Using hook-based attention extraction")
        except Exception as e:
            if debug:
                logger.error(f"Hook-based attention failed ({e}), using simple feature-based attention")
            self.attention_extractor = SimpleDINOv2Extractor(self.backbone)
            self.use_hook_attention = False
            attention_channels = 1

        # Spatial processor for multi-scale features
         # [256, 512, 1024]
        logger.warning(f"  Output channels: {output_channels}")

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
            nn.Dropout2d(0.2),  # Add dropout

            nn.Conv2d(head_conv, 1, kernel_size=1),
            # nn.Sigmoid()
        )
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)  # Learnable temperature

        self.loc_head[-1].bias.data.fill_(0.0)

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

        self._init_weights()

        if debug:
            self._inspect_model()

    def _init_weights(self):
        """Initialize head weights properly."""
        for m in [self.loc_head, self.cls_head, self.attention_head, self.feature_attention]:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize final layers with small weights for stable start
        nn.init.normal_(self.loc_head[-1].weight, std=0.01)
        nn.init.constant_(self.loc_head[-1].bias, -2.0)  # Start with low confidence (sigmoid(-2) ≈ 0.12)

        nn.init.normal_(self.cls_head[-1].weight, std=0.01)
        nn.init.zeros_(self.cls_head[-1].bias)

    def freeze_backbone_completely(self, model):
        """Freeze all parameters in the DINOv2 backbone."""
        for param in model.parameters():
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
        # Extract DINOv2 features and attention maps

        # Interpolate input to match expected resolution
        if x.shape[2:] != (self.patch_size * 37, self.patch_size * 37):
            # logger.info(f"Input shape {x.shape[2:]} does not match expected patch size {self.patch_size * 37}, resizing.")
            x = F.interpolate(
                x,
                size=(self.patch_size * 37, self.patch_size * 37),
                mode='bilinear',
                align_corners=False
            )

        patch_features, attention_maps = self.attention_extractor(x)

        # Process into multi-scale features and attention heatmaps
        multi_scale_features, attention_heatmaps = self.spatial_processor(patch_features, attention_maps)

        # Fuse features with attention guidance
        fused_features = self.feature_upsampler(multi_scale_features, attention_heatmaps)

        # Localization heatmap from fused features
        bottleneck_features = self.bottleneck_conv(fused_features)
        heatmap_logits = self.loc_head(bottleneck_features)
        heatmap = torch.sigmoid(heatmap_logits / self.temperature)

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
                                    (feature_magnitude.max(dim=1, keepdim=True)[0] -
                                     feature_magnitude.min(dim=1, keepdim=True)[0] + 1e-8)

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
