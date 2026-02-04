import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from loguru import logger
from typing import List, Tuple, Optional, Dict
from .register import MODELS


# --- 1. Robust Hook-Based Extractor ---
class DINOExtractor(nn.Module):
    """
    Extracts BOTH intermediate features (for IDA) and Attention Maps (for contrast).
    Works by registering hooks on the standard backbone, bypassing timm's feature wrapper.
    """

    def __init__(self, backbone, layer_indices):
        super().__init__()
        self.backbone = backbone
        self.layer_indices = layer_indices
        self.feature_maps = {}
        self.attention_maps = {}
        self.hooks = []

        # 1. Find the blocks container
        # Try common naming conventions for ViT blocks
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
        elif hasattr(self.backbone, 'layers'):
            blocks = self.backbone.layers
        else:
            raise AttributeError("Could not find 'blocks' or 'layers' in backbone.")

        # 2. Register Hooks
        for i, layer_idx in enumerate(layer_indices):
            # Safe index handling
            if layer_idx < 0: layer_idx += len(blocks)

            block = blocks[layer_idx]

            # --- Hook for Features (Output of the Block) ---
            # We hook the block itself to get the output features
            hook_feat = block.register_forward_hook(
                lambda module, inp, out, idx=layer_idx: self._save_feature(idx, out)
            )
            self.hooks.append(hook_feat)

            # --- Hook for Attention (Internal QKV) ---
            # We look for the Attention module inside the block
            if hasattr(block, 'attn'):
                target_attn = block.attn
            elif hasattr(block, 'self_attn'):  # Some variations
                target_attn = block.self_attn
            else:
                logger.warning(f"Block {layer_idx} has no attention module. Skipping attention extraction.")
                continue

            hook_attn = target_attn.register_forward_hook(
                lambda module, inp, out, idx=layer_idx: self._save_attention(module, inp, idx)
            )
            self.hooks.append(hook_attn)

    def _save_feature(self, idx, output):
        # ViT output is [B, N, C]
        self.feature_maps[idx] = output

    def _save_attention(self, module, input, idx):
        # Re-compute attention weights to get the map
        x = input[0]  # [B, N, C]
        B, N, C = x.shape

        # Robust QKV extraction
        if hasattr(module, 'qkv'):
            # Standard ViT / DINO
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        elif hasattr(module, 'to_qkv'):
            # Some newer implementations
            qkv = module.to_qkv(x)
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            # Fallback (skip if unknown structure)
            return

        # Compute Attention Map: Softmax(Q @ K.T)
        scale = getattr(module, 'scale', (C // module.num_heads) ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)  # [B, Heads, N, N]

        # Extract CLS token attention (Index 0 attending to 1:)
        # This is the "Saliency Map"
        cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # [B, N-1]
        self.attention_maps[idx] = cls_attn

    def clear(self):
        self.feature_maps = {}
        self.attention_maps = {}


# --- 2. The IDA Node (Aggregation Logic) ---
class IDANode(nn.Module):
    def __init__(self, channels, up_factor=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, up_factor * 2, stride=up_factor, padding=up_factor // 2,
                               groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        ) if up_factor > 1 else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, shallow, deep):
        deep_up = self.up(deep)
        if deep_up.shape[2:] != shallow.shape[2:]:
            deep_up = F.interpolate(deep_up, size=shallow.shape[2:], mode='nearest')
        return self.conv(torch.cat([shallow, deep_up], dim=1))


# --- 3. IDA Adapter (Feature Pyramid Construction) ---
class DINO_IDA_Adapter(nn.Module):
    def __init__(self, in_channels_list, dim=256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU())
            for c in in_channels_list
        ])

        # Resizers: Assuming [L2, L5, L8, L11] -> [1/4, 1/8, 1/16, 1/32]
        self.resize_0 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 4, stride=4), nn.BatchNorm2d(dim), nn.ReLU())
        self.resize_1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 2, stride=2), nn.BatchNorm2d(dim), nn.ReLU())
        self.resize_2 = nn.Identity()
        self.resize_3 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.BatchNorm2d(dim), nn.ReLU())

        self.node_1 = IDANode(dim, 2)
        self.node_2 = IDANode(dim, 2)
        self.node_3 = IDANode(dim, 2)

    def forward(self, features):
        # features list: [L2, L5, L8, L11]
        projs = [p(f) for p, f in zip(self.projs, features)]

        c0 = self.resize_0(projs[0])  # 1/4
        c1 = self.resize_1(projs[1])  # 1/8
        c2 = self.resize_2(projs[2])  # 1/16
        c3 = self.resize_3(projs[3])  # 1/32

        # IDA Cascade (Deep to Shallow)
        f2 = self.node_1(c2, c3)
        f1 = self.node_2(c1, f2)
        f0 = self.node_3(c0, f1)  # 1/4 Scale output
        return f0


# --- 4. Main Model Class ---
@MODELS.register()
class HerdNetDINOv3Attn(nn.Module):
    def __init__(self, backbone='vit_base_patch16_dinov3.sat493m',
                 num_classes=2, pretrained=True, freeze_backbone=True,
                 head_conv=64, fusion_dim=256, input_resolution=(512, 512), **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_resolution = input_resolution
        self.out_indices = [2, 5, 8, 11]

        # 1. Load Backbone (Standard mode, NOT features_only)
        # We must avoid 'features_only=True' because it wraps the model and hides blocks
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        if freeze_backbone:
            for param in self.backbone.parameters(): param.requires_grad = False

        # 2. Hook Extractor (Handles both Features and Attention)
        self.extractor = DINOExtractor(self.backbone, self.out_indices)

        # 3. Get Channels dimensions
        # We assume standard ViT dimension if not easily available
        embed_dim = self.backbone.embed_dim
        in_channels = [embed_dim] * len(self.out_indices)

        # 4. Adapter & Fusion
        self.ida = DINO_IDA_Adapter(in_channels, dim=fusion_dim)

        # Fusion Conv: [IDA_Features (256) + Attention (1)] -> [256]
        self.attn_fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim + 1, fusion_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True)
        )

        # 5. Heads (2-Channel Output for Background/Foreground Separation)
        self.loc_head = nn.Sequential(
            nn.Conv2d(fusion_dim, head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, 1)  # <--- Output 2 Channels
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(fusion_dim, head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(head_conv, num_classes)
        )

    def forward(self, x, debug=False):
        if x.shape[2:] != self.input_resolution:
            x = F.interpolate(x, size=self.input_resolution, mode='bicubic', align_corners=False)

        self.extractor.clear()

        # 1. Forward Pass
        # We run the backbone. The hooks will silently populate self.extractor.feature_maps
        _ = self.backbone(x)

        # 2. Retrieve Features from Hooks
        # Reshape [B, N, C] -> [B, C, H, W]
        raw_features = []
        for idx in self.out_indices:
            feat = self.extractor.feature_maps[idx]

            # Handle CLS token if present
            # ViT usually [B, N+1, C] or [B, N, C]
            B, N, C = feat.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                # Likely has CLS token (N = H*W + 1) or Registers
                patch_feat = feat[:, -int(math.sqrt(N - 1)) ** 2:, :]
                H = W = int(math.sqrt(patch_feat.shape[1]))
            else:
                patch_feat = feat

            spatial = patch_feat.permute(0, 2, 1).reshape(B, C, H, W)
            raw_features.append(spatial)

        # 3. IDA Processing
        fused_features = self.ida(raw_features)

        # 4. Attention Injection
        attn_maps = [self.extractor.attention_maps[i] for i in self.out_indices if i in self.extractor.attention_maps]

        if attn_maps:
            # Reshape attention [B, N] -> [B, 1, H, W]
            reshaped = []
            for m in attn_maps:
                H_a = int(math.sqrt(m.shape[1]))
                reshaped.append(m.reshape(m.shape[0], 1, H_a, H_a))

            # Average attention maps
            mean_attn = torch.mean(torch.stack(reshaped), dim=0)

            # Upsample to match feature resolution
            mean_attn_up = F.interpolate(mean_attn, size=fused_features.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate
            combined = torch.cat([fused_features, mean_attn_up], dim=1)
            final_features = self.attn_fusion_conv(combined)
        else:
            final_features = fused_features

        # 5. Prediction
        logits = self.loc_head(final_features)

        # Softmax and take Foreground channel
        heatmap = torch.softmax(logits, dim=1)[:, 1:2, :, :]

        # Ensure 128x128
        if heatmap.shape[2:] != (128, 128):
            heatmap = F.interpolate(heatmap, size=(128, 128), mode='bilinear')

        cls_logits = self.cls_head(final_features)
        cls_out = cls_logits.view(cls_logits.size(0), self.num_classes, 1, 1)
        cls_out_16x16 = F.interpolate(cls_out, size=(16, 16), mode='nearest')

        return heatmap, cls_out_16x16