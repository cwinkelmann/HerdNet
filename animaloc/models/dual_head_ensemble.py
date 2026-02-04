"""
DualBranchHerdNet - Combining ConvNeXt and DINOv2 for Camouflaged Object Detection

Architecture:
- Branch 1: CamouflageHerdNetConvNeXt (texture/edge-aware, full resolution)
- Branch 2: HerdNetDINOv2 (semantic features, attention-guided)

Fusion strategies:
1. HEATMAP_FUSION: Late fusion of detection heatmaps
2. FEATURE_FUSION: Early fusion before detection heads
3. MULTI_LEVEL_FUSION: Both feature and heatmap fusion

Why this combination works:
- ConvNeXt: Excellent at local texture/edge patterns (Gabor + edge modules)
- DINOv2: Strong semantic understanding + attention-guided saliency
- Together: Texture discrimination + semantic awareness = better camouflage detection

Author: Christian (Dual-branch architecture)
Date: 2025-12-16
"""

from typing import Optional, List, Dict, Tuple, Literal
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .herdnet_timm_convnext_camouflaged import CamouflageHerdNetConvNeXt
from .herdnet_timm_dinoSvin import HerdNetDINOv2
# Import the original models - adjust import paths for your project structure

from .register import MODELS


class FusionStrategy(Enum):
    """Available fusion strategies."""
    HEATMAP = "heatmap"           # Late fusion: fuse final heatmaps
    FEATURE = "feature"           # Early fusion: fuse features before heads
    MULTI_LEVEL = "multi_level"   # Both: fuse at multiple levels


# =============================================================================
# Checkpoint Loading Utilities
# =============================================================================

def _load_backbone_checkpoint(model, pretrained_path: str):
    """
    Load checkpoint with proper error handling.

    Follows the same pattern as HerdNetTimmDLA and HerdNetDINOv2.

    Args:
        model: The model/backbone to load weights into
        pretrained_path: Path to checkpoint file

    Returns:
        Model with loaded weights
    """
    logger.info(f"  Loading checkpoint from: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Clean up state dict keys (remove "module." prefix from DDP)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"  Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {len(unexpected)}")

    return model


def _load_branch_checkpoint(branch: nn.Module, pretrained_path: str, branch_name: str):
    """
    Load a checkpoint for a complete branch (full HerdNet model).

    This handles loading a full branch checkpoint (e.g., a trained
    CamouflageHerdNetConvNeXt or HerdNetDINOv2 checkpoint).

    Args:
        branch: The branch model (CamouflageHerdNetConvNeXt or HerdNetDINOv2)
        pretrained_path: Path to checkpoint
        branch_name: Name for logging ('convnext' or 'dinov2')

    Returns:
        Branch with loaded weights
    """
    logger.info(f"  Loading {branch_name} branch checkpoint from: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Clean up keys
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Try to load the full branch
    missing, unexpected = branch.load_state_dict(state_dict, strict=False)

    loaded_keys = len(state_dict) - len(unexpected)
    logger.info(f"  Loaded {loaded_keys}/{len(state_dict)} keys for {branch_name} branch")

    if missing:
        logger.warning(f"  Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for k in missing:
                logger.debug(f"    - {k}")
    if unexpected:
        logger.info(f"  Unexpected keys (ignored): {len(unexpected)}")
        if len(unexpected) <= 10:
            for k in unexpected:
                logger.debug(f"    - {k}")

    return branch


# =============================================================================
# Fusion Modules
# =============================================================================

class HeatmapFusion(nn.Module):
    """
    Late fusion of heatmaps from both branches.

    Options:
    - learned: Learnable weighted combination
    - attention: Spatial attention-based fusion
    - max: Element-wise maximum (union)
    - mean: Simple average
    """

    def __init__(
            self,
            fusion_type: Literal["learned", "attention", "max", "mean"] = "learned"
    ):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "learned":
            # Learnable weights for each branch
            self.branch_weights = nn.Parameter(torch.ones(2) * 0.5)
            # Refinement after fusion
            self.refine = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Sigmoid()
            )

        elif fusion_type == "attention":
            # Cross-attention between heatmaps
            self.query_conv = nn.Conv2d(1, 8, kernel_size=1)
            self.key_conv = nn.Conv2d(1, 8, kernel_size=1)
            self.value_conv = nn.Conv2d(1, 8, kernel_size=1)
            self.output_conv = nn.Conv2d(8, 1, kernel_size=1)
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
            self,
            heatmap_convnext: torch.Tensor,
            heatmap_dino: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse heatmaps from both branches.

        Args:
            heatmap_convnext: [B, 1, H1, W1] from ConvNeXt branch
            heatmap_dino: [B, 1, H2, W2] from DINOv2 branch

        Returns:
            Fused heatmap [B, 1, H, W]
        """
        # Align resolutions (use larger resolution)
        target_size = max(heatmap_convnext.shape[2], heatmap_dino.shape[2])
        target_size = (target_size, target_size)

        if heatmap_convnext.shape[2:] != target_size:
            heatmap_convnext = F.interpolate(
                heatmap_convnext, size=target_size,
                mode='bilinear', align_corners=False
            )
        if heatmap_dino.shape[2:] != target_size:
            heatmap_dino = F.interpolate(
                heatmap_dino, size=target_size,
                mode='bilinear', align_corners=False
            )

        if self.fusion_type == "learned":
            weights = F.softmax(self.branch_weights, dim=0)
            combined = torch.cat([
                heatmap_convnext * weights[0],
                heatmap_dino * weights[1]
            ], dim=1)
            fused = self.refine(combined)

        elif self.fusion_type == "attention":
            B, C, H, W = heatmap_convnext.shape

            # Query from ConvNeXt, Key/Value from DINOv2
            q = self.query_conv(heatmap_convnext).view(B, -1, H * W)
            k = self.key_conv(heatmap_dino).view(B, -1, H * W)
            v = self.value_conv(heatmap_dino).view(B, -1, H * W)

            # Attention
            attn = torch.bmm(q.transpose(1, 2), k)  # [B, HW, HW]
            attn = F.softmax(attn / (8 ** 0.5), dim=-1)

            out = torch.bmm(v, attn.transpose(1, 2))  # [B, C, HW]
            out = out.view(B, -1, H, W)
            out = self.output_conv(out)

            # Residual connection
            fused = heatmap_convnext + self.gamma * out
            fused = torch.sigmoid(fused)  # Ensure [0, 1]

        elif self.fusion_type == "max":
            fused = torch.max(heatmap_convnext, heatmap_dino)

        elif self.fusion_type == "mean":
            fused = (heatmap_convnext + heatmap_dino) / 2

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return fused


class FeatureFusion(nn.Module):
    """
    Early fusion of features before detection heads.

    Fuses 256-channel features from both branches into a unified representation.
    """

    def __init__(
            self,
            convnext_channels: int = 256,
            dino_channels: int = 256,
            output_channels: int = 256,
            fusion_type: Literal["concat", "attention", "gated"] = "gated"
    ):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            # Simple concatenation + projection
            self.fuse = nn.Sequential(
                nn.Conv2d(convnext_channels + dino_channels, output_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
            )

        elif fusion_type == "attention":
            # Cross-attention fusion
            self.convnext_proj = nn.Conv2d(convnext_channels, output_channels, 1)
            self.dino_proj = nn.Conv2d(dino_channels, output_channels, 1)

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=output_channels,
                num_heads=8,
                batch_first=True
            )

            self.ffn = nn.Sequential(
                nn.Linear(output_channels, output_channels * 4),
                nn.GELU(),
                nn.Linear(output_channels * 4, output_channels),
            )
            self.norm1 = nn.LayerNorm(output_channels)
            self.norm2 = nn.LayerNorm(output_channels)

        elif fusion_type == "gated":
            # Gated fusion - learns what to take from each branch
            self.convnext_proj = nn.Conv2d(convnext_channels, output_channels, 1, bias=False)
            self.dino_proj = nn.Conv2d(dino_channels, output_channels, 1, bias=False)

            # Gate computation
            self.gate = nn.Sequential(
                nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels, kernel_size=1),
                nn.Sigmoid()
            )

            # Refinement
            self.refine = nn.Sequential(
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
            )

    def forward(
            self,
            feat_convnext: torch.Tensor,
            feat_dino: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features from both branches.

        Args:
            feat_convnext: [B, C1, H1, W1] from ConvNeXt FPN
            feat_dino: [B, C2, H2, W2] from DINOv2 upsampler

        Returns:
            Fused features [B, C_out, H, W]
        """
        # Align spatial dimensions
        target_size = feat_convnext.shape[2:]  # Use ConvNeXt resolution

        if feat_dino.shape[2:] != target_size:
            feat_dino = F.interpolate(
                feat_dino, size=target_size,
                mode='bilinear', align_corners=False
            )

        if self.fusion_type == "concat":
            combined = torch.cat([feat_convnext, feat_dino], dim=1)
            fused = self.fuse(combined)

        elif self.fusion_type == "attention":
            B, C, H, W = feat_convnext.shape

            # Project to same channels
            conv_proj = self.convnext_proj(feat_convnext)  # [B, C, H, W]
            dino_proj = self.dino_proj(feat_dino)  # [B, C, H, W]

            # Reshape for attention: [B, HW, C]
            conv_flat = conv_proj.flatten(2).transpose(1, 2)
            dino_flat = dino_proj.flatten(2).transpose(1, 2)

            # Cross attention: ConvNeXt attends to DINOv2
            attn_out, _ = self.cross_attn(conv_flat, dino_flat, dino_flat)
            attn_out = self.norm1(conv_flat + attn_out)

            # FFN
            ffn_out = self.ffn(attn_out)
            fused_flat = self.norm2(attn_out + ffn_out)

            # Reshape back
            fused = fused_flat.transpose(1, 2).view(B, -1, H, W)

        elif self.fusion_type == "gated":
            # Project both to same channels
            conv_proj = self.convnext_proj(feat_convnext)
            dino_proj = self.dino_proj(feat_dino)

            # Compute gate
            combined = torch.cat([conv_proj, dino_proj], dim=1)
            gate = self.gate(combined)

            # Gated fusion
            fused = gate * conv_proj + (1 - gate) * dino_proj
            fused = self.refine(fused)

        return fused


class SharedDetectionHead(nn.Module):
    """
    Shared detection head for fused features.

    Similar architecture to CamouflageDetectionHead but works on fused features.
    """

    def __init__(
            self,
            in_channels: int = 256,
            hidden_channels: int = 128,
    ):
        super().__init__()

        # Multi-scale processing (preserved from original)
        self.fine_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        self.medium_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        self.coarse_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # Output
        self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fine = self.fine_branch(x)
        medium = self.medium_branch(x)
        coarse = self.coarse_branch(x)

        weights = F.softmax(self.scale_weights, dim=0)
        fine = fine * weights[0]
        medium = medium * weights[1]
        coarse = coarse * weights[2]

        multi_scale = torch.cat([fine, medium, coarse], dim=1)
        fused = self.fusion(multi_scale)

        logits = self.output(fused)
        heatmap = torch.sigmoid(logits / self.temperature.clamp(min=0.1))

        return heatmap


# =============================================================================
# Main Model - DualBranchHerdNet
# =============================================================================

@MODELS.register()
class DualBranchHerdNet(nn.Module):
    """
    Dual-branch HerdNet combining ConvNeXt and DINOv2 for camouflaged object detection.

    This model uses BOTH branches intact and adds fusion modules on top.
    No reimplementation of original model components.

    Args:
        fusion_strategy: How to combine branches ('heatmap', 'feature', 'multi_level')
        convnext_config: Config dict for CamouflageHerdNetConvNeXt
        dinov2_config: Config dict for HerdNetDINOv2
        convnext_pretrained_path: Path to ConvNeXt branch checkpoint (full branch or backbone)
        dinov2_pretrained_path: Path to DINOv2 branch checkpoint (full branch or backbone)
        heatmap_fusion_type: Type of heatmap fusion ('learned', 'attention', 'max', 'mean')
        feature_fusion_type: Type of feature fusion ('concat', 'attention', 'gated')
        freeze_branches: Whether to freeze pre-trained branch weights initially
        num_classes: Number of output classes
        debug: Print debug information
    """

    def __init__(
            self,
            fusion_strategy: str = "multi_level",
            convnext_config: Optional[Dict] = None,
            dinov2_config: Optional[Dict] = None,
            convnext_pretrained_path: Optional[str] = None,
            dinov2_pretrained_path: Optional[str] = None,
            heatmap_fusion_type: str = "learned",
            feature_fusion_type: str = "gated",
            freeze_branches: bool = False,
            num_classes: int = 2,
            debug: bool = True,
            down_ratio = None
    ):
        super().__init__()

        self.fusion_strategy = FusionStrategy(fusion_strategy)
        self.num_classes = num_classes

        if debug:
            logger.info("\n" + "=" * 60)
            logger.info("Initializing DualBranchHerdNet")
            logger.info("=" * 60)
            logger.info(f"  Fusion strategy: {fusion_strategy}")

        # =================================================================
        # Default configs for branches
        # =================================================================

        # ConvNeXt defaults - note: pretrained_path handled separately
        _convnext_defaults = {
            'backbone_size': 'tiny',
            'num_classes': num_classes,
            'img_size': 512,
            'pretrained': True,
            'freeze_backbone': freeze_branches,
            'fpn_channels': 256,
            'use_gabor': True,
            'use_edge_enhancement': True,
            'use_multi_res': False,
            'debug': debug,
        }

        # DINOv2 defaults - note: pretrained_path handled separately
        _dinov2_defaults = {
            'backbone': 'vit_large_patch14_dinov2.lvd142m',
            'num_classes': num_classes,
            'pretrained': True,
            'down_ratio': 2,
            'head_conv': 64,
            'attention_layers': [-4, -3, -2, -1],
            'output_channels': [256, 512, 1024],
            'freeze_backbone': freeze_branches,
            'debug': debug,
        }

        # Merge user configs with defaults
        convnext_config = {**_convnext_defaults, **(convnext_config or {})}
        dinov2_config = {**_dinov2_defaults, **(dinov2_config or {})}

        # =================================================================
        # Branch 1: CamouflageHerdNetConvNeXt
        # =================================================================
        if debug:
            logger.info("\n--- Branch 1: ConvNeXt ---")

        self.convnext_branch = CamouflageHerdNetConvNeXt(**convnext_config)

        # Load ConvNeXt checkpoint if provided
        if convnext_pretrained_path:
            self.convnext_branch = _load_branch_checkpoint(
                self.convnext_branch,
                convnext_pretrained_path,
                "ConvNeXt"
            )

        # =================================================================
        # Branch 2: HerdNetDINOv2
        # =================================================================
        if debug:
            logger.info("\n--- Branch 2: DINOv2 ---")

        self.dinov2_branch = HerdNetDINOv2(**dinov2_config)

        # Load DINOv2 checkpoint if provided
        if dinov2_pretrained_path:
            self.dinov2_branch = _load_branch_checkpoint(
                self.dinov2_branch,
                dinov2_pretrained_path,
                "DINOv2"
            )

        # =================================================================
        # Fusion Modules
        # =================================================================
        if debug:
            logger.info("\n--- Fusion Modules ---")

        # Feature fusion (for FEATURE and MULTI_LEVEL strategies)
        if self.fusion_strategy in [FusionStrategy.FEATURE, FusionStrategy.MULTI_LEVEL]:
            self.feature_fusion = FeatureFusion(
                convnext_channels=256,  # FPN output channels
                dino_channels=256,      # Feature upsampler output
                output_channels=256,
                fusion_type=feature_fusion_type
            )

            # Shared detection head for fused features
            self.shared_detection_head = SharedDetectionHead(
                in_channels=256,
                hidden_channels=128,
            )

            if debug:
                logger.info(f"  Feature fusion: {feature_fusion_type}")

        # Heatmap fusion (for HEATMAP and MULTI_LEVEL strategies)
        if self.fusion_strategy in [FusionStrategy.HEATMAP, FusionStrategy.MULTI_LEVEL]:
            self.heatmap_fusion = HeatmapFusion(fusion_type=heatmap_fusion_type)

            if debug:
                logger.info(f"  Heatmap fusion: {heatmap_fusion_type}")

        # Multi-level combination weights
        if self.fusion_strategy == FusionStrategy.MULTI_LEVEL:
            self.level_weights = nn.Parameter(torch.ones(2) * 0.5)
            if debug:
                logger.info("  Multi-level: learnable combination weights")

        # Shared classification head
        self.cls_fusion = nn.Sequential(
            nn.Conv2d(num_classes * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        if debug:
            self._log_parameters()

    def _log_parameters(self):
        """Log parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        convnext_params = sum(p.numel() for p in self.convnext_branch.parameters())
        dino_params = sum(p.numel() for p in self.dinov2_branch.parameters())
        fusion_params = total - convnext_params - dino_params

        logger.info("\n--- Parameter Summary ---")
        logger.info(f"  Total: {total:,}")
        logger.info(f"  Trainable: {trainable:,} ({100 * trainable / total:.1f}%)")
        logger.info(f"  ConvNeXt branch: {convnext_params:,}")
        logger.info(f"  DINOv2 branch: {dino_params:,}")
        logger.info(f"  Fusion modules: {fusion_params:,}")

    def check_trainable_parameters(self) -> Dict[str, int]:
        """Check and log trainable parameters per component."""
        stats = {}

        # Total
        stats['total'] = sum(p.numel() for p in self.parameters())
        stats['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # ConvNeXt branch
        stats['convnext_total'] = sum(p.numel() for p in self.convnext_branch.parameters())
        stats['convnext_trainable'] = sum(p.numel() for p in self.convnext_branch.parameters() if p.requires_grad)
        stats['convnext_backbone_trainable'] = sum(p.numel() for p in self.convnext_branch.backbone.parameters() if p.requires_grad)

        # DINOv2 branch
        stats['dinov2_total'] = sum(p.numel() for p in self.dinov2_branch.parameters())
        stats['dinov2_trainable'] = sum(p.numel() for p in self.dinov2_branch.parameters() if p.requires_grad)
        stats['dinov2_backbone_trainable'] = sum(p.numel() for p in self.dinov2_branch.backbone.parameters() if p.requires_grad)

        # Fusion
        fusion_params = []
        for name, param in self.named_parameters():
            if 'convnext_branch' not in name and 'dinov2_branch' not in name:
                fusion_params.append(param)
        stats['fusion_total'] = sum(p.numel() for p in fusion_params)
        stats['fusion_trainable'] = sum(p.numel() for p in fusion_params if p.requires_grad)

        logger.info("\n--- Detailed Parameter Summary ---")
        logger.info(f"  Total: {stats['total']:,} ({stats['trainable']:,} trainable)")
        logger.info(f"  ConvNeXt: {stats['convnext_total']:,} ({stats['convnext_trainable']:,} trainable, backbone: {stats['convnext_backbone_trainable']:,})")
        logger.info(f"  DINOv2: {stats['dinov2_total']:,} ({stats['dinov2_trainable']:,} trainable, backbone: {stats['dinov2_backbone_trainable']:,})")
        logger.info(f"  Fusion: {stats['fusion_total']:,} ({stats['fusion_trainable']:,} trainable)")

        return stats

    def forward(
            self,
            x: torch.Tensor,
            debug: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Dict:
        """
        Forward pass through both branches with fusion.

        Args:
            x: Input tensor [B, 3, H, W]
            debug: Return debug information

        Returns:
            If debug=False:
                heatmap: Fused detection heatmap [B, 1, H/4, W/4]
                classification: Fused classification [B, num_classes, H', W']
            If debug=True:
                Dictionary with all intermediate outputs
        """
        original_size = x.shape[2:]

        # =================================================================
        # Run both branches (with debug to get intermediate features)
        # =================================================================
        convnext_out = self.convnext_branch(x, debug=True)
        dinov2_out = self.dinov2_branch(x, debug=True)

        # Extract outputs
        heatmap_convnext = convnext_out['prediction']
        heatmap_dino = dinov2_out['prediction']

        cls_convnext = convnext_out['classification']
        cls_dino = dinov2_out['classification']

        # =================================================================
        # Fusion based on strategy
        # =================================================================

        if self.fusion_strategy == FusionStrategy.HEATMAP:
            # Late fusion: just fuse heatmaps
            fused_heatmap = self.heatmap_fusion(heatmap_convnext, heatmap_dino)

        elif self.fusion_strategy == FusionStrategy.FEATURE:
            # Early fusion: fuse features, use shared head
            # Get features before detection heads
            feat_convnext = convnext_out['fpn_features']['scale_0']  # Finest FPN scale
            feat_dino = dinov2_out['fused']  # Feature upsampler output

            fused_features = self.feature_fusion(feat_convnext, feat_dino)
            fused_heatmap = self.shared_detection_head(fused_features)

        elif self.fusion_strategy == FusionStrategy.MULTI_LEVEL:
            # Both feature and heatmap fusion

            # Feature-level fusion
            feat_convnext = convnext_out['fpn_features']['scale_0']
            feat_dino = dinov2_out['fused']

            fused_features = self.feature_fusion(feat_convnext, feat_dino)
            heatmap_from_features = self.shared_detection_head(fused_features)

            # Heatmap-level fusion
            heatmap_from_heatmaps = self.heatmap_fusion(heatmap_convnext, heatmap_dino)

            # Combine both
            weights = F.softmax(self.level_weights, dim=0)

            # Align sizes
            target_size = heatmap_from_features.shape[2:]
            if heatmap_from_heatmaps.shape[2:] != target_size:
                heatmap_from_heatmaps = F.interpolate(
                    heatmap_from_heatmaps, size=target_size,
                    mode='bilinear', align_corners=False
                )

            fused_heatmap = weights[0] * heatmap_from_features + weights[1] * heatmap_from_heatmaps

        # =================================================================
        # Classification fusion
        # =================================================================

        # Align classification outputs
        target_cls_size = cls_convnext.shape[2:]
        if cls_dino.shape[2:] != target_cls_size:
            cls_dino = F.interpolate(
                cls_dino, size=target_cls_size,
                mode='bilinear', align_corners=False
            )

        cls_combined = torch.cat([cls_convnext, cls_dino], dim=1)
        fused_cls = self.cls_fusion(cls_combined)

        # =================================================================
        # Return
        # =================================================================

        if debug:
            return {
                'prediction': fused_heatmap,
                'classification': fused_cls,
                'convnext_branch': convnext_out,
                'dinov2_branch': dinov2_out,
                'heatmap_convnext': heatmap_convnext,
                'heatmap_dino': heatmap_dino,
                'fusion_strategy': self.fusion_strategy.value,
            }

        return fused_heatmap, fused_cls

    def get_optimizer_params(
            self,
            lr_convnext_backbone: float = 1e-5,
            lr_dino_backbone: float = 1e-5,
            lr_heads: float = 1e-4,
            lr_fusion: float = 1e-4
    ) -> List[Dict]:
        """
        Get parameter groups with different learning rates.

        Recommended strategy:
        - Backbones: Very low LR (pretrained)
        - Heads: Medium LR
        - Fusion modules: Medium-high LR (new parameters)
        """
        param_groups = []

        # ConvNeXt backbone
        param_groups.append({
            'params': list(self.convnext_branch.backbone.parameters()),
            'lr': lr_convnext_backbone,
            'name': 'convnext_backbone'
        })

        # ConvNeXt heads (everything except backbone)
        convnext_head_params = []
        for name, param in self.convnext_branch.named_parameters():
            if 'backbone' not in name:
                convnext_head_params.append(param)
        param_groups.append({
            'params': convnext_head_params,
            'lr': lr_heads,
            'name': 'convnext_heads'
        })

        # DINOv2 backbone
        param_groups.append({
            'params': list(self.dinov2_branch.backbone.parameters()),
            'lr': lr_dino_backbone,
            'name': 'dinov2_backbone'
        })

        # DINOv2 heads (everything except backbone)
        dino_head_params = []
        for name, param in self.dinov2_branch.named_parameters():
            if 'backbone' not in name:
                dino_head_params.append(param)
        param_groups.append({
            'params': dino_head_params,
            'lr': lr_heads,
            'name': 'dinov2_heads'
        })

        # Fusion modules (everything not in branches)
        fusion_params = []
        for name, param in self.named_parameters():
            if 'convnext_branch' not in name and 'dinov2_branch' not in name:
                fusion_params.append(param)

        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': lr_fusion,
                'name': 'fusion'
            })

        return param_groups

    def freeze_branches(self):
        """Freeze both branch backbones."""
        for param in self.convnext_branch.backbone.parameters():
            param.requires_grad = False
        for param in self.dinov2_branch.backbone.parameters():
            param.requires_grad = False
        logger.info("Both branch backbones frozen")

    def unfreeze_branches(self):
        """Unfreeze both branch backbones for fine-tuning."""
        for param in self.convnext_branch.backbone.parameters():
            param.requires_grad = True
        for param in self.dinov2_branch.backbone.parameters():
            param.requires_grad = True
        logger.info("Both branch backbones unfrozen")

    def freeze(self, layers: list) -> None:
        """Freeze all layers mentioned in the input list."""
        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = False

    def reshape_classes(self, num_classes: int) -> None:
        """Reshape architecture according to a new number of classes."""
        # Reshape both branches
        if hasattr(self.convnext_branch, 'reshape_classes'):
            self.convnext_branch.reshape_classes(num_classes)
        self.dinov2_branch.reshape_classes(num_classes)

        # Reshape fusion classification head
        self.cls_fusion = nn.Sequential(
            nn.Conv2d(num_classes * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.cls_fusion[-1].bias.data.fill_(0.00)

        self.num_classes = num_classes


# =============================================================================
# Factory Functions
# =============================================================================

def create_dual_branch_herdnet(
        fusion_strategy: str = "multi_level",
        convnext_size: str = "tiny",
        dinov2_size: str = "large",
        pretrained: bool = True,
        freeze_branches: bool = True,
        num_classes: int = 2,
        img_size: int = 512,
        convnext_pretrained_path: Optional[str] = None,
        dinov2_pretrained_path: Optional[str] = None,
) -> DualBranchHerdNet:
    """
    Create a DualBranchHerdNet with sensible defaults.

    Args:
        fusion_strategy: 'heatmap', 'feature', or 'multi_level'
        convnext_size: 'tiny', 'small', or 'base'
        dinov2_size: 'small', 'base', 'large', or 'giant'
        pretrained: Use pretrained ImageNet weights for backbones
        freeze_branches: Freeze backbones initially
        num_classes: Number of output classes
        img_size: Input image size
        convnext_pretrained_path: Path to ConvNeXt branch checkpoint
        dinov2_pretrained_path: Path to DINOv2 branch checkpoint

    Returns:
        DualBranchHerdNet model
    """
    # DINOv2 model mapping
    dinov2_models = {
        'small': 'vit_small_patch14_dinov2.lvd142m',
        'base': 'vit_base_patch14_dinov2.lvd142m',
        'large': 'vit_large_patch14_dinov2.lvd142m',
        'giant': 'vit_giant_patch14_dinov2.lvd142m',
    }

    convnext_config = {
        'backbone_size': convnext_size,
        'num_classes': num_classes,
        'img_size': img_size,
        'pretrained': pretrained,
        'freeze_backbone': freeze_branches,
        'fpn_channels': 256,
        'use_gabor': True,
        'use_edge_enhancement': True,
        'use_multi_res': False,
        'debug': True,
    }

    dinov2_config = {
        'backbone': dinov2_models[dinov2_size],
        'num_classes': num_classes,
        'pretrained': pretrained,
        'down_ratio': 2,
        'head_conv': 64,
        'attention_layers': [-4, -3, -2, -1],
        'output_channels': [256, 512, 1024],
        'freeze_backbone': freeze_branches,
        'debug': True,
    }

    model = DualBranchHerdNet(
        fusion_strategy=fusion_strategy,
        convnext_config=convnext_config,
        dinov2_config=dinov2_config,
        convnext_pretrained_path=convnext_pretrained_path,
        dinov2_pretrained_path=dinov2_pretrained_path,
        num_classes=num_classes,
        freeze_branches=freeze_branches,
        debug=True,
    )

    logger.info("\n" + "=" * 60)
    logger.info("DualBranchHerdNet Summary")
    logger.info("=" * 60)
    logger.info(f"  ConvNeXt: {convnext_size} (texture + edge aware)")
    logger.info(f"  DINOv2: {dinov2_size} (semantic + attention)")
    logger.info(f"  Fusion: {fusion_strategy}")
    if convnext_pretrained_path:
        logger.info(f"  ConvNeXt checkpoint: {convnext_pretrained_path}")
    if dinov2_pretrained_path:
        logger.info(f"  DINOv2 checkpoint: {dinov2_pretrained_path}")
    logger.info("=" * 60)

    return model


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    """Test dual-branch model."""

    print("\n" + "=" * 80)
    print("TEST: DualBranchHerdNet")
    print("=" * 80)

    # Test each fusion strategy
    for strategy in ["heatmap", "feature", "multi_level"]:
        print(f"\n--- Testing {strategy} fusion ---")

        model = create_dual_branch_herdnet(
            fusion_strategy=strategy,
            convnext_size="tiny",
            dinov2_size="large",
            pretrained=False,  # For testing
            freeze_branches=False,
            # Example checkpoint paths (uncomment to test):
            # convnext_pretrained_path="/path/to/convnext_checkpoint.pth",
            # dinov2_pretrained_path="/path/to/dinov2_checkpoint.pth",
        )

        # Test forward pass
        dummy_input = torch.randn(2, 3, 512, 512)
        heatmap, classification = model(dummy_input)

        print(f"  Heatmap shape: {heatmap.shape}")
        print(f"  Classification shape: {classification.shape}")

        # Test debug mode
        debug_out = model(dummy_input, debug=True)
        print(f"  Debug keys: {list(debug_out.keys())}")

        # Test parameter stats
        model.check_trainable_parameters()

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)