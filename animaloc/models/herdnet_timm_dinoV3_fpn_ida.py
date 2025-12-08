import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from loguru import logger
from typing import List, Tuple, Optional
from .register import MODELS


# --- 1. The Pyramid Builder (Fixes the Resolution Mismatch) ---
class SyntheticPyramid(nn.Module):
    """
    Converts columnar ViT features (all 1/16 scale) into a
    Hierarchical Feature Pyramid (1/32, 1/16, 1/8, 1/4).
    """

    def __init__(self, in_channels_list, dim=256):
        super().__init__()

        # We expect 4 layers: [L2, L5, L8, L11]
        assert len(in_channels_list) == 4

        # 1. Layer 11 -> 1/32 Scale (Downsample)
        self.p4 = nn.Sequential(
            nn.Conv2d(in_channels_list[3], dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # 2. Layer 8 -> 1/16 Scale (Identity / Projection)
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels_list[2], dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # 3. Layer 5 -> 1/8 Scale (Upsample)
        self.p2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels_list[1], dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # 4. Layer 2 -> 1/4 Scale (Double Upsample)
        self.p1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels_list[0], dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, features):
        # features = [L2, L5, L8, L11]
        c1 = self.p1(features[0])  # -> 1/4 (High Res)
        c2 = self.p2(features[1])  # -> 1/8
        c3 = self.p3(features[2])  # -> 1/16
        c4 = self.p4(features[3])  # -> 1/32 (Low Res)

        return [c1, c2, c3, c4]  # Returns Pyramid [1/4, 1/8, 1/16, 1/32]


# --- 2. The Fusion (Feature Pyramid Network Style) ---
class PyramidFusion(nn.Module):
    """
    Fuses the synthetic pyramid top-down (Deep -> Shallow)
    """

    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

        # Lateral connections (optional smoothing)
        self.smooth1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.smooth2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.smooth3 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, features):
        # features = [c1(1/4), c2(1/8), c3(1/16), c4(1/32)]
        c1, c2, c3, c4 = features

        # Top-down pathway
        # 1. P4 (1/32) -> upsample -> add to P3 (1/16)
        p4_up = F.interpolate(c4, scale_factor=2, mode='nearest')
        p3_fused = self.smooth3(c3 + p4_up)

        # 2. P3 (1/16) -> upsample -> add to P2 (1/8)
        p3_up = F.interpolate(p3_fused, scale_factor=2, mode='nearest')
        p2_fused = self.smooth2(c2 + p3_up)

        # 3. P2 (1/8) -> upsample -> add to P1 (1/4)
        p2_up = F.interpolate(p2_fused, scale_factor=2, mode='nearest')
        p1_fused = self.smooth1(c1 + p2_up)

        return p1_fused  # Returns the highest resolution fused map (1/4 scale)


# --- 3. The Model Class ---
@MODELS.register()
class HerdNetDINOv3Pyramid(nn.Module):
    def __init__(
            self,
            backbone='vit_base_patch16_dinov3.sat493m',
            num_classes: int = 2,
            pretrained: bool = True,
            freeze_backbone: bool = True,
            head_conv: int = 64,
            out_indices: Optional[List[int]] = None,
            fusion_dim: int = 256,
            input_resolution: Tuple[int, int] = (512, 512),
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_resolution = tuple(input_resolution)

        # 1. Determine Indices (Indices [2, 5, 8, 11] are good)
        logger.info(f"Loading backbone structure: {backbone}")
        raw_model = timm.create_model(backbone, pretrained=False, num_classes=0)
        total_blocks = len(raw_model.blocks)
        del raw_model

        if out_indices is None:
            self.out_indices = [2, 5, 8, 11] if total_blocks <= 12 else [5, 11, 17, 23]
        else:
            self.out_indices = out_indices

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

        # 3. Channels
        feature_info = self.backbone.feature_info
        if len(feature_info) > len(self.out_indices):
            in_channels_list = [feature_info[i]['num_chs'] for i in self.out_indices]
        else:
            in_channels_list = [info['num_chs'] for info in feature_info]

        # 4. Pyramid & Fusion
        self.pyramid = SyntheticPyramid(in_channels_list, dim=fusion_dim)
        self.fusion = PyramidFusion(dim=fusion_dim)

        # 5. Heads
        # Input to head is 1/4 scale (feature stride 4)
        # We need to output stride 4 (128px for 512px input)
        self.loc_head = nn.Sequential(
            nn.Conv2d(fusion_dim, head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, 1),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(-4.6)  # Focal Init

        self.cls_head = nn.Sequential(
            nn.Conv2d(fusion_dim, head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(head_conv, num_classes)
        )
        self.cls_head[-1].bias.data.fill_(0.0)

    def forward(self, x, debug=False):
        if x.shape[2:] != self.input_resolution:
            x = F.interpolate(x, size=self.input_resolution, mode='bicubic', align_corners=False)

        # 1. Extract [L2, L5, L8, L11] (All 1/16 scale)
        raw_features = self.backbone(x)

        # 2. Build Pyramid [1/4, 1/8, 1/16, 1/32]
        pyramid_feats = self.pyramid(raw_features)

        # 3. Fuse Top-Down -> 1/4 scale
        fused = self.fusion(pyramid_feats)

        # 4. Predict
        heatmap = self.loc_head(fused)

        # Sanity check: Ensure 128x128 output
        if heatmap.shape[2:] != (128, 128):
            heatmap = F.interpolate(heatmap, size=(128, 128), mode='bilinear')

        cls_logits = self.cls_head(fused)
        cls_out = cls_logits.view(cls_logits.size(0), self.num_classes, 1, 1)
        cls_out_16x16 = F.interpolate(cls_out, size=(16, 16), mode='nearest')

        if debug:
            def _r(t): return F.interpolate(t, size=(128, 128), mode='bilinear')

            return {
                'prediction': heatmap,
                'backbone': {f'layer_{k}': _r(v) for k, v in zip(self.out_indices, raw_features)},
                'processed': {f'p{i + 1}': _r(v) for i, v in enumerate(pyramid_feats)},  # Scales
                'fused': _r(fused)
            }

        return heatmap, cls_out_16x16