__copyright__ = \
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

from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .register import MODELS


class DLAFeatureUpsampler(nn.Module):
    """Mimics DLAUp with top-down feature aggregation like FPN."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])

    def forward(self, features):
        """
        Args:
            features: list of feature maps, deepest first
        Returns:
            Fused feature map at highest spatial resolution
        """
        x = self.projects[-1](features[-1])  # smallest resolution
        for i in range(len(features) - 2, -1, -1):
            up = F.interpolate(x, size=features[i].shape[2:], mode='nearest')
            lateral = self.projects[i](features[i])
            x = up + lateral
        return x

def _load_backbone_checkpoint(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Optionally remove "module." if trained with DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 3. Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return model

@MODELS.register()
class HerdNetTimm(nn.Module):
    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True,
        down_ratio: Optional[int] = 2,
        head_conv: int = 64,
            pretrained_path=None,
            debug=True,
            backbone='dla34'
    ):
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], f"Invalid down_ratio: {down_ratio}"
        assert num_layers == 34, "Only DLA-34 is supported with timm currently"

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.first_level = int(np.log2(down_ratio)) - 1 #  There is one layer less than in the original DLA

        # Backbone
        base = timm.create_model("dla34", pretrained=pretrained, features_only=True)

        base = _load_backbone_checkpoint(base, pretrained_path) if pretrained_path else base



        # base = timm.create_model("dla34", checkpoint_path=pretrained_path, features_only=True)
        # base = timm.create_model("dla169", pretrained=pretrained, features_only=True)

        # base = timm.create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=pretrained, features_only=True)
        # TODO convnextv2_large would require an upsample from 128 to 256

        # base = timm.create_model("swinv2_large_window12to16_192to256.ms_in22k_ft_in1k", pretrained=pretrained, features_only=True)
        # TODO seems great but requires some more magic

        self.backbone = base

        # Get feature info from timm
        feature_info = self.backbone.feature_info
        self.feature_channels = [info['num_chs'] for info in feature_info]
        self.feature_strides = [info['reduction'] for info in feature_info]
        self.num_features = len(feature_info)

        if debug:
            print(f"\nBackbone '{backbone}' provides {self.num_features} feature levels:")
            for i, info in enumerate(feature_info):
                print(f"  Level {i}: channels={info['num_chs']}, stride={info['reduction']}, module={info['module']}")
        


        # Inspect what the backbone actually returns
        if debug:
            self._inspect_backbone()

        self.feature_channels = base.feature_info.channels()

        # Subset of features depending on down_ratio
        selected_channels = self.feature_channels[self.first_level:]
        self.dla_up = DLAFeatureUpsampler(selected_channels, out_channels=selected_channels[0])

        # Bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            selected_channels[0], selected_channels[0],
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Localization head
        self.loc_head = nn.Sequential(
            nn.Conv2d(selected_channels[0], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(0.0)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(selected_channels[-1], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1)
        )
        self.cls_head[-1].bias.data.fill_(0.0)


    def _inspect_backbone(self):
        """Debug function to inspect what backbone returns"""
        print(f"\nInspecting backbone outputs:")
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        print(f"Backbone returns {len(features)} features:")
        for i, feat in enumerate(features):
            print(f"  Feature {i}: shape={feat.shape}")

    def forward(self, x):
        feats = self.backbone(x)                          # full feature pyramid
        selected_feats = feats[self.first_level:]         # according to down_ratio
        selected_feats = feats         # according to down_ratio

        upsampled = self.dla_up(selected_feats)           # DLAUp-like fused map

        # Localization heatmap
        fused = self.bottleneck_conv(upsampled)
        heatmap = self.loc_head(fused)                    # shape: (B, 1, H, W)

        # Classification from deepest feature map (for global task)
        cls_out = self.cls_head(selected_feats[-1])       # shape: (B, C, h, w)

        return heatmap, cls_out
    
    def freeze(self, layers: list) -> None:
        ''' Freeze all layers mentioned in the input list '''
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
    
    def reshape_classes(self, num_classes: int) -> None:
        ''' Reshape architecture according to a new number of classes.

        Arg:
            num_classes (int): new number of classes
        '''
        
        self.cls_head[-1] = nn.Conv2d(
                self.head_conv, num_classes, 
                kernel_size=1, stride=1, 
                padding=0, bias=True
                )

        self.cls_head[-1].bias.data.fill_(0.00)

        self.num_classes = num_classes