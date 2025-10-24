
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from . import DLAUp
from .register import MODELS





def _load_backbone_checkpoint(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 3. Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return model


@MODELS.register()
class HerdNetTimmDLA(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: Optional[int] = 2,
            head_conv: int = 64,
            pretrained_path=None,
            debug=True,
            backbone='timm/dla34'
    ):
        super().__init__()

        assert down_ratio in [2, 4, 8, 16], f"Invalid down_ratio: {down_ratio}"

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv

        if "dla" in backbone:
            # DLA timm backbone has a different downsampling scheme
            # The first level is the one with stride 2, so we need to adjust accordingly
            self.first_level = int(np.log2(down_ratio)) - 1 # should be 0 for dr 2
        elif "convnext" in backbone:
            # ConvNext backbone has a different downsampling scheme
            # The first level is the one with stride 4, so we need to adjust accordingly
            # This is because ConvNext has a different feature map structure
            # compared to DLA, where the first level is the one with stride 4.
            # So we need to adjust the first_level accordingly.
            assert down_ratio >= 4, "ConvNext backbone requires down_ratio >= 4"
            self.first_level = int(np.log2(down_ratio)) - 2  # There is two layer less than in the original DLA
        elif "efficientnet" in backbone:
            # EfficientNet backbone has a different downsampling scheme
            # The first level is the one with stride 4, so we need to adjust accordingly
            self.first_level = int(np.log2(down_ratio)) - 1
        else:
            raise ValueError(f"Backbone {backbone} not supported.")

        # Backbone
        base = timm.create_model(backbone,
                                 pretrained=pretrained,
                                 features_only=True)

        base = _load_backbone_checkpoint(base, pretrained_path) if pretrained_path else base

        self.backbone = base
        # TODO give it a name to freeze it later

        # Get feature info from timm
        feature_info = self.backbone.feature_info
        self.feature_channels = [info['num_chs'] for info in feature_info]
        self.feature_strides = [info['reduction'] for info in feature_info]
        self.num_features = len(feature_info)

        if debug:
            logger.info(f"\nBackbone '{backbone}' provides {self.num_features} feature levels:")
            for i, info in enumerate(feature_info):
                logger.info(f"  Level {i}: reduction(aka down_ratio)={info['reduction']},channels={info['num_chs']}, stride={info['reduction']}, module={info['module']}")

            self._inspect_backbone()

        self.feature_channels = base.feature_info.channels()
        channels = self.feature_channels
        # Subset of features depending on down_ratio
        selected_channels = self.feature_channels[self.first_level:]
        scales = [2 ** i for i in range(len(selected_channels))]
        # selected_channels = self.feature_channels

        # self.dla_up = DLAFeatureUpsampler(selected_channels, out_channels=selected_channels[0])
        self.dla_up = DLAUp(selected_channels, scales=scales)

        # Bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1],
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Localization head
        self.loc_head = nn.Sequential(
            nn.Conv2d(channels[self.first_level], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(0.0)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(channels[-1], head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1)
        )
        self.cls_head[-1].bias.data.fill_(0.0)

        if debug:
            logger.info(f"\nModel initialized with down_ratio={down_ratio}, num_classes={num_classes}, head_conv={head_conv}")
            logger.info(f"First level for feature selection: {self.first_level}")
            self._test_forward()

    def freeze_backbone_completely(self):
        """Freeze all parameters in the backbone."""
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

    def _test_forward(self):
        """Debug function to test forward pass"""
        logger.debug(f"\nTesting forward pass with dummy input:")
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            heatmap, clsmap = self.forward(dummy_input)

        logger.debug(f"Output heatmap shape: {heatmap.shape}")
        logger.debug(f"Output clsmap shape: {clsmap.shape}")

        if self.down_ratio == 1:
            assert heatmap.shape[1:] == (1, 512,512)
            assert clsmap.shape[1:] == (self.num_classes, 16,16)
        elif self.down_ratio == 2:
            assert heatmap.shape[1:] == (1, 256,256)
            assert clsmap.shape[1:] == (self.num_classes, 16, 16)
        elif self.down_ratio == 4:
            assert heatmap.shape[1:] == (1, 128,128)
            assert clsmap.shape[1:] == (self.num_classes, 16, 16)
        elif self.down_ratio == 8:
            assert heatmap.shape[1:] == (1, 64,64)
            assert clsmap.shape[1:] == (self.num_classes, 16, 16)
        elif self.down_ratio == 16:
            assert heatmap.shape[1:] == (1, 32, 32)
            assert clsmap.shape[1:] == (self.num_classes, 16, 16)

        return heatmap, clsmap

    def _inspect_backbone(self):
        """Debug function to inspect what backbone returns"""
        logger.debug(f"\nInspecting backbone outputs:")
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        logger.debug(f"Backbone returns {len(features)} features:")
        for i, feat in enumerate(features):
            logger.debug(f"  Feature {i}: shape={feat.shape}")

    def forward(self, x):
        feats = self.backbone(x)  # full feature set
        selected_feats = feats[self.first_level:]  # according to down_ratio

        bottlenecked = self.bottleneck_conv(feats[-1])

        selected_feats[-1] = bottlenecked
        # selected_feats = feats  # according to down_ratio

        upsampled = self.dla_up(selected_feats)  # DLAUp-like fused map

        # Localization heatmap
        # fused = self.bottleneck_conv(upsampled)
        heatmap = self.loc_head(upsampled)  # shape: (B, 1, H, W)

        # Classification from deepest feature map (for global task)
        clsmap = self.cls_head(selected_feats[-1])  # shape: (B, C, h, w)

        return heatmap, clsmap

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


