import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List, Dict

from .register import MODELS

class DLAFeatureAggregator(nn.Module):
    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])

    def forward(self, features):
        target_size = features[0].shape[2:]
        aggregated = 0
        for i, feat in enumerate(features):
            x = self.projections[i](feat)
            if x.shape[2:] != target_size:
                x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            aggregated = aggregated + x
        return aggregated / len(features)

@MODELS.register()
class HerdNetPlus(nn.Module):
    """HerdNet architecture with pretrained backbone from timm"""

    def __init__(
            self,
            backbone: str = 'dla34',
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: int = 4,
            head_conv: int = 64,
            pretrained_path: Optional[str] = None,
            debug: bool = True
    ):
        """
        Args:
            backbone: timm model name (e.g., 'dla34', 'resnet50')
            num_classes: number of output classes
            pretrained: use ImageNet pretrained weights
            down_ratio: output downsample ratio (1, 2, 4, 8, 16)
            head_conv: channels in head convolutions
            pretrained_path: path to custom pretrained weights
            debug: print debug information about features
        """
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], f'Invalid down_ratio: {down_ratio}'

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.debug = debug

        # Create backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained and pretrained_path is None,
            features_only=True
        )

        # Load custom weights if provided
        if pretrained_path:
            self._load_custom_weights(pretrained_path)

        # Inspect what the backbone actually returns
        if debug:
            self._inspect_backbone()

        # Get feature info from timm
        feature_info = self.backbone.feature_info
        self.feature_channels = [info['num_chs'] for info in feature_info]
        self.feature_strides = [info['reduction'] for info in feature_info]
        self.num_features = len(feature_info)

        if debug:
            print(f"\nBackbone '{backbone}' provides {self.num_features} feature levels:")
            for i, info in enumerate(feature_info):
                print(f"  Level {i}: channels={info['num_chs']}, stride={info['reduction']}, module={info['module']}")

        # Select which features to use based on down_ratio
        # We'll use features starting from the one that matches or exceeds our target down_ratio
        self.first_level_idx = 0
        for i, stride in enumerate(self.feature_strides):
            if stride >= down_ratio:
                self.first_level_idx = i
                break

        # Channels from the selected features
        self.selected_channels = self.feature_channels[self.first_level_idx:]
        self.selected_strides = self.feature_strides[self.first_level_idx:]


        if debug:
            print(f"\nUsing features from level {self.first_level_idx} onwards")
            print(f"Selected channels: {self.selected_channels}")
            print(f"Selected strides: {self.selected_strides}")

        # Create decoder
        self._init_decoder()

        # Create heads
        self._init_heads()

    def _inspect_backbone(self):
        """Debug function to inspect what backbone returns"""
        print(f"\nInspecting backbone outputs:")
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        print(f"Backbone returns {len(features)} features:")
        for i, feat in enumerate(features):
            print(f"  Feature {i}: shape={feat.shape}")

    def _init_decoder(self):
        """Initialize decoder to aggregate multi-scale features"""
        # Use the shallowest selected feature's channels as decoder dimension
        decoder_dim = self.selected_channels[0]

        # Create lateral connections (1x1 convs to match channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in self.selected_channels:
            if in_channels != decoder_dim:
                self.lateral_convs.append(
                    nn.Conv2d(in_channels, decoder_dim, 1, bias=False)
                )
            else:
                self.lateral_convs.append(nn.Identity())

        # Create fusion blocks for combining features
        self.fusion_blocks = nn.ModuleList()
        for i in range(len(self.selected_channels) - 1):
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True)
                )
            )

    def _init_heads(self):
        """Initialize detection heads"""
        decoder_dim = self.selected_channels[0]

        # Localization head
        loc_layers = []

        # Check if we need additional upsampling to reach target down_ratio
        output_stride = self.selected_strides[0]
        if self.down_ratio < output_stride:
            scale = output_stride // self.down_ratio
            loc_layers.append(
                nn.ConvTranspose2d(
                    decoder_dim, decoder_dim,
                    kernel_size=scale * 2, stride=scale,
                    padding=scale // 2, bias=False
                )
            )
            loc_layers.append(nn.BatchNorm2d(decoder_dim))
            loc_layers.append(nn.ReLU(inplace=True))

        loc_layers.extend([
            nn.Conv2d(decoder_dim, self.head_conv, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 1, 1, bias=True),
            nn.Sigmoid()
        ])

        self.loc_head = nn.Sequential(*loc_layers)

        # Classification head (operates on deepest features)
        deepest_channels = self.selected_channels[-1]
        self.cls_head = nn.Sequential(
            nn.Conv2d(deepest_channels, self.head_conv, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.num_classes, 1, bias=True)
        )

        # Initialize biases (matching original HerdNet initialization)
        # The bias is in the Conv2d layer before the Sigmoid (index -2)
        self.loc_head[-2].bias.data.fill_(0.00)
        # Classification head's last layer is Conv2d, so -1 is correct
        self.cls_head[-1].bias.data.fill_(0.00)

    def _load_custom_weights(self, path: str):
        """Load custom pretrained weights"""
        try:
            checkpoint = torch.load(path, map_location='cpu')

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter backbone weights
            backbone_dict = {}
            for k, v in state_dict.items():
                # Remove common prefixes
                key = k
                for prefix in ['backbone.', 'base_0.', 'encoder.']:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        break

                # Skip non-backbone keys
                if any(x in key for x in ['head', 'lateral', 'fusion', 'decoder']):
                    continue

                backbone_dict[key] = v

            # Load weights
            missing, unexpected = self.backbone.load_state_dict(backbone_dict, strict=False)
            if self.debug:
                print(f"\nLoaded {len(backbone_dict)} backbone parameters from {path}")
                if missing:
                    print(f"Missing keys: {len(missing)}")
                if unexpected:
                    print(f"Unexpected keys: {len(unexpected)}")

        except Exception as e:
            print(f"Error loading weights from {path}: {e}")

    def forward(self, x):
        # Extract all features from backbone
        all_features = self.backbone(x)

        # Select features we want to use
        features = all_features[self.first_level_idx:]

        if self.debug and not hasattr(self, '_debug_printed'):
            print(f"\nForward pass - using {len(features)} features:")
            for i, feat in enumerate(features):
                print(f"  Feature {i}: shape={feat.shape}")
            self._debug_printed = True

        # Apply lateral convolutions
        laterals = []
        for feat, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feat))

        # Top-down path with feature fusion
        # Start from the shallowest feature
        output = laterals[0]

        # Fuse with deeper features
        for i in range(1, len(laterals)):
            # Upsample deeper feature to match current resolution
            if laterals[i].shape[2:] != output.shape[2:]:
                upsampled = F.interpolate(
                    laterals[i],
                    size=output.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                upsampled = laterals[i]

            # Add and fuse
            output = output + upsampled
            if i - 1 < len(self.fusion_blocks):
                output = self.fusion_blocks[i - 1](output)

        # Generate outputs
        heatmap = self.loc_head(output)

        # Classification uses deepest features
        clsmap = self.cls_head(features[-1])

        return heatmap, clsmap

    def freeze(self, layers: List[str]):
        """Freeze specified layers"""
        for layer in layers:
            if hasattr(self, layer):
                for param in getattr(self, layer).parameters():
                    param.requires_grad = False
                print(f"Froze layer: {layer}")

    def reshape_classes(self, num_classes: int):
        """Change number of output classes"""
        self.num_classes = num_classes

        # Get input channels from existing head
        in_channels = self.cls_head[0].in_channels

        # Recreate classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, num_classes, 1, bias=True)
        )
        self.cls_head[-1].bias.data.zero_()


# Utility function to explore what different backbones provide
def explore_timm_backbone(backbone_name: str):
    """Helper function to explore what features a timm backbone provides"""
    print(f"\n{'=' * 60}")
    print(f"Exploring backbone: {backbone_name}")
    print('=' * 60)

    model = timm.create_model(backbone_name, pretrained=False, features_only=True)

    # Get feature info
    feature_info = model.feature_info
    print(f"\nFeature info ({len(feature_info)} levels):")
    for i, info in enumerate(feature_info):
        print(f"  Level {i}: channels={info['num_chs']}, stride={info['reduction']}, module={info['module']}")

    # Test with dummy input
    x = torch.randn(1, 3, 224, 224)
    features = model(x)

    print(f"\nActual output shapes:")
    for i, feat in enumerate(features):
        print(f"  Level {i}: {feat.shape}")

    return model, features


# Example usage
if __name__ == "__main__":
    # Explore different backbones
    for backbone in ['dla34', 'resnet50', 'efficientnet_b0']:
        explore_timm_backbone(backbone)

    print("\n" + "=" * 60)
    print("Creating HerdNetPlus model")
    print("=" * 60)

    # Create model with debug info
    model = HerdNetPlus(
        backbone='dla34',
        num_classes=2,
        pretrained=True,
        down_ratio=4,
        debug=True
    )

    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    heatmap, clsmap = model(x)
    print(f"\nOutput shapes:")
    print(f"  Heatmap: {heatmap.shape}")
    print(f"  Classification map: {clsmap.shape}")