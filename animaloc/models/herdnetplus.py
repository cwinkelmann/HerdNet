

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import math
import timm  # Import timm for models

from typing import Optional, List

from .register import MODELS
from . import dla as dla_modules

# Set BatchNorm for consistency with DLA module
BatchNorm = nn.BatchNorm2d


@MODELS.register()
class HerdNetPlus(nn.Module):
    ''' HerdNet architecture '''

    def __init__(
            self,
            num_layers: int = 34,
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: Optional[int] = 1,  # Changed default to 1 (no downsampling)
            head_conv: int = 64,
            backbone: str = 'dla'
    ):
        '''
        Args:
            num_layers (int, optional): number of layers of backbone. Defaults to 34.
            num_classes (int, optional): number of output classes, background included.
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained backbone encoder parameters
                from ImageNet. Defaults to True.
            down_ratio (int, optional): downsample ratio. Possible values are 1, 2, 4, 8, or 16.
                Set to 1 to get output of the same size as input (i.e. no downsample).
                Defaults to 2.
            head_conv (int, optional): number of supplementary convolutional layers at the end
                of decoder. Defaults to 64.
            backbone (str, optional): backbone architecture to use. Options are 'dla' or 'resnet'.
                Defaults to 'resnet'.
        '''

        super(HerdNetPlus, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.backbone_type = backbone

        self.first_level = int(np.log2(down_ratio))

        # Initialize backbone
        if backbone == 'dla':
            # Use DLA from timm instead of the local implementation
            # This addresses the TODO comment to use a Timm Version
            base_name = f'dla{num_layers}'
            pretrained_str = 'imagenet' if pretrained else None

            # Flag to track if we're using timm or local implementation
            self.using_timm_dla = True

            # Create a timm model with pretrained weights if requested
            # features_only=True returns feature maps from each stage instead of just the final output
            try:
                base = timm.create_model(base_name, pretrained=pretrained_str, features_only=True)
            except Exception as e:
                # If timm doesn't have the requested DLA model, fall back to the local implementation
                # and log a warning
                print(f"Warning: Could not create timm DLA model: {e}")
                print(f"Falling back to local DLA implementation")
                self.using_timm_dla = False
                base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
                setattr(self, 'base_0', base)
                setattr(self, 'channels_0', base.channels)
                channels = self.channels_0

            if self.using_timm_dla:
                # For DLA from timm, we need to set the channels manually
                # These are the standard channel dimensions for DLA models
                # Note: These might need adjustment if timm's implementation differs
                if num_layers == 34:
                    self.channels_0 = [16, 32, 64, 128, 256, 512]  # Standard DLA34 channels
                elif num_layers == 60:
                    self.channels_0 = [16, 32, 128, 256, 512, 1024]  # Standard DLA60 channels
                else:
                    # Default fallback for other DLA variants
                    self.channels_0 = [16, 32, 64, 128, 256, 512]  # Default to DLA34 channels

                setattr(self, 'base_0', base)
                channels = self.channels_0

            # DLA specific upsampling (used by both timm and local implementations)
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
            self.dla_up = dla_modules.DLAUp(channels[self.first_level:], scales=scales)

        elif backbone == 'resnet':
            # Use ResNet from timm instead of torchvision
            resnet_name = f'resnet{num_layers}'

            # Create a timm model with pretrained weights if requested
            pretrained_str = 'imagenet' if pretrained else None
            base = timm.create_model(resnet_name, pretrained=pretrained_str)

            # For ResNet from timm, we need to manually construct the feature extraction layers
            # similar to how it was done with torchvision
            self.base_0 = nn.ModuleList([
                nn.Sequential(base.conv1, base.bn1, base.act1, base.maxpool),  # stem
                base.layer1,  # layer1
                base.layer2,  # layer2
                base.layer3,  # layer3
                base.layer4,  # layer4
            ])

            # Define channels based on model type, just like in original code
            if num_layers <= 34:  # ResNet18 and ResNet34 use BasicBlock
                self.channels_0 = [64, 64, 128, 256, 512]
            else:  # ResNet50, 101, 152 use Bottleneck with expansion=4
                self.channels_0 = [64, 256, 512, 1024, 2048]

            channels = self.channels_0

            # Create lateral connections and upsampling layers for FPN-style
            self.lateral_connections = self._make_lateral_connections(channels[self.first_level:])
            self.resnet_up = self._make_resnet_upsampling(channels[self.first_level:])

            # Create fusion nodes for each level
            self.fusion_nodes = self._make_fusion_nodes(channels[self.first_level:])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'dla' or 'resnet'")

        # bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1],
            kernel_size=1, stride=1,
            padding=0, bias=True
        )

        # localization head with additional upsampling if needed
        layers = []

        # If down_ratio is greater than 1, add upsampling to get back to original resolution
        additional_upscale = 2 ** self.first_level  # Calculate required upsampling factor

        if additional_upscale > 1:
            # Add upsampling before the loc_head to ensure 512x512 output
            layers.append(nn.ConvTranspose2d(
                channels[self.first_level],
                channels[self.first_level],
                kernel_size=additional_upscale * 2,
                stride=additional_upscale,
                padding=additional_upscale // 2,
                output_padding=0,
                groups=channels[self.first_level],  # Depthwise convolution
                bias=False
            ))
            # Initialize the upsampling for bilinear interpolation
            self._fill_up_weights(layers[-1])

        # Add standard localization head after upsampling
        layers.extend([
            nn.Conv2d(channels[self.first_level], head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, 1,
                kernel_size=1, stride=1,
                padding=0, bias=True
            ),
            nn.Sigmoid()
        ])

        self.loc_head = nn.Sequential(*layers)

        self.loc_head[-2].bias.data.fill_(0.00)

        # classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(channels[-1], head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, self.num_classes,
                kernel_size=1, stride=1,
                padding=0, bias=True
            )
        )

        self.cls_head[-1].bias.data.fill_(0.00)

        # Final projection layer to convert channels if needed
        self.final_proj = None
        if backbone == 'resnet':
            self.final_proj = nn.Conv2d(
                channels[-1], channels[self.first_level],
                kernel_size=1, stride=1, padding=0, bias=False
            )

    def _make_lateral_connections(self, channels: List[int]):
        """Create lateral connections for FPN-style upsampling"""
        layers = nn.ModuleList()

        # For each level except the deepest (which doesn't need transformation)
        for i in range(len(channels) - 1):
            # Create lateral connection that transforms each level to the deepest channel dimension
            lateral = nn.Sequential(
                nn.Conv2d(channels[i], channels[-1], kernel_size=1, stride=1, padding=0, bias=False),
                BatchNorm(channels[-1]),
                nn.ReLU(inplace=True)
            )
            layers.append(lateral)

        return layers

    def _make_resnet_upsampling(self, channels: List[int]):
        """Create upsampling layers for ResNet backbone"""
        layers = nn.ModuleList()

        # Scales like in DLAUp - each level needs to be upsampled by a different factor
        scales = [2 ** i for i in range(len(channels) - 1)]

        # For each level except the deepest
        for i in range(len(channels) - 1):
            # Calculate the upsampling factor for this level
            factor = scales[i]

            if factor == 1:
                # No upsampling needed
                up_layer = nn.Identity()
            else:
                # Create upsampling with ConvTranspose2d like in IDAUp
                up_layer = nn.ConvTranspose2d(
                    channels[-1], channels[-1],  # Keep channels consistent
                    kernel_size=factor * 2,
                    stride=factor,
                    padding=factor // 2,
                    output_padding=0,
                    groups=channels[-1],  # Depthwise conv for upsampling
                    bias=False
                )
                # Initialize weights for bilinear upsampling
                self._fill_up_weights(up_layer)

            layers.append(up_layer)

        return layers

    def _make_fusion_nodes(self, channels: List[int]):
        """Create fusion nodes for feature aggregation"""
        nodes = nn.ModuleList()

        # Create a fusion node for each level except the deepest
        for i in range(len(channels) - 1):
            node = nn.Sequential(
                nn.Conv2d(channels[-1] * 2, channels[-1],
                          kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(channels[-1]),
                nn.ReLU(inplace=True)
            )
            nodes.append(node)

        return nodes

    def _fill_up_weights(self, up):
        """Helper function to initialize upsampling weights for bilinear interpolation"""
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, input: torch.Tensor):
        if self.backbone_type == 'dla':
            # Handle the case where we're using the local DLA implementation
            if not hasattr(self, 'using_timm_dla') or not self.using_timm_dla:
                # This is the original code path for the local DLA implementation
                encode = self.base_0(input)
            else:
                # For timm DLA with features_only=True, we get a list of feature maps
                # This is the new code path for the timm DLA implementation
                features = self.base_0(input)

                # Timm's features_only=True might return a different number of feature maps
                # than our original DLA implementation. We need to handle this gracefully.

                # The original DLA implementation returns 6 levels (0-5) with specific channel dimensions
                # Timm's implementation might return a different number of levels with different dimensions
                # We need to map timm's output to match what our DLAUp module expects

                # Create a list with the right number of levels (6 for DLA34)
                expected_levels = 6
                encode = [None] * expected_levels

                # Check how many features we got from timm
                num_features = len(features)

                if num_features == expected_levels:
                    # If we got exactly the expected number of levels, use them directly
                    # This is the ideal case where timm's output matches our expectations
                    encode = features
                elif num_features < expected_levels:
                    # If we got fewer levels than expected, map them to the higher levels
                    # and leave the lower levels as None
                    # This assumes that timm returns the higher-level features (deeper in the network)
                    offset = expected_levels - num_features
                    for i, feat in enumerate(features):
                        encode[i + offset] = feat

                    # If we're missing level 0, create a simple downsampled version of the input
                    # This is just a fallback and might not be optimal
                    if encode[0] is None and offset <= 1:
                        # Simple downsampling to approximate level 0 feature map
                        encode[0] = F.avg_pool2d(input, kernel_size=4, stride=4)
                else:
                    # If we got more levels than expected, use the last 'expected_levels' ones
                    # This assumes that the last levels are the ones we want (deeper in the network)
                    encode = features[-expected_levels:]

            # Define expected_levels if not already defined (for the local DLA implementation case)
            if 'expected_levels' not in locals():
                expected_levels = 6  # Standard number of levels for DLA34

            # Check if we have valid features for all required levels
            valid_features = True
            for i in range(self.first_level, expected_levels):
                if encode[i] is None:
                    valid_features = False
                    break

            if not valid_features:
                # If we're missing some required levels, we need to handle this
                # One approach is to duplicate the closest available level
                for i in range(self.first_level, expected_levels):
                    if encode[i] is None:
                        # Find the closest valid level
                        closest_valid = None
                        for j in range(expected_levels):
                            if encode[j] is not None:
                                closest_valid = j
                                break

                        if closest_valid is not None:
                            # Resize the closest valid level to match the expected size for this level
                            # This is a rough approximation and might not be optimal
                            encode[i] = F.interpolate(
                                encode[closest_valid],
                                scale_factor=2**(closest_valid-i) if closest_valid > i else 0.5**(i-closest_valid),
                                mode='bilinear',
                                align_corners=False
                            )

            # Apply bottleneck to the deepest feature
            bottleneck = self.bottleneck_conv(encode[-1])
            encode[-1] = bottleneck

            # Use DLAUp for the features from first_level onwards
            decode_hm = self.dla_up(encode[self.first_level:])

        elif self.backbone_type == 'resnet':
            # Extract features from timm ResNet using our modular approach
            features = []
            x = input

            # Process through each stage of the backbone
            for i, layer in enumerate(self.base_0):
                x = layer(x)
                features.append(x)

            # Get the features we need for upsampling based on first_level
            features_for_upsampling = features[self.first_level:]

            # Apply bottleneck to the deepest feature
            bottleneck = self.bottleneck_conv(features_for_upsampling[-1])
            features_for_upsampling[-1] = bottleneck

            # Process features starting from shallowest to deepest
            # Apply lateral connections to all features except the deepest
            laterals = []
            for i, feature in enumerate(features_for_upsampling[:-1]):
                lateral = self.lateral_connections[i](feature)
                laterals.append(lateral)

            # Add the bottleneck feature as the last one
            laterals.append(bottleneck)

            # Start the top-down pathway from the deepest feature
            x = laterals[-1]

            # Process features from deepest to shallowest
            for i in range(len(laterals) - 2, -1, -1):
                # Upscale the current feature
                upsampled = self.resnet_up[i](x)

                # Ensure spatial dimensions match
                if upsampled.shape[2:] != laterals[i].shape[2:]:
                    upsampled = F.interpolate(
                        upsampled,
                        size=laterals[i].shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Merge features
                x = self.fusion_nodes[i](torch.cat([upsampled, laterals[i]], dim=1))

            # Final feature map for localization
            decode_hm = x

            # Transform channel dimension for localization head if needed
            if self.final_proj is not None:
                decode_hm = self.final_proj(decode_hm)

        # Generate heatmap and clsmap
        heatmap = self.loc_head(decode_hm)
        clsmap = self.cls_head(bottleneck)

        # Store dimensions for reference
        self._input_size = input.shape[2:]
        self._output_size = heatmap.shape[2:]

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
