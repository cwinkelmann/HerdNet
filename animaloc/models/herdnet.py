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

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

from typing import List
import torch

import torch.nn as nn
import numpy as np
import torchvision.transforms as T

from typing import Optional

from .register import MODELS

from . import dla as dla_modules
import timm
BatchNorm = nn.BatchNorm2d

@MODELS.register()
class HerdNet(nn.Module):
    ''' HerdNet architecture '''

    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True, 
        down_ratio: Optional[int] = 2, 
        head_conv: int = 64
        ):
        '''
        Args:
            num_layers (int, optional): number of layers of DLA. Defaults to 34.
            num_classes (int, optional): number of output classes, background included. 
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained DLA encoder parameters
                from ImageNet. Defaults to True.
            down_ratio (int, optional): downsample ratio. Possible values are 1, 2, 4, 8, or 16. 
                Set to 1 to get output of the same size as input (i.e. no downsample).
                Defaults to 2.
            head_conv (int, optional): number of supplementary convolutional layers at the end 
                of decoder. Defaults to 64.
        '''

        super(HerdNet, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'
        
        base_name = 'dla{}'.format(num_layers)

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv

        self.first_level = int(np.log2(down_ratio))

        # backbone
        base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
        setattr(self, 'base_0', base)
        setattr(self, 'channels_0', base.channels)

        channels = self.channels_0

        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = dla_modules.DLAUp(channels[self.first_level:], scales=scales)
        # self.cls_dla_up = dla_modules.DLAUp(channels[-3:], scales=scales[:3])

        # bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1], 
            kernel_size=1, stride=1, 
            padding=0, bias=True
        )

        # localization head
        self.loc_head = nn.Sequential(
            nn.Conv2d(channels[self.first_level], head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, 1, 
                kernel_size=1, stride=1, 
                padding=0, bias=True
                ),
            nn.Sigmoid()
            )

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
        
    def forward(self, input: torch.Tensor):

        encode = self.base_0(input)    
        bottleneck = self.bottleneck_conv(encode[-1])
        encode[-1] = bottleneck

        decode_hm = self.dla_up(encode[self.first_level:])
        # decode_cls = self.cls_dla_up(encode[-3:])

        heatmap = self.loc_head(decode_hm)
        clsmap = self.cls_head(bottleneck)
        # clsmap = self.cls_head(decode_cls)

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


@MODELS.register()
class HerdNetResNet(nn.Module):
    ''' HerdNet architecture with ResNet backbone '''

    def __init__(
            self,
            num_layers: int = 34,
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: int = 2,
            head_conv: int = 64
    ):
        '''
        Args:
            num_layers (int, optional): number of layers of ResNet backbone. Defaults to 34.
            num_classes (int, optional): number of output classes, background included.
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained backbone encoder parameters
                from ImageNet. Defaults to True.
            down_ratio (int, optional): downsample ratio. Possible values are 1, 2, 4, 8, or 16.
                Set to 1 to get output of the same size as input (i.e. no downsample).
                Defaults to 2.
            head_conv (int, optional): number of supplementary convolutional layers at the end
                of decoder. Defaults to 64.
        '''

        super(HerdNetResNet, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.first_level = int(np.log2(down_ratio))

        # Initialize ResNet backbone
        resnet_name = f'resnet{num_layers}'
        pretrained_str = 'imagenet' if pretrained else None
        base = timm.create_model(resnet_name, pretrained=pretrained_str)

        # Extract ResNet components
        self.backbone = nn.ModuleList([
            nn.Sequential(base.conv1, base.bn1, base.act1, base.maxpool),  # stem
            base.layer1,  # layer1
            base.layer2,  # layer2
            base.layer3,  # layer3
            base.layer4,  # layer4
        ])

        # Define channels based on ResNet architecture
        if num_layers <= 34:  # ResNet18 and ResNet34 use BasicBlock
            self.channels = [64, 64, 128, 256, 512]
        else:  # ResNet50, 101, 152 use Bottleneck with expansion=4
            self.channels = [64, 256, 512, 1024, 2048]

        # Get channels for the levels we'll use
        channels_for_upsampling = self.channels[self.first_level:]

        # Bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            self.channels[-1], self.channels[-1],
            kernel_size=1, stride=1,
            padding=0, bias=True
        )

        # Create FPN-style components
        self.lateral_connections = self._make_lateral_connections(channels_for_upsampling)
        self.upsampling_layers = self._make_upsampling_layers(channels_for_upsampling)
        self.fusion_nodes = self._make_fusion_nodes(channels_for_upsampling)

        # Final projection layer to match expected channels for localization head
        self.final_proj = nn.Conv2d(
            self.channels[-1], self.channels[self.first_level],
            kernel_size=1, stride=1, padding=0, bias=False
        )

        # Localization head with additional upsampling if needed
        self._build_localization_head()

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.channels[-1], head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, self.num_classes,
                kernel_size=1, stride=1,
                padding=0, bias=True
            )
        )

        self.cls_head[-1].bias.data.fill_(0.00)

    def _make_lateral_connections(self, channels: List[int]):
        """Create lateral connections for FPN-style upsampling"""
        layers = nn.ModuleList()

        # For each level except the deepest (which doesn't need transformation)
        for i in range(len(channels) - 1):
            lateral = nn.Sequential(
                nn.Conv2d(channels[i], channels[-1], kernel_size=1, stride=1, padding=0, bias=False),
                BatchNorm(channels[-1]),
                nn.ReLU(inplace=True)
            )
            layers.append(lateral)

        return layers

    def _make_upsampling_layers(self, channels: List[int]):
        """Create upsampling layers for ResNet backbone"""
        layers = nn.ModuleList()

        # Scales - each level needs to be upsampled by a different factor
        scales = [2 ** i for i in range(len(channels) - 1)]

        # For each level except the deepest
        for i in range(len(channels) - 1):
            factor = scales[i]

            if factor == 1:
                up_layer = nn.Identity()
            else:
                up_layer = nn.ConvTranspose2d(
                    channels[-1], channels[-1],
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

    def _build_localization_head(self):
        """Build the localization head with proper upsampling"""
        layers = []

        # If down_ratio is greater than 1, add upsampling to get back to original resolution
        additional_upscale = 2 ** self.first_level

        if additional_upscale > 1:
            layers.append(nn.ConvTranspose2d(
                self.channels[self.first_level], self.channels[self.first_level],
                kernel_size=additional_upscale * 2,
                stride=additional_upscale,
                padding=additional_upscale // 2,
                output_padding=0,
                groups=self.channels[self.first_level],  # Depthwise convolution
                bias=False
            ))
            # Initialize the upsampling for bilinear interpolation
            self._fill_up_weights(layers[-1])

        # Add standard localization head after upsampling
        layers.extend([
            nn.Conv2d(self.channels[self.first_level], self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.head_conv, 1,
                kernel_size=1, stride=1,
                padding=0, bias=True
            ),
            nn.Sigmoid()
        ])

        self.loc_head = nn.Sequential(*layers)
        self.loc_head[-2].bias.data.fill_(0.00)

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
        # Extract features from ResNet backbone
        features = []
        x = input

        # Process through each stage of the backbone
        for layer in self.backbone:
            x = layer(x)
            features.append(x)

        # Get the features we need for upsampling based on first_level
        features_for_upsampling = features[self.first_level:]

        # Apply bottleneck to the deepest feature
        bottleneck = self.bottleneck_conv(features_for_upsampling[-1])
        features_for_upsampling[-1] = bottleneck

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
            upsampled = self.upsampling_layers[i](x)

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

        # Transform channel dimension for localization head
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