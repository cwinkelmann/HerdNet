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
import torchvision.transforms as T
import torchvision.models as models

from typing import Optional, List

from .register import MODELS

from . import dla as dla_modules


@MODELS.register()
class HerdNet(nn.Module):
    ''' HerdNet architecture '''

    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True, 
        down_ratio: Optional[int] = 2, 
        head_conv: int = 64,
        backbone: str = 'resnet'
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

        super(HerdNet, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'
        
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.backbone_type = backbone

        self.first_level = int(np.log2(down_ratio))

        # Initialize backbone
        if backbone == 'dla':
            base_name = f'dla{num_layers}'
            base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
            setattr(self, 'base_0', base)
            setattr(self, 'channels_0', base.channels)
            channels = self.channels_0
            
            # DLA specific upsampling
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
            self.dla_up = dla_modules.DLAUp(channels[self.first_level:], scales=scales)
            
        elif backbone == 'resnet':
            # Use ResNet from torchvision
            resnet_name = f'resnet{num_layers}'
            weights = 'IMAGENET1K_V1' if pretrained else None
            base = getattr(models, resnet_name)(weights=weights)
            
            # Remove the final layers (avgpool and fc)
            self.base_0 = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,  # 1/4
                base.layer2,  # 1/8
                base.layer3,  # 1/16
                base.layer4,  # 1/32
            )
            
            # Define channels for ResNet
            if num_layers <= 34:  # ResNet18 and ResNet34 use BasicBlock
                self.channels_0 = [64, 64, 128, 256, 512]
            else:  # ResNet50, 101, 152 use Bottleneck with expansion=4
                self.channels_0 = [64, 256, 512, 1024, 2048]
                
            channels = self.channels_0
            
            # Create upsampling layers for ResNet
            self.resnet_up = self._make_resnet_upsampling(channels[self.first_level:])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'dla' or 'resnet'")

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
    
    def _make_resnet_upsampling(self, channels: List[int]):
        """Create upsampling layers for ResNet backbone"""
        layers = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_channels = channels[i + 1]
            out_channels = channels[i]
            
            up_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            layers.append(up_layer)
            
        return layers
        
    def forward(self, input: torch.Tensor):
        if self.backbone_type == 'dla':
            encode = self.base_0(input)    
            bottleneck = self.bottleneck_conv(encode[-1])
            encode[-1] = bottleneck
            decode_hm = self.dla_up(encode[self.first_level:])
            
        elif self.backbone_type == 'resnet':
            # Extract features from ResNet
            features = []
            x = input
            
            # Extract features at different levels
            x = self.base_0[0](x)  # conv1
            x = self.base_0[1](x)  # bn1
            x = self.base_0[2](x)  # relu
            features.append(x)     # 1/2 resolution
            
            x = self.base_0[3](x)  # maxpool
            x = self.base_0[4](x)  # layer1
            features.append(x)     # 1/4 resolution
            
            x = self.base_0[5](x)  # layer2
            features.append(x)     # 1/8 resolution
            
            x = self.base_0[6](x)  # layer3
            features.append(x)     # 1/16 resolution
            
            x = self.base_0[7](x)  # layer4
            features.append(x)     # 1/32 resolution
            
            # Apply bottleneck to the last feature
            bottleneck = self.bottleneck_conv(features[-1])
            features[-1] = bottleneck
            
            # Upsample features
            decode_features = [features[self.first_level + len(self.resnet_up)]]
            
            for i, up_layer in enumerate(self.resnet_up):
                src_idx = len(features) - 2 - i
                decode_features.append(up_layer(decode_features[-1]) + features[src_idx])
                
            decode_hm = decode_features[-1]

        heatmap = self.loc_head(decode_hm)
        clsmap = self.cls_head(bottleneck)

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