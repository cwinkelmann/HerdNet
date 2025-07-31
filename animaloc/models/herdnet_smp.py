import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List, Dict

from .register import MODELS
import segmentation_models_pytorch as smp


@MODELS.register()
class HerdNetSMP(nn.Module):
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

        model = smp.create_model(
            arch="DeepLabV3Plus",  # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
            encoder_name="mit_b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes-1,
        )

        self.model = model

    def forward(self, x):
        class_predictions = self.model(x)
        B, C, H, W = class_predictions.shape

        # Option 1a: Heatmap = max across all classes (localization)
        heatmap, _ = torch.max(class_predictions, dim=1, keepdim=True)  # (B, 1, H, W)

        # Downsample to match target resolution (7x7)
        target_size = 7  # or extract from your target tensor

        # Downsample predictions to target size
        predictions_small = F.adaptive_avg_pool2d(class_predictions, (target_size, target_size))

        # Option 1b: Cls_map = add background class
        background = torch.zeros(predictions_small.shape[0], 1, target_size, target_size,
                                 device=x.device)
        cls_map = torch.cat([background, predictions_small], dim=1)  # (B, num_classes, H, W)

        return heatmap, cls_map