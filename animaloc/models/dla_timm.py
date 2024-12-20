
import timm
import torch
from torch import nn

class DLASegRefactored(nn.Module):
    def __init__(self, base_model='dla34', heads=None, head_conv=256, down_ratio=4):
        super(DLASegRefactored, self).__init__()
        self.heads = heads
        self.down_ratio = down_ratio

        # Load DLA backbone from timm
        self.base = timm.create_model(base_model, pretrained=True, features_only=True, out_indices=(1, 2, 3, 4))

        # Channels of the output feature maps at different levels
        self.channels = self.base.feature_info.channels()
        first_level = int(down_ratio.bit_length()) - 1  # Compute first level based on down_ratio
        self.first_level = first_level

        # Define upsampling
        self.upsample = nn.Upsample(scale_factor= 2 **first_level, mode='bilinear', align_corners=True)

        # Define detection heads
        for head in heads:
            classes = heads[head]
            fc = nn.Sequential(
                nn.Conv2d(self.channels[-1], head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes, kernel_size=1)
            )
            self.__setattr__(head, fc)

    def forward(self, x):
        # Extract features from the backbone
        features = self.base(x)
        x = features[-1]
        x = self.upsample(x)  # Upsample the feature map

        # Apply heads
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

# Example usage
if __name__ == "__main__":
    heads = {'hm': 80, 'wh': 2, 'reg': 2}  # Example task heads
    model = DLASegRefactored(base_model='dla34', heads=heads, head_conv=256, down_ratio=4)
    input_tensor = torch.randn(1, 3, 512, 512)  # Example input
    outputs = model(input_tensor)
    for k, v in outputs[0].items():
        print(f"{k}: {v.shape}")
