import torch
import torch.nn as nn
import timm


class HerdNetP2P(torch.nn.Module):
    def __init__(self, backbone='timm/vit_large_patch16_dinov3.sat493m',
                 num_classes=2,
                 pretrained=True,
                 freeze_backbone=False,
                 hidden_dim=256,
                 criterion=None):

        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_classes = num_classes

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )
        # Initialize to predict background initially
        self.cls_head[-1].bias.data[0].fill_(2.0)  # Background bias positive
        if num_classes > 1:
            self.cls_head[-1].bias.data[1:].fill_(-4.0)  # Foreground bias negative

        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1)
        )
        nn.init.constant_(self.reg_head[-1].weight, 0)
        nn.init.constant_(self.reg_head[-1].bias, 0)

        self.criterion = criterion

    def set_criterion(self, criterion: torch.nn.Module) -> None:
        self.criterion = criterion

    def forward(self, x: torch.Tensor, targets=None) -> dict:
        B, _, img_h, img_w = x.shape

        # 1. Feature Extraction
        features = self.backbone.forward_features(x)

        B, N, C = features.shape
        grid_h = img_h // self.patch_size
        grid_w = img_w // self.patch_size

        # Remove CLS token if present
        if N > grid_h * grid_w:
            features = features[:, -grid_h * grid_w:]

        features = features.transpose(1, 2).reshape(B, C, grid_h, grid_w)

        feat = self.adapter(features)
        logits = self.cls_head(feat)
        raw_offsets = self.reg_head(feat)

        # CRITICAL: Bound offsets to [-0.5, 0.5] grid cells using tanh
        offsets = torch.tanh(raw_offsets) * 0.5

        # Decode to normalized coordinates [0, 1]
        points_normalized = self._decode_normalized(offsets, grid_h, grid_w)

        # Also compute pixel coordinates for inference
        points_pixel = points_normalized.clone()
        points_pixel[..., 0] *= img_w
        points_pixel[..., 1] *= img_h

        outputs = {
            'logits': logits,
            'points': points_pixel,  # Pixel coords for inference
            'points_normalized': points_normalized,  # [0,1] coords for loss
            'offsets': offsets,
            'image_size': (img_h, img_w)
        }

        if targets is not None and self.criterion is not None:
            loss = self.criterion(outputs, targets)
            outputs['loss_p2p'] = loss

        return outputs

    def _decode_normalized(self, offsets, h, w):
        """Decode offsets to normalized [0, 1] coordinates."""
        device = offsets.device

        # Create grid centers in normalized coordinates
        # Grid cell centers: (0.5/h, 1.5/h, ...) for y, (0.5/w, 1.5/w, ...) for x
        y_centers = (torch.arange(h, device=device, dtype=offsets.dtype) + 0.5) / h
        x_centers = (torch.arange(w, device=device, dtype=offsets.dtype) + 0.5) / w

        y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing='ij')

        # Stack as [2, H, W] with [x, y] order
        grid = torch.stack([x_grid, y_grid], dim=0)  # [2, H, W]

        # offsets is [B, 2, H, W], grid is [2, H, W]
        # offsets are in range [-0.5, 0.5] grid cells, convert to normalized coords
        offset_x = offsets[:, 0:1, :, :] / w  # Scale by grid width
        offset_y = offsets[:, 1:2, :, :] / h  # Scale by grid height
        offsets_normalized = torch.cat([offset_x, offset_y], dim=1)

        # Add grid centers + offsets
        points = grid.unsqueeze(0) + offsets_normalized  # [B, 2, H, W]

        # Reshape to [B, H*W, 2]
        points = points.flatten(2).transpose(1, 2)  # [B, N, 2]

        return points