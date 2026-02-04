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

from typing import Optional

from .register import LOSSES
import torch
from torch import nn
from torch.nn import functional as F
@LOSSES.register()
class FocalLoss(torch.nn.Module):
    ''' Focal Loss module '''

    def __init__(
        self,
        alpha: int = 2,
        beta: int = 4,
        reduction: str = 'sum',
        weights: Optional[torch.Tensor] = None,
        density_weight: Optional[str] = None,
        normalize: bool = False,
        eps: float = 1e-6
        ) -> None:
        '''
        Args:
            alpha (int, optional): alpha parameter. Defaults to 2
            beta (int, optional): beta parameter. Defaults to 4
            reduction (str, optional): batch losses reduction. Possible
                values are 'sum' and 'mean'. Defaults to 'sum'
            weights (torch.Tensor, optional): channels weights, if specified
                must be a torch Tensor. Defaults to None
            density_weight (str, optional): to weight each sample by objects density
                (high factor for high density). Possible values are: 'linear', 'squared',
                or 'cubic' for choosing a linear, squared or cubic exponent to apply to
                the number of locations. Defaults to None
            normalize (bool, optional): set to True to normalize the loss according to
                the number of positive samples. Defaults to False
            eps (float, optional): for numerical stability. Defaults to 1e-6.
        '''

        super().__init__()

        assert reduction in ['mean', 'sum'], \
            f'Reduction must be either \'mean\' or \'sum\', got {reduction}'

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.weights = weights
        self.density_weight = density_weight
        self.normalize = normalize
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output (torch.Tensor): [B,C,H,W]
            target (torch.Tensor): [B,C,H,W]

        Returns:
            torch.Tensor
        '''

        return self._neg_loss(output, target)

    def _neg_loss(self, output: torch.Tensor, target: torch.Tensor):
        ''' Focal loss, adapted from CenterNet
        https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
        Which again is from CornerNet
        Args:
            output (torch.Tensor): [B,C,H,W]
            target (torch.Tensor): [B,C,H,W]

        Returns:
            torch.Tensor
        '''

        B, C, _, _ = target.shape

        if self.weights is not None:
            assert self.weights.shape[0] == C, \
                'Number of weights must match the number of channels, ' \
                    f'got {C} channels and {self.weights.shape[0]} weights'
        # This kind of restores the original point which was converted into a contiues distributin
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, self.beta)

        loss = torch.zeros((B, C))

         # avoid NaN when net output is 1.0 or 0.0
        output = torch.clamp(output, min=self.eps, max=1-self.eps)

        pos_loss = torch.log(output) * torch.pow(1 - output, self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, self.alpha) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum(3).sum(2)
        pos_loss = pos_loss.sum(3).sum(2)
        neg_loss = neg_loss.sum(3).sum(2)

        for b in range(B):
            for c in range(C):
                density = torch.tensor([1]).to(neg_loss.device)
                if self.density_weight == 'linear':
                    density = num_pos[b][c]
                elif self.density_weight == 'squared':
                    density = num_pos[b][c] ** 2
                elif self.density_weight == 'cubic':
                    density = num_pos[b][c] ** 3

                if num_pos[b][c] == 0:
                    loss[b][c] = loss[b][c] - neg_loss[b][c]
                else:
                    loss[b][c] = density * (loss[b][c] - (pos_loss[b][c] + neg_loss[b][c]))
                    if self.normalize:
                         loss[b][c] =  loss[b][c] / num_pos[b][c]

        if self.weights is not None:
            loss = self.weights * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()





@LOSSES.register()
class FastFocalLoss(nn.Module):
    '''
    Vectorized Focal Loss.
    Supports reduction='none' for OHEM.
    '''

    def __init__(self, alpha=2, beta=4, eps=1e-12, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, gt):
        '''
        pred:  [B, C, H, W] (Output of Sigmoid)
        gt:    [B, C, H, W] (Ground truth Gaussian heatmap)
        '''
        # 1. Clamp for stability
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)

        # 2. Define masks
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # 3. Calculate Elements
        neg_weights = torch.pow(1 - gt, self.beta)

        # 4. Calculate Pixel-wise Loss components
        # We calculate the loss map BEFORE summing

        # Positive loss: -log(p) * (1-p)^alpha
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds

        # Negative loss: -log(1-p) * p^alpha * (1-gt)^beta
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        # 5. Combine to full loss map [B, C, H, W]
        # Note: We return positive values for minimization (negative of the log likelihood)
        loss_map = -(pos_loss + neg_loss)

        # 6. Handle Reduction
        if self.reduction == 'none':
            return loss_map

        # For standard training (mean/sum), we normalize by Number of Positives
        num_pos = pos_inds.float().sum()

        if self.reduction == 'mean':
            if num_pos == 0:
                return loss_map.sum()  # Just background loss
            else:
                return loss_map.sum() / num_pos

        elif self.reduction == 'sum':
            return loss_map.sum()


@LOSSES.register()
class OHEMFocalLoss(nn.Module):
    def __init__(self, top_k_percent=0.2, alpha=2, beta=4):
        super().__init__()
        self.top_k_percent = top_k_percent

        # FIX: We HARDCODE reduction='none' here.
        # OHEM requires the full map to sort pixels.
        # It cannot accept a reduction argument from outside.
        self.focal = FastFocalLoss(alpha=alpha, beta=beta, reduction='none')

    def forward(self, pred, gt):
        # 1. Get full pixel-wise loss map [B, C, H, W]
        loss_map = self.focal(pred, gt)

        # Safety check to ensure we got a tensor back
        if loss_map is None:
            raise ValueError("FastFocalLoss returned None. Check reduction mode.")

        B, C, H, W = loss_map.shape
        num_pixels = H * W
        num_keep = int(num_pixels * self.top_k_percent)

        # 2. Flatten for sorting
        loss_flat = loss_map.view(B, -1)

        # 3. Determine Positives (We must ALWAYS keep these)
        # Assuming GT peaks are 1.0. If using Gaussian, > 0.9 is safer.
        pos_mask_flat = gt.view(B, -1).eq(1)
        num_pos = pos_mask_flat.float().sum()

        # 4. Determine Hard Negatives
        # Zero out positive losses in the copy so they aren't picked as "negatives"
        neg_loss_flat = loss_flat.clone()
        neg_loss_flat[pos_mask_flat] = 0

        # Sort and pick Top K hard negatives per image
        _, topk_indices = neg_loss_flat.topk(num_keep, dim=1)

        # Create Hard Negative Mask
        hard_neg_mask = torch.zeros_like(loss_flat, dtype=torch.bool)
        hard_neg_mask.scatter_(1, topk_indices, True)

        # 5. Combine Masks (Positives OR Hard Negatives)
        final_mask = pos_mask_flat | hard_neg_mask

        # 6. Select and Normalize
        selected_loss = loss_flat[final_mask]

        if selected_loss.numel() == 0:
            return loss_flat.sum() * 0.0

        # Normalize by the total number of objects (num_pos)
        if num_pos > 0:
            return selected_loss.sum() / num_pos
        else:
            return selected_loss.mean()


@LOSSES.register()
class HerdNetLoss(nn.Module):
    """
    Combines Focal Loss (Pixel-wise precision) and Dice Loss (Global shape).
    Automatically handles Channel Mismatches (e.g., 2-class GT vs 1-class Pred).
    """

    def __init__(self, alpha=2, beta=4, dice_weight=1.0):
        super().__init__()
        self.focal = FastFocalLoss(alpha=alpha, beta=beta, reduction='mean')
        self.dice_weight = dice_weight

    def forward(self, pred, gt):
        """
        pred: [B, 1, H, W] (Localization 'Objectness')
        gt:   [B, Num_Classes, H, W] (Class-specific heatmaps)
        """

        # --- CRITICAL FIX: Collapse GT Channels ---
        # If model predicts 1 channel (Objectness) but GT has multiple (Classes),
        # we merge GT channels via Max. (If ANY class is there, Objectness = 1)
        if pred.shape[1] == 1 and gt.shape[1] > 1:
            gt, _ = torch.max(gt, dim=1, keepdim=True)
        # ------------------------------------------

        # 1. Focal Loss (Pixel-wise)
        focal_l = self.focal(pred, gt)

        # 2. Soft Dice Loss (Global)
        eps = 1e-6
        B, C, H, W = pred.shape

        # Flatten spatial dims: [B, C, H, W] -> [B, C, H*W]
        pred_flat = pred.view(B, C, -1)
        gt_flat = gt.view(B, C, -1)

        # Intersection & Union
        intersection = (pred_flat * gt_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + gt_flat.sum(dim=2)

        # Dice Score: 2*Int / Union
        dice_score = (2. * intersection + eps) / (union + eps)

        # Average over Channels and Batch
        dice_l = (1 - dice_score).mean()

        return focal_l + (self.dice_weight * dice_l)


@LOSSES.register()
class DensityAwareFocalLoss(nn.Module):
    """
    Focal Loss with density-aware weighting.

    Isolated objects (low local density) receive higher weight than
    objects in dense colonies. This helps the model learn to detect
    harder-to-find lone individuals.

    Density is computed using a local window around each positive location.
    """

    def __init__(
            self,
            alpha: float = 2,
            beta: float = 4,
            density_radius: int = 32,
            min_weight: float = 0.5,
            max_weight: float = 3.0,
            density_scale: str = 'inverse',  # 'inverse', 'inverse_sqrt', 'inverse_log'
            eps: float = 1e-6,
            reduction: str = 'mean',
    ):
        """
        Args:
            alpha: Focal loss alpha (power for positive samples)
            beta: Focal loss beta (power for negative weight)
            density_radius: Radius (pixels) to compute local density
            min_weight: Minimum weight for high-density regions
            max_weight: Maximum weight for isolated objects
            density_scale: How to scale weight by density:
                - 'inverse': weight = 1 / density
                - 'inverse_sqrt': weight = 1 / sqrt(density)
                - 'inverse_log': weight = 1 / log(1 + density)
            eps: Numerical stability
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.density_radius = density_radius
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.density_scale = density_scale
        self.eps = eps
        self.reduction = reduction

        # Create density counting kernel (circular)
        kernel_size = 2 * density_radius + 1
        y, x = torch.meshgrid(
            torch.arange(kernel_size) - density_radius,
            torch.arange(kernel_size) - density_radius,
            indexing='ij'
        )
        kernel = ((x ** 2 + y ** 2) <= density_radius ** 2).float()
        kernel[density_radius, density_radius] = 0  # Don't count self
        self.register_buffer('density_kernel', kernel.unsqueeze(0).unsqueeze(0))

    def compute_density_map(self, gt: torch.Tensor) -> torch.Tensor:
        """
        Compute local density at each pixel.

        Args:
            gt: [B, C, H, W] ground truth heatmap

        Returns:
            density_map: [B, C, H, W] local object count at each pixel
        """
        B, C, H, W = gt.shape

        # Binary mask of object centers (peaks)
        pos_mask = gt.eq(1).float()

        # Move kernel to same device and dtype as input
        kernel = self.density_kernel.to(device=gt.device, dtype=gt.dtype)

        # Count neighbors using convolution
        padding = self.density_radius
        density_map = F.conv2d(
            pos_mask,
            kernel.expand(C, -1, -1, -1),
            padding=padding,
            groups=C
        )

        return density_map

    def compute_density_weights(self, gt: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel weights based on local density.
        Isolated objects get higher weights.

        Args:
            gt: [B, C, H, W] ground truth heatmap

        Returns:
            weights: [B, C, H, W] density-based weights
        """
        density_map = self.compute_density_map(gt)

        # Only apply density weighting to positive locations
        pos_mask = gt.eq(1).float()

        # Compute weight based on density scaling method
        # Add 1 to density to account for self (density=0 means truly isolated)
        local_density = density_map + 1  # Now isolated = 1, pair = 2, etc.

        if self.density_scale == 'inverse':
            raw_weight = 1.0 / local_density
        elif self.density_scale == 'inverse_sqrt':
            raw_weight = 1.0 / torch.sqrt(local_density)
        elif self.density_scale == 'inverse_log':
            raw_weight = 1.0 / torch.log1p(local_density)
        else:
            raise ValueError(f"Unknown density_scale: {self.density_scale}")

        # Normalize to [min_weight, max_weight] range
        # raw_weight is highest for isolated (density=1), lowest for dense
        raw_min = raw_weight[pos_mask.bool()].min() if pos_mask.sum() > 0 else torch.tensor(0.0)
        raw_max = raw_weight[pos_mask.bool()].max() if pos_mask.sum() > 0 else torch.tensor(1.0)

        if raw_max - raw_min > self.eps:
            normalized_weight = (raw_weight - raw_min) / (raw_max - raw_min + self.eps)
            weight = self.min_weight + normalized_weight * (self.max_weight - self.min_weight)
        else:
            weight = torch.ones_like(raw_weight) * (self.min_weight + self.max_weight) / 2

        # Apply weights only to positive locations, negatives get weight=1
        final_weights = torch.ones_like(gt)
        final_weights = torch.where(pos_mask.bool(), weight, final_weights)

        return final_weights

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] predicted heatmap (after sigmoid)
            gt: [B, C, H, W] ground truth heatmap

        Returns:
            loss: scalar tensor
        """
        # Clamp predictions for numerical stability
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)

        # Compute density-aware weights
        density_weights = self.compute_density_weights(gt)

        # Standard focal loss components
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, self.beta)

        # Positive loss: -log(p) * (1-p)^alpha * density_weight
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask * density_weights

        # Negative loss: -log(1-p) * p^alpha * (1-gt)^beta
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_mask

        # Combine (note: both are negative, so we negate)
        loss_map = -(pos_loss + neg_loss)

        if self.reduction == 'none':
            return loss_map

        # Normalize by weighted positive count
        num_pos = (pos_mask * density_weights).sum()

        if self.reduction == 'mean':
            if num_pos < self.eps:
                return loss_map.sum()
            return loss_map.sum() / num_pos
        elif self.reduction == 'sum':
            return loss_map.sum()


@LOSSES.register()
class DensityAwareHerdNetLoss(nn.Module):
    """
    Combines Density-Aware Focal Loss with Dice Loss.

    Isolated iguanas receive higher weight than those in dense colonies.
    """

    def __init__(
            self,
            alpha: float = 2,
            beta: float = 4,
            density_radius: int = 32,
            min_weight: float = 0.5,
            max_weight: float = 3.0,
            density_scale: str = 'inverse_sqrt',
            dice_weight: float = 1.0,
    ):
        super().__init__()
        self.focal = DensityAwareFocalLoss(
            alpha=alpha,
            beta=beta,
            density_radius=density_radius,
            min_weight=min_weight,
            max_weight=max_weight,
            density_scale=density_scale,
            reduction='mean',
        )
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] predicted heatmap
            gt: [B, C, H, W] ground truth (may have multiple classes)
        """
        # Collapse GT channels if needed
        if pred.shape[1] == 1 and gt.shape[1] > 1:
            gt, _ = torch.max(gt, dim=1, keepdim=True)

        # Density-aware focal loss
        focal_loss = self.focal(pred, gt)

        # Soft dice loss
        eps = 1e-6
        B, C, H, W = pred.shape

        pred_flat = pred.view(B, C, -1)
        gt_flat = gt.view(B, C, -1)

        intersection = (pred_flat * gt_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + gt_flat.sum(dim=2)

        dice_score = (2. * intersection + eps) / (union + eps)
        dice_loss = (1 - dice_score).mean()

        return focal_loss + self.dice_weight * dice_loss