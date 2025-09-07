
import torch

from typing import Optional

from .register import LOSSES

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


@LOSSES.register()
class TverskyLoss(torch.nn.Module):
    ''' Tversky Loss module for recall/precision optimization '''

    def __init__(
            self,
            alpha: float = 0.3,
            beta: float = 0.7,
            gamma: float = 1.0,
            reduction: str = 'mean',
            weights: Optional[torch.Tensor] = None,
            density_weight: Optional[str] = None,
            normalize: bool = False,
            eps: float = 1e-6
    ) -> None:
        '''
        Args:
            alpha (float, optional): weight for false positives. Lower alpha = less FP penalty.
                Defaults to 0.3
            beta (float, optional): weight for false negatives. Higher beta = more FN penalty.
                For recall optimization, use beta > alpha. Defaults to 0.7
            gamma (float, optional): focal parameter to focus on hard examples. 
                gamma=1 is standard Tversky, gamma>1 adds focal behavior. Defaults to 1.0
            reduction (str, optional): batch losses reduction. Possible
                values are 'sum' and 'mean'. Defaults to 'mean'
            weights (torch.Tensor, optional): Not used in binary object detection mode. 
                Defaults to None
            density_weight (str, optional): to weight each sample by objects density 
                (high factor for high density). Possible values are: 'linear', 'squared', 
                or 'cubic' for choosing a linear, squared or cubic exponent to apply to
                the number of object locations. Defaults to None
            normalize (bool, optional): set to True to normalize the loss according to 
                the number of positive object pixels. Defaults to False
            eps (float, optional): for numerical stability. Defaults to 1e-6.
        '''

        super().__init__()

        assert reduction in ['mean', 'sum'], \
            f'Reduction must be either \'mean\' or \'sum\', got {reduction}'

        assert alpha >= 0 and beta >= 0, \
            f'Alpha and beta must be non-negative, got alpha={alpha}, beta={beta}'

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights
        self.density_weight = density_weight
        self.normalize = normalize
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output (torch.Tensor): [B,1,H,W] - binary object confidence (object vs no-object)
            target (torch.Tensor): [B,C,H,W] - multi-class ground truth 
                (automatically converted to binary object/no-object)

        Returns:
            torch.Tensor: Tversky loss for binary object detection
        '''

        return self._tversky_loss(output, target)

    def _tversky_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ''' Simple and robust Tversky loss for binary object detection

        Args:
            output (torch.Tensor): [B,1,H,W] - object confidence
            target (torch.Tensor): [B,C,H,W] - class targets

        Returns:
            torch.Tensor
        '''

        B, C, H, W = target.shape
        B_out, C_out, H_out, W_out = output.shape

        if C_out != 1:
            raise ValueError(f"Output must have 1 channel for binary detection, got {C_out}")

        # Convert multi-class target to binary: 1 if any class present, 0 otherwise
        binary_target = torch.clamp(target.sum(dim=1, keepdim=True), 0, 1)  # [B, 1, H, W]

        # Clamp output for numerical stability
        output = torch.clamp(output, min=self.eps, max=1 - self.eps)

        # Flatten for computation: [B, H*W]
        pred_flat = output.view(B, -1)
        target_flat = binary_target.view(B, -1)

        # Compute TP, FP, FN for each sample in batch
        tp = (pred_flat * target_flat).sum(dim=1)  # [B]
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)  # [B]
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)  # [B]

        # Tversky index: TP / (TP + α*FP + β*FN)
        denominator = tp + self.alpha * fp + self.beta * fn + self.eps
        tversky_index = tp / denominator

        # Tversky loss: 1 - Tversky index
        tversky_loss = 1.0 - tversky_index

        # Apply focal term if gamma != 1
        if self.gamma != 1.0:
            tversky_loss = torch.pow(tversky_loss, self.gamma)

        # Apply reduction
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()


@LOSSES.register()
class FocalTverskyLoss2(torch.nn.Module):
    ''' Focal Tversky Loss - combines Tversky loss with focal mechanism '''

    def __init__(
            self,
            alpha: float = 0.3,
            beta: float = 0.7,
            gamma: float = 2.0,
            reduction: str = 'mean',
            weights: Optional[torch.Tensor] = None,
            density_weight: Optional[str] = None,
            normalize: bool = False,
            eps: float = 1e-6
    ) -> None:
        '''
        Focal Tversky Loss with gamma > 1 for focusing on hard examples
        Designed for binary object detection (object vs no-object)

        Args:
            alpha (float, optional): weight for false positives. Defaults to 0.3
            beta (float, optional): weight for false negatives. For high recall, use beta > alpha. 
                Defaults to 0.7
            gamma (float, optional): focal parameter. Higher gamma focuses more on hard examples.
                Defaults to 2.0
            reduction (str, optional): batch losses reduction. Defaults to 'mean'
            weights (torch.Tensor, optional): Not used in binary detection mode. Defaults to None
            density_weight (str, optional): density weighting strategy. Defaults to None
            normalize (bool, optional): normalize by positive object pixels. Defaults to False
            eps (float, optional): numerical stability. Defaults to 1e-6
        '''

        super().__init__()

        # Use TverskyLoss as base with focal gamma
        self.tversky_loss = TverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            reduction=reduction,
            weights=weights,
            density_weight=density_weight,
            normalize=normalize,
            eps=eps
        )

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output (torch.Tensor): [B,1,H,W] - binary object confidence
            target (torch.Tensor): [B,C,H,W] - multi-class ground truth
                (automatically converted to binary object/no-object)

        Returns:
            torch.Tensor: Focal Tversky loss for binary object detection
        '''
        return self.tversky_loss(output, target)


@LOSSES.register()
class BinaryObjectLoss(torch.nn.Module):
    ''' Simple Binary Object Detection Loss with direct FN/FP control '''

    def __init__(
            self,
            fn_penalty: float = 5.0,
            fp_penalty: float = 1.0,
            reduction: str = 'mean',
            density_weight: Optional[str] = None,
            normalize: bool = False,
            eps: float = 1e-6
    ) -> None:
        '''
        Simple loss for binary object detection with direct control over FN vs FP penalty

        Args:
            fn_penalty (float, optional): penalty weight for false negatives (missed objects).
                Higher = more recall focus. Defaults to 5.0
            fp_penalty (float, optional): penalty weight for false positives (false alarms).
                Lower = less precision penalty. Defaults to 1.0
            reduction (str, optional): batch losses reduction. Defaults to 'mean'
            density_weight (str, optional): density weighting strategy. Defaults to None
            normalize (bool, optional): normalize by positive object pixels. Defaults to False
            eps (float, optional): numerical stability. Defaults to 1e-6
        '''

        super().__init__()

        assert reduction in ['mean', 'sum'], \
            f'Reduction must be either \'mean\' or \'sum\', got {reduction}'

        self.fn_penalty = fn_penalty
        self.fp_penalty = fp_penalty
        self.reduction = reduction
        self.density_weight = density_weight
        self.normalize = normalize
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output (torch.Tensor): [B,1,H,W] - binary object confidence
            target (torch.Tensor): [B,C,H,W] - multi-class ground truth
                (automatically converted to binary object/no-object)

        Returns:
            torch.Tensor: Binary object detection loss
        '''

        return self._binary_loss(output, target)

    def _binary_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ''' Simple and robust binary object detection loss '''

        B, C, H, W = target.shape
        B_out, C_out, H_out, W_out = output.shape

        if C_out != 1:
            raise ValueError(f"Output must have 1 channel for binary detection, got {C_out}")

        # Convert multi-class target to binary: 1 if any class present, 0 otherwise
        binary_target = torch.clamp(target.sum(dim=1, keepdim=True), 0, 1)  # [B, 1, H, W]

        # Clamp output for numerical stability
        output = torch.clamp(output, min=self.eps, max=1 - self.eps)

        # Compute false negatives and false positives
        false_negatives = binary_target * (1 - output)  # High when target=1, pred=low
        false_positives = (1 - binary_target) * output  # High when target=0, pred=high

        # Apply penalties
        fn_loss = self.fn_penalty * false_negatives.mean()
        fp_loss = self.fp_penalty * false_positives.mean()

        return fn_loss + fp_loss


@LOSSES.register()
class WeightedBCELoss(torch.nn.Module):
    ''' Weighted Binary Cross Entropy for binary object detection '''

    def __init__(
            self,
            pos_weight: float = 5.0,
            reduction: str = 'mean',
            eps: float = 1e-7
    ) -> None:
        '''
        Simple weighted BCE loss for binary object detection

        Args:
            pos_weight (float): weight for positive samples (objects). 
                Higher = more recall focus. Defaults to 5.0
            reduction (str): reduction method. Defaults to 'mean'
            eps (float): numerical stability. Defaults to 1e-7
        '''

        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output (torch.Tensor): [B,1,H,W] - object confidence
            target (torch.Tensor): [B,C,H,W] - multi-class targets

        Returns:
            torch.Tensor: Weighted BCE loss
        '''

        B, C, H, W = target.shape

        # Convert to binary target
        binary_target = torch.clamp(target.sum(dim=1, keepdim=True), 0, 1)

        # Standard BCE loss with weighting
        output = torch.clamp(output, self.eps, 1 - self.eps)

        # Weighted BCE: weight positive samples more heavily
        bce_loss = -(binary_target * self.pos_weight * torch.log(output) +
                     (1 - binary_target) * torch.log(1 - output))

        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss


@LOSSES.register()
class FocalTverskyLoss(torch.nn.Module):
    ''' Focal Tversky Loss - combines Tversky loss with focal mechanism '''

    def __init__(
            self,
            alpha: float = 0.3,
            beta: float = 0.7,
            gamma: float = 2.0,
            reduction: str = 'mean',
            weights: Optional[torch.Tensor] = None,
            density_weight: Optional[str] = None,
            normalize: bool = False,
            eps: float = 1e-6
    ) -> None:
        '''
        Focal Tversky Loss with gamma > 1 for focusing on hard examples

        Args:
            alpha (float, optional): weight for false positives. Defaults to 0.3
            beta (float, optional): weight for false negatives. For high recall, use beta > alpha.
                Defaults to 0.7
            gamma (float, optional): focal parameter. Higher gamma focuses more on hard examples.
                Defaults to 2.0
            reduction (str, optional): batch losses reduction. Defaults to 'mean'
            weights (torch.Tensor, optional): channels weights. Defaults to None
            density_weight (str, optional): density weighting strategy. Defaults to None
            normalize (bool, optional): normalize by positive samples. Defaults to False
            eps (float, optional): numerical stability. Defaults to 1e-6
        '''

        super().__init__()

        # Use TverskyLoss as base with focal gamma
        self.tversky_loss = TverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            reduction=reduction,
            weights=weights,
            density_weight=density_weight,
            normalize=normalize,
            eps=eps
        )

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output (torch.Tensor): [B,1,H,W] or [B,C,H,W] - model predictions
                If [B,1,H,W]: single confidence channel compared against each target class
                If [B,C,H,W]: channel-wise comparison with target
            target (torch.Tensor): [B,C,H,W] - ground truth

        Returns:
            torch.Tensor: Focal Tversky loss
        '''
        return self.tversky_loss(output, target)