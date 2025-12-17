import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianPointLoss(nn.Module):
    """
    Hungarian Matching Loss for P2P point detection.

    Key changes from original:
    - Works with normalized [0, 1] coordinates to ensure consistent loss scale
    - Uses L1 loss for regression (more stable than MSE for localization)
    - Balanced loss weighting
    """

    def __init__(self, cost_class=1.0, cost_point=5.0, focal_alpha=0.9, focal_gamma=2.0, reg_loss_weight=5.0):
        """
        Args:
            cost_class: Weight for classification cost in matching
            cost_point: Weight for point distance cost in matching
            focal_alpha: Focal loss alpha (class balance)
            focal_gamma: Focal loss gamma (hard example mining)
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def matcher(self, pred_logits, pred_points_norm, targets, image_size):
        """
        Hungarian matching between predictions and ground truth.

        Args:
            pred_logits: [B, N, C] classification logits
            pred_points_norm: [B, N, 2] predicted points in normalized [0,1] coords
            targets: list of dicts with 'points' (pixel coords) and 'labels'
            image_size: (H, W) of the input image
        """
        indices = []
        img_h, img_w = image_size

        # Softmax for class probability
        out_prob = pred_logits.softmax(-1)[:, :, 1]  # Prob of foreground class

        for b, tgt_dict in enumerate(targets):
            tgt_pts_tensor = tgt_dict['points']  # [M, 2] in (y, x) pixel format

            if len(tgt_pts_tensor) == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
                continue

            # Convert GT from (y, x) pixels to (x, y) normalized [0, 1]
            tgt_pts_yx = tgt_pts_tensor.float()
            tgt_pts_normalized = torch.zeros_like(tgt_pts_yx)
            tgt_pts_normalized[:, 0] = tgt_pts_yx[:, 1] / img_w  # x = col / width
            tgt_pts_normalized[:, 1] = tgt_pts_yx[:, 0] / img_h  # y = row / height

            # Ensure on same device
            tgt_pts_normalized = tgt_pts_normalized.to(pred_points_norm.device)

            # Cost Matrix Components:
            # 1. Classification cost: -probability of foreground
            cost_class = -out_prob[b].unsqueeze(1).expand(-1, len(tgt_pts_tensor))

            # 2. L1 distance in normalized coordinates (scale ~0-1)
            cost_dist = torch.cdist(pred_points_norm[b], tgt_pts_normalized, p=1)

            # Combined cost
            C = self.cost_class * cost_class + self.cost_point * cost_dist

            # Hungarian algorithm
            row_i, col_i = linear_sum_assignment(C.cpu().numpy())
            indices.append((torch.as_tensor(row_i, dtype=torch.long),
                            torch.as_tensor(col_i, dtype=torch.long)))

        return indices

    def forward(self, outputs, targets):
        """
        Compute the Hungarian matching loss.

        Args:
            outputs: dict with 'logits', 'points_normalized', 'image_size'
            targets: list of dicts with 'points' and 'labels'
        """
        # Get predictions
        pred_logits = outputs['logits'].flatten(2).transpose(1, 2)  # [B, N, C]
        pred_points_norm = outputs['points_normalized']  # [B, N, 2] in [0,1]
        image_size = outputs['image_size']  # (H, W)

        device = pred_logits.device
        B, N, C = pred_logits.shape
        img_h, img_w = image_size

        # Get Hungarian matching
        indices = self.matcher(pred_logits, pred_points_norm, targets, image_size)

        # Initialize target classes as background (class 0)
        target_classes = torch.zeros(B, N, dtype=torch.long, device=device)

        loss_reg = torch.tensor(0.0, device=device)
        n_objects = 0

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue

            # Set matched predictions to foreground class (1)
            target_classes[b, pred_idx] = 1

            # Get GT points and convert to normalized (x, y)
            tgt_pts_yx = targets[b]['points'].float()  # (y, x) pixel
            tgt_pts_normalized = torch.zeros_like(tgt_pts_yx)
            tgt_pts_normalized[:, 0] = tgt_pts_yx[:, 1] / img_w  # x
            tgt_pts_normalized[:, 1] = tgt_pts_yx[:, 0] / img_h  # y
            tgt_pts_normalized = tgt_pts_normalized.to(device)

            # Get matched predictions and targets
            matched_pred_pts = pred_points_norm[b, pred_idx]  # [K, 2]
            matched_tgt_pts = tgt_pts_normalized[tgt_idx]  # [K, 2]

            # L1 loss on normalized coordinates (more stable than MSE)
            loss_reg = loss_reg + F.l1_loss(matched_pred_pts, matched_tgt_pts, reduction='sum')
            n_objects += len(tgt_idx)

        # Average regression loss by number of objects
        if n_objects > 0:
            loss_reg = loss_reg / n_objects

        # Classification loss with focal loss for class imbalance
        loss_cls = self._focal_loss(pred_logits, target_classes)

        # Combined loss - regression now has similar scale to classification
        # Since both are in [0,1] range approximately
        total_loss = loss_cls + loss_reg * self.cost_point

        return total_loss

    def _focal_loss(self, pred_logits, target_classes):
        """
        Focal loss for classification with class imbalance handling.

        Args:
            pred_logits: [B, N, C] raw logits
            target_classes: [B, N] class indices (0=bg, 1=fg)
        """
        B, N, C = pred_logits.shape

        # Flatten for cross-entropy: [B*N, C] and [B*N]
        pred_flat = pred_logits.reshape(-1, C)
        target_flat = target_classes.reshape(-1)

        # Compute probabilities
        probs = F.softmax(pred_flat, dim=-1)

        # Get probability of true class
        ce_loss = F.cross_entropy(pred_flat, target_flat, reduction='none')

        # Get pt (probability of correct class)
        pt = probs.gather(1, target_flat.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.focal_gamma

        # Alpha weighting for class balance
        # alpha for positive class (1), (1-alpha) for negative class (0)
        alpha_t = torch.where(target_flat == 1,
                              self.focal_alpha,
                              1 - self.focal_alpha)

        # Final focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()