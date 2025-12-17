"""
Simple Hungarian Loss for Debugging

No focal loss complexity - just plain CE + L1.
Easier to debug and verify the pipeline is working.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional


class SimpleHungarianLoss(nn.Module):
    """
    Simplified Hungarian Loss for debugging.

    - Plain cross-entropy (no focal weighting)
    - L1 regression loss
    - Optional auxiliary loss for unmatched queries
    """

    def __init__(
            self,
            cost_class: float = 1.0,
            cost_point: float = 5.0,
            reg_loss_weight: float = 2.0,
            aux_loss_weight: float = 1.0,
            gt_format: str = 'xy',
            debug: bool = False,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.reg_loss_weight = reg_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.gt_format = gt_format
        self.debug = debug

    def _get_tensor(self, outputs: Dict, *keys) -> Optional[torch.Tensor]:
        for key in keys:
            if key in outputs and outputs[key] is not None:
                return outputs[key]
        return None

    def _normalize_outputs(self, outputs: Dict) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        logits = self._get_tensor(outputs, 'logits', 'pred_logits')
        if logits is None:
            raise ValueError("No logits found")

        image_size = outputs.get('image_size')
        if isinstance(image_size, torch.Tensor):
            image_size = (image_size[0].item(), image_size[1].item())

        if logits.dim() == 4:  # Dense
            pred_logits = logits.flatten(2).transpose(1, 2)
            pred_points_norm = self._get_tensor(outputs, 'points_normalized', 'pred_points_normalized')
            if pred_points_norm.dim() == 4:
                pred_points_norm = pred_points_norm.flatten(2).transpose(1, 2)
        else:  # Sparse
            pred_logits = logits
            pred_points_norm = self._get_tensor(outputs, 'pred_points_normalized', 'points_normalized')

        return pred_logits, pred_points_norm, image_size

    def _normalize_gt_points(self, tgt_pts: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        tgt_pts = tgt_pts.float()
        tgt_pts_normalized = torch.zeros_like(tgt_pts)

        if self.gt_format == 'xy':
            tgt_pts_normalized[:, 0] = tgt_pts[:, 0] / img_w
            tgt_pts_normalized[:, 1] = tgt_pts[:, 1] / img_h
        else:
            tgt_pts_normalized[:, 0] = tgt_pts[:, 1] / img_w
            tgt_pts_normalized[:, 1] = tgt_pts[:, 0] / img_h

        return tgt_pts_normalized

    @torch.no_grad()
    def matcher(
            self,
            pred_logits: torch.Tensor,
            pred_points_norm: torch.Tensor,
            targets: List[Dict],
            image_size: Tuple[int, int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Hungarian matching."""
        indices = []
        img_h, img_w = image_size

        # Use softmax probability for foreground class
        out_prob = pred_logits.softmax(-1)[:, :, 1]  # [B, N]

        for b, tgt_dict in enumerate(targets):
            tgt_pts = tgt_dict['points']

            if len(tgt_pts) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            tgt_pts_norm = self._normalize_gt_points(tgt_pts, img_h, img_w)
            tgt_pts_norm = tgt_pts_norm.to(pred_points_norm.device)

            # Cost matrix: classification + localization
            cost_class = -out_prob[b].unsqueeze(1).expand(-1, len(tgt_pts))  # [N, M]
            cost_point = torch.cdist(pred_points_norm[b], tgt_pts_norm, p=1)  # [N, M]

            C = self.cost_class * cost_class + self.cost_point * cost_point

            row_idx, col_idx = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.long),
                torch.as_tensor(col_idx, dtype=torch.long)
            ))

        return indices

    def forward(self, outputs: Dict, targets: List[Dict]) -> torch.Tensor:
        pred_logits, pred_points_norm, image_size = self._normalize_outputs(outputs)

        device = pred_logits.device
        B, N, C = pred_logits.shape
        img_h, img_w = image_size

        # Match predictions to GT
        indices = self.matcher(pred_logits, pred_points_norm, targets, image_size)

        # Build classification targets (0 = background, 1 = foreground)
        target_classes = torch.zeros(B, N, dtype=torch.long, device=device)

        loss_reg = torch.tensor(0.0, device=device)
        n_matched = 0

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue

            # Set matched queries to foreground
            target_classes[b, pred_idx] = 1

            # Regression loss for matched pairs
            tgt_pts_norm = self._normalize_gt_points(
                targets[b]['points'], img_h, img_w
            ).to(device)

            matched_pred = pred_points_norm[b, pred_idx]
            matched_tgt = tgt_pts_norm[tgt_idx]

            loss_reg = loss_reg + F.l1_loss(matched_pred, matched_tgt, reduction='sum')
            n_matched += len(tgt_idx)

        # Normalize regression loss
        if n_matched > 0:
            loss_reg = loss_reg / n_matched

        # Classification loss - weighted to handle class imbalance
        # With 25 queries and ~5 GT, we have 5 positives and 20 negatives
        n_pos = max(n_matched, 1)
        n_neg = B * N - n_matched

        # Weight positive class higher
        pos_weight = n_neg / (n_pos + 1e-6)
        pos_weight = min(pos_weight, 10.0)  # Cap at 10x

        weight = torch.tensor([1.0, pos_weight], device=device)

        loss_cls = F.cross_entropy(
            pred_logits.reshape(-1, C),
            target_classes.reshape(-1),
            weight=weight,
        )

        # Auxiliary loss: push ALL predictions toward nearest GT
        loss_aux = self._auxiliary_loss(pred_points_norm, targets, image_size, indices)

        total_loss = loss_cls + loss_reg * self.reg_loss_weight + loss_aux * self.aux_loss_weight

        if self.debug:
            with torch.no_grad():
                probs = pred_logits.softmax(-1)
                fg_probs = probs[:, :, 1]
                print(f"\n[Loss Debug]")
                print(f"  Matched: {n_matched}, Pos weight: {pos_weight:.1f}")
                print(
                    f"  loss_cls: {loss_cls.item():.4f}, loss_reg: {loss_reg.item():.4f}, loss_aux: {loss_aux.item():.4f}")
                print(f"  FG prob range: [{fg_probs.min().item():.3f}, {fg_probs.max().item():.3f}]")

                # Check if matched queries have higher scores
                for b, (pred_idx, _) in enumerate(indices):
                    if len(pred_idx) > 0:
                        matched_scores = fg_probs[b, pred_idx].mean().item()
                        unmatched_mask = torch.ones(N, dtype=torch.bool, device=device)
                        unmatched_mask[pred_idx] = False
                        unmatched_scores = fg_probs[b, unmatched_mask].mean().item()
                        print(
                            f"  Batch {b}: matched_score={matched_scores:.3f}, unmatched_score={unmatched_scores:.3f}")

        return total_loss

    def _auxiliary_loss(
            self,
            pred_points_norm: torch.Tensor,
            targets: List[Dict],
            image_size: Tuple[int, int],
            indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Auxiliary loss: push ALL queries toward their nearest GT.

        This gives coordinate gradients to unmatched queries too.
        """
        B, N, _ = pred_points_norm.shape
        img_h, img_w = image_size
        device = pred_points_norm.device

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b, (matched_pred_idx, _) in enumerate(indices):
            tgt_pts = targets[b]['points']
            if len(tgt_pts) == 0:
                continue

            tgt_pts_norm = self._normalize_gt_points(tgt_pts, img_h, img_w).to(device)

            # Get unmatched queries
            unmatched_mask = torch.ones(N, dtype=torch.bool, device=device)
            if len(matched_pred_idx) > 0:
                unmatched_mask[matched_pred_idx.to(device)] = False

            if unmatched_mask.sum() == 0:
                continue

            # For each unmatched query, find nearest GT and compute loss
            unmatched_points = pred_points_norm[b, unmatched_mask]  # [M, 2]

            # Distance to all GT
            dists = torch.cdist(unmatched_points, tgt_pts_norm)  # [M, num_gt]

            # Get nearest GT
            min_dists, nearest_idx = dists.min(dim=1)  # [M]
            nearest_gt = tgt_pts_norm[nearest_idx]  # [M, 2]

            # Smooth L1 loss toward nearest GT
            loss = F.smooth_l1_loss(unmatched_points, nearest_gt, reduction='sum', beta=0.1)

            total_loss = total_loss + loss
            count += unmatched_mask.sum().item()

        if count > 0:
            return total_loss / count
        return total_loss