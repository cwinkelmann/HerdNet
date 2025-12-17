"""
Simple Hungarian Loss for Debugging - V2

Handles both pixel and normalized GT coordinates robustly.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional


class MinimalHungarianLoss(nn.Module):
    def __init__(self, cost_class=2.0, cost_point=5.0, cls_weight=1.0, reg_weight=5.0, image_size=512, debug=False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.image_size = image_size
        self.debug = debug
        self._step = 0

    def _normalize_gt(self, pts):
        pts = pts.float()
        if pts.numel() == 0 or pts.max() <= 1.0:
            return pts
        return pts / self.image_size

    @torch.no_grad()
    def _match(self, logits, points, targets):
        probs = logits.softmax(-1)[:, :, 1]
        indices = []
        for b, tgt in enumerate(targets):
            gt_pts = tgt['points']
            if len(gt_pts) == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            gt_norm = self._normalize_gt(gt_pts).to(points.device)
            C_cls = -probs[b].unsqueeze(1).expand(-1, len(gt_pts))
            C_loc = torch.cdist(points[b], gt_norm, p=1)
            C = self.cost_class * C_cls + self.cost_point * C_loc
            row, col = linear_sum_assignment(C.cpu().numpy())
            indices.append((torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return indices

    def forward(self, outputs, targets):
        logits = outputs['logits']
        points = outputs['pred_points_normalized']
        device = logits.device
        B, N, C = logits.shape

        indices = self._match(logits, points, targets)
        cls_targets = torch.zeros(B, N, dtype=torch.long, device=device)

        total_reg_loss = 0.0
        n_matched = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) == 0:
                continue
            cls_targets[b, pred_idx] = 1
            gt_norm = self._normalize_gt(targets[b]['points']).to(device)
            total_reg_loss += F.l1_loss(points[b, pred_idx], gt_norm[gt_idx], reduction='sum')
            n_matched += len(gt_idx)

        reg_loss = total_reg_loss / max(n_matched, 1)

        pos_weight = min((B * N - n_matched) / max(n_matched, 1), 10.0)
        weights = torch.tensor([1.0, pos_weight], device=device)
        cls_loss = F.cross_entropy(logits.reshape(-1, C), cls_targets.reshape(-1), weight=weights)

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        self._step += 1

        if self.debug and self._step % 20 == 0:
            with torch.no_grad():
                fg_probs = logits.softmax(-1)[:, :, 1]
                matched_scores, unmatched_scores = [], []
                for b, (pred_idx, _) in enumerate(indices):
                    if len(pred_idx) > 0:
                        matched_scores.append(fg_probs[b, pred_idx].mean().item())
                        mask = torch.ones(N, dtype=torch.bool, device=device)
                        mask[pred_idx] = False
                        unmatched_scores.append(fg_probs[b, mask].mean().item())

                avg_m = np.mean(matched_scores) if matched_scores else 0
                avg_u = np.mean(unmatched_scores) if unmatched_scores else 0
                gap = avg_m - avg_u
                score_std = fg_probs.std().item()

                status = "✓" if gap > 0.1 else "⚠️" if gap > 0 else "❌"
                collapse = "COLLAPSE!" if score_std < 0.05 else ""

                print(f"\n[Step {self._step}] cls={cls_loss.item():.4f} reg={reg_loss.item():.4f} total={total.item():.4f}")
                print(f"  matched={n_matched} pos_weight={pos_weight:.1f}")
                print(f"  scores: matched={avg_m:.3f} unmatched={avg_u:.3f} gap={gap:.3f} {status}")
                print(f"  score_std={score_std:.4f} {collapse}")

        return total


"""
Hungarian Loss with Built-in Diagnostics

Use this to debug training issues.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional


class HungarianLossDebug(nn.Module):
    """
    Hungarian loss with extensive debugging for P2PNet training.

    Expected formats:
    - outputs['logits']: [B, N, num_classes] - classification logits
    - outputs['pred_points_normalized']: [B, N, 2] - points in [0, 1] as (x, y)
    - targets: List[Dict] with 'points' [M, 2] in pixels or normalized
    """

    def __init__(
            self,
            cost_class: float = 2.0,
            cost_point: float = 5.0,
            cls_weight: float = 1.0,
            reg_weight: float = 5.0,
            image_size: int = 512,
            debug: bool = True,
            debug_interval: int = 20,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.image_size = image_size
        self.debug = debug
        self.debug_interval = debug_interval
        self._step = 0

        # Track statistics
        self._best_gap = -999
        self._warnings = set()

    def _normalize_gt(self, pts: torch.Tensor) -> torch.Tensor:
        """Normalize GT points to [0, 1] if in pixel format."""
        pts = pts.float()
        if pts.numel() == 0:
            return pts
        if pts.max() > 1.0:
            return pts / self.image_size
        return pts

    def _get_points(self, outputs: Dict) -> torch.Tensor:
        """Get normalized points from outputs, handling different key names."""
        for key in ['pred_points_normalized', 'points_normalized', 'pred_points_norm']:
            if key in outputs:
                pts = outputs[key]
                if pts.max() <= 1.0:
                    return pts
                else:
                    if 'coord_not_normalized' not in self._warnings:
                        print(f"⚠️ WARNING: {key} appears to be in pixel format (max={pts.max():.1f})")
                        print(f"   Normalizing by image_size={self.image_size}")
                        self._warnings.add('coord_not_normalized')
                    return pts / self.image_size

        # Fallback to pixel coords
        if 'pred_points' in outputs or 'points' in outputs:
            pts = outputs.get('pred_points', outputs.get('points'))
            if 'using_pixel_coords' not in self._warnings:
                print(f"⚠️ WARNING: Using pixel coordinates, normalizing by {self.image_size}")
                self._warnings.add('using_pixel_coords')
            return pts / self.image_size

        raise KeyError(f"Could not find points in outputs. Keys: {list(outputs.keys())}")

    @torch.no_grad()
    def _match(
            self,
            logits: torch.Tensor,
            points: torch.Tensor,
            targets: List[Dict],
    ) -> List[tuple]:
        """Hungarian matching between predictions and GT."""
        probs = logits.softmax(-1)[:, :, 1]  # Foreground probability
        indices = []

        for b, tgt in enumerate(targets):
            gt_pts = tgt['points']

            if len(gt_pts) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            gt_norm = self._normalize_gt(gt_pts).to(points.device)

            # Cost matrix
            C_cls = -probs[b].unsqueeze(1).expand(-1, len(gt_pts))
            C_loc = torch.cdist(points[b], gt_norm, p=1)
            C = self.cost_class * C_cls + self.cost_point * C_loc

            row, col = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.as_tensor(row, dtype=torch.long),
                torch.as_tensor(col, dtype=torch.long)
            ))

        return indices

    def forward(self, outputs: Dict, targets: List[Dict]) -> torch.Tensor:
        """Compute loss with optional debugging."""
        logits = outputs['logits']
        points = self._get_points(outputs)

        device = logits.device
        B, N, C = logits.shape

        # Validate on first step
        if self._step == 0:
            self._validate_inputs(outputs, targets, points)

        # Hungarian matching
        indices = self._match(logits, points, targets)

        # Classification targets
        cls_targets = torch.zeros(B, N, dtype=torch.long, device=device)

        # Regression loss
        total_reg_loss = 0.0
        n_matched = 0
        match_distances = []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) == 0:
                continue

            cls_targets[b, pred_idx] = 1

            gt_norm = self._normalize_gt(targets[b]['points']).to(device)
            pred_pts = points[b, pred_idx]
            tgt_pts = gt_norm[gt_idx]

            # Track distances
            dists = (pred_pts - tgt_pts).abs().sum(dim=-1)
            match_distances.extend(dists.cpu().tolist())

            total_reg_loss += F.l1_loss(pred_pts, tgt_pts, reduction='sum')
            n_matched += len(gt_idx)

        reg_loss = total_reg_loss / max(n_matched, 1)

        # Classification loss with class weighting
        pos_weight = min((B * N - n_matched) / max(n_matched, 1), 10.0)
        weights = torch.tensor([1.0, pos_weight], device=device)

        cls_loss = F.cross_entropy(
            logits.reshape(-1, C),
            cls_targets.reshape(-1),
            weight=weights
        )

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        self._step += 1

        # Debug output
        if self.debug and self._step % self.debug_interval == 0:
            self._print_debug(
                logits, indices, cls_loss, reg_loss, total,
                n_matched, pos_weight, match_distances
            )

        return total

    def _validate_inputs(self, outputs: Dict, targets: List[Dict], points: torch.Tensor):
        """Validate input formats on first step."""
        print("\n" + "=" * 60)
        print("LOSS VALIDATION (first step)")
        print("=" * 60)

        print(f"\nOutputs keys: {list(outputs.keys())}")
        print(f"Pred points shape: {points.shape}")
        print(f"Pred points range: [{points.min():.3f}, {points.max():.3f}]")

        if points.max() > 1.0:
            print("❌ ERROR: Predicted points should be normalized to [0, 1]!")
        else:
            print("✓ Pred points are normalized")

        print(f"\nTargets: {len(targets)} samples")
        for i, tgt in enumerate(targets[:2]):
            print(f"\n  Target {i}:")
            print(f"    Keys: {list(tgt.keys())}")
            gt_pts = tgt['points']
            print(f"    Points shape: {gt_pts.shape}")
            if len(gt_pts) > 0:
                print(f"    Points range: [{gt_pts.min():.1f}, {gt_pts.max():.1f}]")
                print(f"    First 3: {gt_pts[:3].tolist()}")

                gt_norm = self._normalize_gt(gt_pts)
                print(f"    After normalization: {gt_norm[:3].tolist()}")

        print("\n" + "=" * 60 + "\n")

    def _print_debug(
            self,
            logits: torch.Tensor,
            indices: List[tuple],
            cls_loss: torch.Tensor,
            reg_loss: torch.Tensor,
            total: torch.Tensor,
            n_matched: int,
            pos_weight: float,
            match_distances: List[float],
    ):
        """Print debug information."""
        with torch.no_grad():
            B, N, _ = logits.shape
            fg_probs = logits.softmax(-1)[:, :, 1]
            device = logits.device

            # Compute score statistics
            matched_scores = []
            unmatched_scores = []

            for b, (pred_idx, _) in enumerate(indices):
                if len(pred_idx) > 0:
                    matched_scores.append(fg_probs[b, pred_idx].mean().item())
                    mask = torch.ones(N, dtype=torch.bool, device=device)
                    mask[pred_idx] = False
                    unmatched_scores.append(fg_probs[b, mask].mean().item())

            avg_matched = np.mean(matched_scores) if matched_scores else 0
            avg_unmatched = np.mean(unmatched_scores) if unmatched_scores else 0
            gap = avg_matched - avg_unmatched
            score_std = fg_probs.std().item()
            avg_match_dist = np.mean(match_distances) if match_distances else 0

            # Track best gap
            if gap > self._best_gap:
                self._best_gap = gap

            # Status indicators
            status = "✓" if gap > 0.3 else "⚠️" if gap > 0 else "❌"
            collapse = " COLLAPSE!" if score_std < 0.05 else ""

            print(f"\n[Step {self._step}] cls={cls_loss.item():.4f} reg={reg_loss.item():.4f} total={total.item():.4f}")
            print(f"  matched={n_matched} pos_weight={pos_weight:.1f} avg_match_dist={avg_match_dist:.4f}")
            print(f"  scores: matched={avg_matched:.3f} unmatched={avg_unmatched:.3f} gap={gap:.3f} {status}{collapse}")
            print(f"  score_std={score_std:.4f} best_gap={self._best_gap:.3f}")


class HungarianLoss(HungarianLossDebug):
    """
    Production version with debug off by default.

    Same as HungarianLossDebug but quieter.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('debug', False)
        super().__init__(**kwargs)


# Quick test function
def test_loss():
    """Quick test to verify loss works."""
    loss_fn = HungarianLossDebug(debug=True, debug_interval=1)

    # Fake outputs
    B, N, C = 2, 100, 2
    outputs = {
        'logits': torch.randn(B, N, C),
        'pred_points_normalized': torch.rand(B, N, 2),
    }

    # Fake targets (pixel format)
    targets = [
        {'points': torch.tensor([[100.0, 200.0], [300.0, 400.0]]), 'labels': torch.tensor([1, 1])},
        {'points': torch.tensor([[150.0, 250.0]]), 'labels': torch.tensor([1])},
    ]

    loss = loss_fn(outputs, targets)
    print(f"\nTest loss: {loss.item():.4f}")

    # Second step
    loss = loss_fn(outputs, targets)
    print(f"Test loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_loss()