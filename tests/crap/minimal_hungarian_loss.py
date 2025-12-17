"""
Minimal Hungarian Loss - No Conflicting Signals

The auxiliary loss in previous versions created a conflict:
- Classification: "unmatched queries should predict background"
- Aux loss: "unmatched queries should move toward GT"

This version removes that conflict. Diversity is maintained by:
1. Fixed reference points in the model (v4)
2. Only matched queries get coordinate gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional


class MinimalHungarianLoss(nn.Module):
    """
    Clean Hungarian loss without conflicting auxiliary objectives.
    
    For P2PNet with fixed reference points:
    - Matched queries: learn foreground + exact location
    - Unmatched queries: learn background only (no coord gradient)
    - Reference points prevent collapse
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_point: float = 5.0,
        cls_weight: float = 1.0,
        reg_weight: float = 5.0,
        gt_format: str = 'xy',
        image_size: Tuple[int, int] = (512, 512),
        debug: bool = False,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.gt_format = gt_format
        self.image_size = image_size
        self.debug = debug
        
        self._step = 0

    def _normalize_gt(self, pts: torch.Tensor) -> torch.Tensor:
        """Normalize pixel coords to [0,1]."""
        pts = pts.float()
        if pts.max() <= 1.0:
            return pts
        
        h, w = self.image_size
        out = torch.zeros_like(pts)
        if self.gt_format == 'xy':
            out[:, 0] = pts[:, 0] / w
            out[:, 1] = pts[:, 1] / h
        else:
            out[:, 0] = pts[:, 1] / w
            out[:, 1] = pts[:, 0] / h
        return out

    def _get_preds(self, outputs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract predictions from model output."""
        logits = outputs.get('logits', outputs.get('pred_logits'))
        points = outputs.get('pred_points_normalized', outputs.get('points_normalized'))
        
        if logits.dim() == 4:
            B, C, H, W = logits.shape
            logits = logits.permute(0, 2, 3, 1).reshape(B, H*W, C)
            points = points.permute(0, 2, 3, 1).reshape(B, H*W, 2)
        
        return logits, points

    @torch.no_grad()
    def _match(
        self,
        logits: torch.Tensor,
        points: torch.Tensor,
        targets: List[Dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Hungarian matching."""
        probs = logits.softmax(-1)[:, :, 1]  # FG probability
        indices = []
        
        for b, tgt in enumerate(targets):
            gt_pts = tgt['points']
            
            if len(gt_pts) == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                               torch.tensor([], dtype=torch.long)))
                continue
            
            gt_norm = self._normalize_gt(gt_pts).to(points.device)
            
            # Cost matrix
            C_cls = -probs[b].unsqueeze(1).expand(-1, len(gt_pts))
            C_loc = torch.cdist(points[b], gt_norm, p=1)
            C = self.cost_class * C_cls + self.cost_point * C_loc
            
            row, col = linear_sum_assignment(C.cpu().numpy())
            indices.append((torch.as_tensor(row, dtype=torch.long),
                           torch.as_tensor(col, dtype=torch.long)))
        
        return indices

    def forward(self, outputs: Dict, targets: List[Dict]) -> torch.Tensor:
        logits, points = self._get_preds(outputs)
        device = logits.device
        B, N, C = logits.shape
        
        # Match
        indices = self._match(logits, points, targets)
        
        # Build targets
        cls_targets = torch.zeros(B, N, dtype=torch.long, device=device)
        
        total_reg_loss = 0.0
        n_matched = 0
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) == 0:
                continue
            
            cls_targets[b, pred_idx] = 1
            
            gt_norm = self._normalize_gt(targets[b]['points']).to(device)
            
            pred_pts = points[b, pred_idx]
            tgt_pts = gt_norm[gt_idx]
            
            total_reg_loss += F.l1_loss(pred_pts, tgt_pts, reduction='sum')
            n_matched += len(gt_idx)
        
        # Regression loss (only matched)
        if n_matched > 0:
            reg_loss = total_reg_loss / n_matched
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        # Classification loss
        # Simple weighting: more weight on positives since they're rare
        n_pos = max(n_matched, 1)
        n_neg = B * N - n_matched
        pos_weight = min(n_neg / n_pos, 10.0)
        
        weights = torch.tensor([1.0, pos_weight], device=device)
        cls_loss = F.cross_entropy(
            logits.reshape(-1, C),
            cls_targets.reshape(-1),
            weight=weights
        )
        
        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        self._step += 1
        
        if self.debug and self._step % 10 == 0:
            with torch.no_grad():
                fg_probs = logits.softmax(-1)[:, :, 1]
                
                matched_scores = []
                unmatched_scores = []
                
                for b, (pred_idx, _) in enumerate(indices):
                    if len(pred_idx) > 0:
                        matched_scores.append(fg_probs[b, pred_idx].mean().item())
                        mask = torch.ones(N, dtype=torch.bool, device=device)
                        mask[pred_idx] = False
                        unmatched_scores.append(fg_probs[b, mask].mean().item())
                
                avg_matched = sum(matched_scores) / len(matched_scores) if matched_scores else 0
                avg_unmatched = sum(unmatched_scores) / len(unmatched_scores) if unmatched_scores else 0
                
                print(f"\n[Step {self._step}]")
                print(f"  cls_loss: {cls_loss.item():.4f}, reg_loss: {reg_loss.item():.4f}")
                print(f"  total: {total.item():.4f}")
                print(f"  matched: {n_matched}, pos_weight: {pos_weight:.1f}")
                print(f"  FG prob: [{fg_probs.min().item():.3f}, {fg_probs.max().item():.3f}]")
                print(f"  avg matched score: {avg_matched:.3f}, avg unmatched: {avg_unmatched:.3f}")
                
                # This is the key metric - should be positive and growing
                score_gap = avg_matched - avg_unmatched
                status = "✓" if score_gap > 0.1 else "⚠️" if score_gap > 0 else "❌"
                print(f"  score gap: {score_gap:.3f} {status}")
        
        return total


def overfit_test(model, dataloader, criterion, device, num_batches=5, epochs=100, lr=1e-4):
    """
    Overfitting test on a few batches.
    
    If the model can't overfit a few samples, something is fundamentally wrong.
    """
    import itertools
    
    # Get fixed batches
    batches = list(itertools.islice(dataloader, num_batches))
    
    # Count total GT
    total_gt = sum(len(t['points']) for _, targets in batches for t in targets)
    print(f"Overfitting on {num_batches} batches, {total_gt} total GT points")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for images, targets in batches:
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {epoch_loss/num_batches:.4f}")
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL OVERFITTING CHECK")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(batches):
            images = images.to(device)
            outputs = model(images)
            
            logits = outputs.get('logits', outputs.get('pred_logits'))
            points = outputs.get('pred_points_normalized', outputs.get('points_normalized'))
            
            probs = logits.softmax(-1)[:, :, 1]  # FG prob
            
            for b in range(len(targets)):
                gt_pts = targets[b]['points']
                if len(gt_pts) == 0:
                    continue
                
                # Normalize GT
                h, w = 512, 512
                gt_norm = torch.zeros_like(gt_pts.float())
                gt_norm[:, 0] = gt_pts[:, 0].float() / w
                gt_norm[:, 1] = gt_pts[:, 1].float() / h
                gt_norm = gt_norm.to(device)
                
                # Get top-k predictions (k = num_gt)
                k = len(gt_pts)
                top_scores, top_idx = probs[b].topk(k)
                top_pts = points[b, top_idx]
                
                # Check distances
                dists = torch.cdist(top_pts, gt_norm)
                min_dists = dists.min(dim=1)[0]  # Distance to nearest GT
                
                print(f"\nBatch {i}, Sample {b}: {len(gt_pts)} GT points")
                print(f"  Top-{k} scores: {top_scores.cpu().numpy().round(3)}")
                print(f"  Distance to nearest GT: {min_dists.cpu().numpy().round(3)}")
                print(f"  (Good if scores > 0.8 and distances < 0.05)")


# Quick test
if __name__ == "__main__":
    # Dummy test
    loss_fn = MinimalHungarianLoss(debug=True)
    
    # Fake outputs
    B, N, C = 2, 25, 2
    outputs = {
        'logits': torch.randn(B, N, C),
        'pred_points_normalized': torch.rand(B, N, 2),
    }
    
    # Fake targets
    targets = [
        {'points': torch.tensor([[100, 200], [300, 400]])},
        {'points': torch.tensor([[150, 250]])},
    ]
    
    loss = loss_fn(outputs, targets)
    print(f"Loss: {loss.item():.4f}")
