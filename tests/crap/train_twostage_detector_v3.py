"""
Two-Stage Point Detector: Heatmap Proposals + Point Refinement

Goal: Beat HerdNet (F1 ~0.80) on Fernandina

Architecture:
  Stage 1 - Proposal Generation (like HerdNet but simpler):
    - DINOv3 backbone
    - Lightweight decoder → heatmap
    - Local maxima → candidate points (high recall, lower precision)

  Stage 2 - Point Refinement (IMPROVED):
    - Multi-scale ROI features (local texture)
    - Global context (scene understanding)
    - Positional encoding (spatial relationships)
    - Point transformer: refine using full context
    - Per-point classification (true positive vs false positive)
    - Per-point offset regression (sub-pixel refinement)
    - Hard negative mining (focus on difficult examples)

Why this beats HerdNet:
  1. Stage 1 catches most objects (high recall like HerdNet)
  2. Stage 2 filters false positives with rich context (better precision)
  3. Stage 2 refines locations (better localization)
  4. Hard negative mining (learns from mistakes)
  5. End-to-end trainable

Training:
  - Stage 1: MSE loss on heatmap
  - Stage 2: BCE + hard negative mining + L1 loss on proposals matched to GT
"""

import os
import argparse
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALB = True
except ImportError:
    HAS_ALB = False


# =============================================================================
# DATASET
# =============================================================================

class PointDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, image_size: int = 512, augment: bool = False):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()
        self.annotations = {n: g[['x', 'y']].values.astype(np.float32)
                            for n, g in self.df.groupby('images')}

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.transform = self._build_transform()

        print(f"Loaded {len(self.image_names)} images, {sum(len(v) for v in self.annotations.values())} points")

    def _build_transform(self):
        if not HAS_ALB:
            return None
        base = [A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()), ToTensorV2()]
        if self.augment:
            base = [A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(0.2, 0.2, p=0.3)] + base[-2:]
        return A.Compose(base, keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img = np.array(Image.open(os.path.join(self.image_dir, name)).convert('RGB'))
        pts = self.annotations[name].copy()

        h, w = img.shape[:2]
        pts[:, 0] *= self.image_size / w
        pts[:, 1] *= self.image_size / h

        if self.transform:
            r = self.transform(image=img, keypoints=[(p[0], p[1]) for p in pts])
            img = r['image']
            pts = np.array(r['keypoints'], dtype=np.float32) if r['keypoints'] else np.zeros((0, 2))

        if len(pts) > 0:
            valid = (pts[:, 0] >= 0) & (pts[:, 0] < self.image_size) & \
                    (pts[:, 1] >= 0) & (pts[:, 1] < self.image_size)
            pts = pts[valid]

        return img, {'points': torch.from_numpy(pts).float(), 'name': name}


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# =============================================================================
# STAGE 1: HEATMAP PROPOSAL NETWORK
# =============================================================================

class HeatmapProposalNet(nn.Module):
    """
    Stage 1: Generate candidate points via heatmap.
    Simple and fast - just needs high recall.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, output_size: int = 128):
        super().__init__()

        self.output_size = output_size

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Adaptive pool to exact output size
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        # Heatmap head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns heatmap logits (B, H, W)"""
        x = self.decoder(feat)
        x = self.pool(x)
        return self.head(x).squeeze(1)


# =============================================================================
# STAGE 2: POINT REFINEMENT NETWORK (IMPROVED)
# =============================================================================

class PointRefinementNet(nn.Module):
    """
    Stage 2: Refine candidate points with multi-scale + global context.

    For each candidate:
    1. Extract multi-scale local features via bilinear sampling
    2. Add global context (scene understanding)
    3. Add positional encoding (spatial awareness)
    4. Apply transformer for point relationships
    5. Classify (TP vs FP) and regress offset
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 256,
                 roi_size: int = 11, n_layers: int = 4):
        super().__init__()

        self.roi_size = roi_size
        self.hidden_dim = hidden_dim

        # Project backbone features
        self.feat_proj = nn.Conv2d(feat_dim, hidden_dim, 1)

        # Multi-scale ROI sizes
        self.roi_sizes = [7, 11, 15]  # Small, medium, large context

        # Calculate actual output size per scale
        self.per_scale_dim = hidden_dim // len(self.roi_sizes)
        self.concat_dim = self.per_scale_dim * len(self.roi_sizes)

        # ROI embeddings for each scale
        self.roi_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * size * size, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.per_scale_dim),
            ) for size in self.roi_sizes
        ])

        # Fusion of multi-scale features
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Global context branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Positional encoding - encode point coordinates
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        # Point transformer - let points attend to each other
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head (TP vs FP)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),  # Binary: is this a true point?
        )

        # Offset head (refine location)
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),  # dx, dy offset
        )

        # Auxiliary detection head (helps learn discriminative features)
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def extract_roi_features(self, feat: torch.Tensor, points: torch.Tensor,
                            roi_size: int) -> torch.Tensor:
        """
        Extract ROI features around each point at a specific scale.

        Args:
            feat: (B, C, H, W) feature map
            points: (B, N, 2) normalized point coordinates [0, 1]
            roi_size: size of ROI to extract

        Returns:
            roi_feat: (B, N, C*roi_size*roi_size)
        """
        B, C, H, W = feat.shape
        N = points.shape[1]
        device = feat.device

        if N == 0:
            return torch.zeros(B, 0, C * roi_size * roi_size, device=device)

        # Create sampling grid around each point
        roi_half = roi_size // 2
        offsets = torch.linspace(-roi_half, roi_half, roi_size, device=device)
        offsets = offsets / (H / 2)  # Scale to normalized coords

        oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
        offset_grid = torch.stack([ox, oy], dim=-1)  # (roi_size, roi_size, 2)

        # Expand points and add offsets
        points_exp = points[:, :, None, None, :]  # (B, N, 1, 1, 2)
        sample_grid = points_exp + offset_grid[None, None, :, :, :]  # (B, N, roi, roi, 2)

        # Convert to grid_sample format: [-1, 1]
        sample_grid = sample_grid * 2 - 1

        # Sample features for all points at once
        sample_grid_flat = sample_grid.view(B, N * roi_size * roi_size, 1, 2)

        sampled = F.grid_sample(feat, sample_grid_flat, mode='bilinear',
                                padding_mode='border', align_corners=False)
        # sampled: (B, C, N*roi*roi, 1)

        sampled = sampled.squeeze(-1).view(B, C, N, roi_size * roi_size)
        sampled = sampled.permute(0, 2, 3, 1)  # (B, N, roi*roi, C)
        sampled = sampled.reshape(B, N, -1)  # (B, N, roi*roi*C)

        return sampled

    def forward(self, feat: torch.Tensor, points: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat: (B, C, H, W) backbone features
            points: (B, N, 2) candidate points in normalized coords [0, 1]
            mask: (B, N) bool mask, True = valid point

        Returns:
            cls_logits: (B, N) classification logits
            offsets: (B, N, 2) predicted offsets
            aux_logits: (B, N) auxiliary detection logits
        """
        B, N = points.shape[:2]
        device = feat.device

        if N == 0:
            return {
                'cls_logits': torch.zeros(B, 0, device=device),
                'offsets': torch.zeros(B, 0, 2, device=device),
                'aux_logits': torch.zeros(B, 0, device=device),
            }

        # Project features
        feat_proj = self.feat_proj(feat)

        # Extract multi-scale ROI features
        multiscale_features = []
        for roi_size, roi_embed in zip(self.roi_sizes, self.roi_embeds):
            roi_raw = self.extract_roi_features(feat_proj, points, roi_size)
            roi_feat = roi_embed(roi_raw)
            multiscale_features.append(roi_feat)

        # Concatenate and fuse multi-scale features
        roi_feat = torch.cat(multiscale_features, dim=-1)  # (B, N, hidden_dim)
        roi_feat = self.scale_fusion(roi_feat)

        # Add global context
        global_feat = self.global_pool(feat).squeeze(-1).squeeze(-1)  # (B, C)
        global_feat = self.global_proj(global_feat)  # (B, hidden_dim)
        global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)

        # Add positional encoding
        pos_feat = self.pos_encoder(points)  # (B, N, hidden_dim)

        # Combine: Local (multi-scale) + Global (scene) + Position (spatial)
        combined_feat = roi_feat + global_feat + pos_feat

        # Auxiliary prediction before transformer (helps learn discriminative features)
        aux_logits = self.aux_head(combined_feat).squeeze(-1)  # (B, N)

        # Transform with attention
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        refined = self.transformer(combined_feat, src_key_padding_mask=attn_mask)

        # Final predictions
        cls_logits = self.cls_head(refined).squeeze(-1)  # (B, N)
        offsets = self.offset_head(refined)  # (B, N, 2)
        offsets = torch.tanh(offsets) * 0.2  # Limit offset to ±20% of image

        return {
            'cls_logits': cls_logits,
            'offsets': offsets,
            'aux_logits': aux_logits
        }


# =============================================================================
# FULL MODEL
# =============================================================================

class TwoStagePointDetector(nn.Module):
    """
    Two-stage point detector:
    1. Heatmap-based proposal generation
    2. Point-wise refinement with multi-scale + global context
    """

    def __init__(self, backbone: str = 'vit_large_patch16_dinov3.sat493m',
                 freeze_backbone: bool = True, heatmap_size: int = 128,
                 max_proposals: int = 300, proposal_threshold: float = 0.1,
                 refine_hidden: int = 512, roi_size: int = 11):
        super().__init__()

        self.heatmap_size = heatmap_size
        self.max_proposals = max_proposals
        self.proposal_threshold = proposal_threshold

        # Backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            feat = self.backbone.forward_features(dummy)
            self.feat_dim = feat.shape[-1]
            self.num_prefix = getattr(self.backbone, 'num_prefix_tokens', 1)
            n_tokens = feat.shape[1] - self.num_prefix
            self.spatial_size = int(np.sqrt(n_tokens))

        print(f"Backbone: {backbone}")
        print(f"  Features: {self.feat_dim}d, {self.spatial_size}x{self.spatial_size}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Frozen")

        # Stage 1: Heatmap proposals
        self.stage1 = HeatmapProposalNet(self.feat_dim, hidden_dim=128, output_size=heatmap_size)

        # Stage 2: Point refinement with multi-scale + global context
        self.stage2 = PointRefinementNet(self.feat_dim, hidden_dim=refine_hidden, roi_size=roi_size)

        # Stride for coordinate conversion
        self.stride = 512 // heatmap_size  # 4 for heatmap_size=128

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features."""
        B = x.shape[0]
        feat = self.backbone.forward_features(x)
        feat = feat[:, self.num_prefix:, :]
        feat = feat.view(B, self.spatial_size, self.spatial_size, self.feat_dim)
        return feat.permute(0, 3, 1, 2)  # (B, C, H, W)

    def generate_proposals(self, heatmap: torch.Tensor,
                           threshold: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate point proposals from heatmap via local maxima.

        Returns:
            points: (B, N, 2) normalized coordinates
            scores: (B, N) confidence scores
            mask: (B, N) valid mask
        """
        if threshold is None:
            threshold = self.proposal_threshold

        B, H, W = heatmap.shape
        device = heatmap.device

        prob = torch.sigmoid(heatmap)

        # Local maximum detection
        pad = 1
        prob_pad = F.pad(prob.unsqueeze(1), [pad] * 4, mode='replicate')
        local_max = F.max_pool2d(prob_pad, 3, stride=1).squeeze(1)

        # Points are local maxima above threshold
        is_peak = (prob == local_max) & (prob >= threshold)

        # Extract top-k proposals per image
        all_points = []
        all_scores = []
        all_masks = []

        for b in range(B):
            peaks = is_peak[b]
            if not peaks.any():
                # No proposals - add dummy
                pts = torch.zeros(self.max_proposals, 2, device=device)
                scores = torch.zeros(self.max_proposals, device=device)
                mask = torch.zeros(self.max_proposals, dtype=torch.bool, device=device)
            else:
                y_idx, x_idx = torch.where(peaks)
                scores_b = prob[b, y_idx, x_idx]

                # Sort by score and take top-k
                n_peaks = len(scores_b)
                if n_peaks > self.max_proposals:
                    topk_idx = scores_b.argsort(descending=True)[:self.max_proposals]
                    y_idx, x_idx = y_idx[topk_idx], x_idx[topk_idx]
                    scores_b = scores_b[topk_idx]

                # Convert to normalized coordinates
                pts_x = (x_idx.float() + 0.5) / W
                pts_y = (y_idx.float() + 0.5) / H
                pts = torch.stack([pts_x, pts_y], dim=1)  # (n, 2)

                # Pad to max_proposals
                n = len(pts)
                if n < self.max_proposals:
                    pts = F.pad(pts, [0, 0, 0, self.max_proposals - n])
                    scores_b = F.pad(scores_b, [0, self.max_proposals - n])

                scores = scores_b
                mask = torch.zeros(self.max_proposals, dtype=torch.bool, device=device)
                mask[:n] = True

            all_points.append(pts)
            all_scores.append(scores)
            all_masks.append(mask)

        return torch.stack(all_points), torch.stack(all_scores), torch.stack(all_masks)

    def forward(self, x: torch.Tensor,
                return_proposals: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images
            return_proposals: if True, also return stage 1 proposals

        Returns:
            heatmap: (B, H, W) heatmap logits
            points: (B, N, 2) refined point coordinates (normalized)
            scores: (B, N) final confidence scores
            mask: (B, N) valid mask
        """
        # Extract features
        feat = self.extract_features(x)  # (B, C, H, W)

        # Upsample features to match heatmap size for stage 2 ROI extraction
        feat_up = F.interpolate(feat, size=self.heatmap_size, mode='bilinear', align_corners=False)

        # Stage 1: Heatmap
        heatmap = self.stage1(feat)  # (B, heatmap_size, heatmap_size)

        # Generate proposals
        proposals, prop_scores, prop_mask = self.generate_proposals(heatmap)

        # Stage 2: Refine proposals
        stage2_out = self.stage2(feat_up, proposals, prop_mask)

        # Combine scores and apply offset
        cls_logits = stage2_out['cls_logits']
        offsets = stage2_out['offsets']
        aux_logits = stage2_out['aux_logits']

        # Final points = proposals + offsets
        final_points = proposals + offsets
        final_points = final_points.clamp(0, 1)

        # Final score = stage1_score * sigmoid(stage2_logit)
        # This way stage 2 can only reduce confidence, not create new detections
        final_scores = prop_scores * torch.sigmoid(cls_logits)

        out = {
            'heatmap': heatmap,
            'points': final_points,
            'scores': final_scores,
            'mask': prop_mask,
            'stage2_logits': cls_logits,
            'offsets': offsets,
            'aux_logits': aux_logits,
        }

        if return_proposals:
            out['proposals'] = proposals
            out['proposal_scores'] = prop_scores

        return out

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# =============================================================================
# LOSS WITH HARD NEGATIVE MINING
# =============================================================================

class TwoStageLoss(nn.Module):
    """
    Combined loss for two-stage detector:
    - Stage 1: MSE loss on heatmap (Gaussian targets)
    - Stage 2: BCE + hard negative mining + auxiliary loss + L1 loss
    """

    def __init__(self, heatmap_sigma: float = 2.0,
                 stage1_weight: float = 1.0, stage2_weight: float = 1.0,
                 cls_weight: float = 1.0, offset_weight: float = 5.0,
                 aux_weight: float = 0.1, match_radius: float = 0.08,
                 pos_weight: float = 3.0):
        super().__init__()

        self.heatmap_sigma = heatmap_sigma
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight
        self.cls_weight = cls_weight
        self.offset_weight = offset_weight
        self.aux_weight = aux_weight
        self.match_radius = match_radius
        self.pos_weight = pos_weight
        self._step = 0

    def generate_heatmap_target(self, points: torch.Tensor, H: int, W: int,
                                device: torch.device) -> torch.Tensor:
        """Generate Gaussian heatmap target."""
        heatmap = torch.zeros(H, W, device=device)
        if len(points) == 0:
            return heatmap

        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        for pt in points:
            px, py = pt[0].item() * W, pt[1].item() * H
            gaussian = torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * self.heatmap_sigma ** 2))
            heatmap = torch.maximum(heatmap, gaussian)

        return heatmap

    def match_proposals_to_gt(self, proposals: torch.Tensor, gt_points: torch.Tensor,
                              mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match proposals to GT points.

        Returns:
            labels: (N,) 1 for matched, 0 for unmatched
            matched_gt: (N, 2) matched GT point (or zeros if unmatched)
            matched_mask: (N,) True if this proposal has a GT match
        """
        N = len(proposals)
        device = proposals.device

        labels = torch.zeros(N, device=device)
        matched_gt = torch.zeros(N, 2, device=device)
        matched_mask = torch.zeros(N, dtype=torch.bool, device=device)

        if len(gt_points) == 0 or not mask.any():
            return labels, matched_gt, matched_mask

        # Only consider valid proposals
        valid_idx = torch.where(mask)[0]
        valid_proposals = proposals[valid_idx]

        if len(valid_proposals) == 0:
            return labels, matched_gt, matched_mask

        # Distance matrix
        dists = torch.cdist(valid_proposals, gt_points)  # (n_valid, n_gt)

        # Greedy matching
        matched_gt_idx = set()

        for i in range(len(valid_proposals)):
            min_dist, gt_idx = dists[i].min(dim=0)
            gt_idx = gt_idx.item()

            if min_dist < self.match_radius and gt_idx not in matched_gt_idx:
                prop_idx = valid_idx[i].item()
                labels[prop_idx] = 1
                matched_gt[prop_idx] = gt_points[gt_idx]
                matched_mask[prop_idx] = True
                matched_gt_idx.add(gt_idx)

                # Remove this GT from future matching
                dists[:, gt_idx] = float('inf')

        return labels, matched_gt, matched_mask

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: List[Dict]) -> Tuple[torch.Tensor, Dict]:

        heatmap = outputs['heatmap']
        proposals = outputs['points']
        prop_mask = outputs['mask']
        cls_logits = outputs['stage2_logits']
        offsets = outputs['offsets']
        aux_logits = outputs['aux_logits']

        B, H, W = heatmap.shape
        device = heatmap.device

        # === Stage 1 Loss: Heatmap MSE ===
        heatmap_prob = torch.sigmoid(heatmap)
        stage1_loss = 0.0

        for b in range(B):
            gt_pts = targets[b]['points'].to(device)
            gt_norm = gt_pts / 512  # Normalize to [0, 1]

            target_hm = self.generate_heatmap_target(gt_norm, H, W, device)

            # Weighted MSE (more weight on positive regions)
            weight = torch.ones_like(target_hm)
            weight[target_hm > 0.1] = 10.0

            loss = weight * (heatmap_prob[b] - target_hm) ** 2
            stage1_loss += loss.mean()

        stage1_loss = stage1_loss / B

        # === Stage 2 Loss: Classification + Auxiliary + Offset ===
        stage2_cls_loss = 0.0
        stage2_aux_loss = 0.0
        stage2_off_loss = 0.0
        n_matched = 0
        n_proposals = 0
        n_hard_negatives = 0

        for b in range(B):
            gt_pts = targets[b]['points'].to(device)
            gt_norm = gt_pts / 512

            labels, matched_gt, matched_mask = self.match_proposals_to_gt(
                proposals[b], gt_norm, prop_mask[b]
            )

            # Classification loss on valid proposals with hard negative mining
            valid = prop_mask[b]
            if valid.any():
                # Hard negative mining: identify difficult false positives
                with torch.no_grad():
                    pred_probs = torch.sigmoid(cls_logits[b, valid])
                    is_negative = (labels[valid] == 0)
                    hard_negatives = is_negative & (pred_probs > 0.3)  # FPs with >30% confidence
                    n_hard_negatives += hard_negatives.sum().item()

                    # Sample weights: emphasize hard negatives
                    sample_weight = torch.ones_like(labels[valid])
                    sample_weight[hard_negatives] = 2.5  # 2.5x weight on hard FPs

                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_logits[b, valid], labels[valid],
                    weight=sample_weight,
                    pos_weight=torch.tensor(self.pos_weight, device=device)
                )
                stage2_cls_loss += cls_loss
                n_proposals += valid.sum().item()

                # Auxiliary loss (before transformer refinement)
                aux_loss = F.binary_cross_entropy_with_logits(
                    aux_logits[b, valid], labels[valid],
                    pos_weight=torch.tensor(self.pos_weight, device=device)
                )
                stage2_aux_loss += aux_loss

            # Offset loss on matched proposals only
            if matched_mask.any():
                pred_pts = proposals[b, matched_mask] + offsets[b, matched_mask]
                gt_matched = matched_gt[matched_mask]
                off_loss = F.smooth_l1_loss(pred_pts, gt_matched)
                stage2_off_loss += off_loss
                n_matched += matched_mask.sum().item()

        stage2_cls_loss = stage2_cls_loss / B if n_proposals > 0 else torch.tensor(0.0, device=device)
        stage2_aux_loss = stage2_aux_loss / B if n_proposals > 0 else torch.tensor(0.0, device=device)
        stage2_off_loss = stage2_off_loss / B if n_matched > 0 else torch.tensor(0.0, device=device)

        stage2_loss = (self.cls_weight * stage2_cls_loss +
                      self.aux_weight * stage2_aux_loss +
                      self.offset_weight * stage2_off_loss)

        total = self.stage1_weight * stage1_loss + self.stage2_weight * stage2_loss

        # Logging
        self._step += 1
        if self._step % 50 == 0:
            with torch.no_grad():
                hm_max = heatmap_prob.max().item()
                final_scores = outputs['scores']
                score_max = final_scores[prop_mask].max().item() if prop_mask.any() else 0

                print(
                    f"  [Step {self._step}] s1={stage1_loss.item():.4f} "
                    f"s2_cls={stage2_cls_loss.item() if isinstance(stage2_cls_loss, torch.Tensor) else stage2_cls_loss:.4f} "
                    f"s2_aux={stage2_aux_loss.item() if isinstance(stage2_aux_loss, torch.Tensor) else stage2_aux_loss:.4f} "
                    f"s2_off={stage2_off_loss.item() if isinstance(stage2_off_loss, torch.Tensor) else stage2_off_loss:.4f} | "
                    f"hm_max={hm_max:.3f} score_max={score_max:.3f} matched={n_matched} hard_neg={n_hard_negatives}")

        return total, {
            'stage1_loss': stage1_loss.item(),
            'stage2_cls_loss': stage2_cls_loss.item() if isinstance(stage2_cls_loss, torch.Tensor) else 0,
            'stage2_aux_loss': stage2_aux_loss.item() if isinstance(stage2_aux_loss, torch.Tensor) else 0,
            'stage2_off_loss': stage2_off_loss.item() if isinstance(stage2_off_loss, torch.Tensor) else 0,
            'n_matched': n_matched,
            'n_hard_neg': n_hard_negatives,
        }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, dataloader, device, threshold: float = 0.3,
             match_radius: float = 25, image_size: int = 512) -> Dict:
    """Evaluate model."""
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    all_max_scores = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            for b in range(len(targets)):
                gt_pts = targets[b]['points'].to(device)

                scores = outputs['scores'][b]
                mask = outputs['mask'][b]
                points = outputs['points'][b]

                # Record max score
                if mask.any():
                    all_max_scores.append(scores[mask].max().item())

                # Filter by threshold
                keep = mask & (scores >= threshold)
                pred_pts = points[keep] * image_size
                pred_scores = scores[keep]

                # Match
                n_pred, n_gt = len(pred_pts), len(gt_pts)
                matched_gt, matched_pred = set(), set()

                if n_pred > 0 and n_gt > 0:
                    dists = torch.cdist(pred_pts, gt_pts)
                    for idx in dists.flatten().argsort():
                        if dists.flatten()[idx] > match_radius:
                            break
                        pi, gi = (idx // n_gt).item(), (idx % n_gt).item()
                        if pi not in matched_pred and gi not in matched_gt:
                            matched_pred.add(pi)
                            matched_gt.add(gi)

                total_tp += len(matched_pred)
                total_fp += n_pred - len(matched_pred)
                total_fn += n_gt - len(matched_gt)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'avg_max_score': np.mean(all_max_scores) if all_max_scores else 0
    }


def threshold_sweep(model, dataloader, device, match_radius: float = 25):
    """Find optimal threshold."""
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP")
    print("=" * 60)
    print(f"{'Thresh':>7} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 55)

    best_f1, best_t = 0, 0.3
    for t in np.arange(0.05, 0.95, 0.05):
        m = evaluate(model, dataloader, device, t, match_radius)
        marker = " ★" if m['f1'] > best_f1 else ""
        print(f"{t:>7.2f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
              f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}{marker}")
        if m['f1'] > best_f1:
            best_f1, best_t = m['f1'], t

    print("-" * 55)
    print(f"Best: threshold={best_t:.2f} → F1={best_f1:.4f}")
    return best_t, best_f1


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                 train_loader, val_loader, device, output_dir, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.start_epoch = 0

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self):
        self.model.train()
        total_loss, n = 0, 0

        for images, targets in self.train_loader:
            images = images.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss, _ = self.criterion(outputs, targets)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def save(self, name, epoch):
        """Save checkpoint with epoch information."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_f1': self.best_f1,
            'patience_counter': self.patience_counter,
        }, self.output_dir / f'{name}.pth')

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state."""
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        self.patience_counter = checkpoint.get('patience_counter', 0)

        print(f"Resumed from epoch {self.start_epoch}, best F1: {self.best_f1:.4f}")

    def train(self, epochs, patience=25, unfreeze_epoch=None):
        print(f"\nTraining two-stage detector for {epochs} epochs")
        if self.start_epoch > 0:
            print(f"Resuming from epoch {self.start_epoch}")

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()

            if unfreeze_epoch and epoch == unfreeze_epoch:
                print("\n*** Unfreezing backbone ***")
                self.model.unfreeze_backbone()
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= 0.1

            loss = self.train_epoch()
            if self.scheduler:
                self.scheduler.step()

            val_m = {}
            if self.val_loader:
                val_m = evaluate(self.model, self.val_loader, self.device,
                                 self.config['threshold'], self.config['match_radius'])
                if val_m['f1'] > self.best_f1:
                    self.best_f1 = val_m['f1']
                    self.patience_counter = 0
                    self.save('best', epoch)
                else:
                    self.patience_counter += 1

            lr = self.optimizer.param_groups[0]['lr']
            log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | loss={loss:.4f} | lr={lr:.2e}"
            if val_m:
                log += f" | P={val_m['precision']:.3f} R={val_m['recall']:.3f} F1={val_m['f1']:.3f} max={val_m['avg_max_score']:.3f}"
                if val_m['f1'] >= self.best_f1:
                    log += " ★"
            print(log)

            if epoch % 10 == 0:
                self.save('latest', epoch)

            if patience > 0 and self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.save('final', epoch)
        print(f"\nBest F1: {self.best_f1:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')

    parser.add_argument('--backbone', default='vit_large_patch16_dinov3.sat493m')
    parser.add_argument('--heatmap_size', type=int, default=128)
    parser.add_argument('--max_proposals', type=int, default=300)
    parser.add_argument('--proposal_threshold', type=float, default=0.1)
    parser.add_argument('--refine_hidden', type=int, default=512)
    parser.add_argument('--roi_size', type=int, default=11)

    parser.add_argument('--heatmap_sigma', type=float, default=2.0)
    parser.add_argument('--stage1_weight', type=float, default=1.0)
    parser.add_argument('--stage2_weight', type=float, default=1.0)
    parser.add_argument('--aux_weight', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--unfreeze_epoch', type=int, default=None)

    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--match_radius', type=float, default=100)

    parser.add_argument('--output_dir', default='./outputs_twostage_v2')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = vars(args)

    # Data
    train_ds = PointDataset(args.train_csv, args.train_image_dir, 512, augment=True)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)

    val_loader = None
    if args.val_csv and args.val_image_dir:
        val_ds = PointDataset(args.val_csv, args.val_image_dir, 512, augment=False)
        val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = TwoStagePointDetector(
        backbone=args.backbone,
        freeze_backbone=True,
        heatmap_size=args.heatmap_size,
        max_proposals=args.max_proposals,
        proposal_threshold=args.proposal_threshold,
        refine_hidden=args.refine_hidden,
        roi_size=args.roi_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}, Trainable: {n_train:,}")

    # Loss
    criterion = TwoStageLoss(
        heatmap_sigma=args.heatmap_sigma,
        stage1_weight=args.stage1_weight,
        stage2_weight=args.stage2_weight,
        aux_weight=args.aux_weight,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    # Trainer
    trainer = Trainer(model, criterion, optimizer, scheduler,
                      train_loader, val_loader, device, args.output_dir, config)

    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(args.epochs, args.patience, args.unfreeze_epoch)

    # Threshold sweep
    if val_loader:
        best_path = Path(args.output_dir) / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        threshold_sweep(model, val_loader, device, args.match_radius)


if __name__ == '__main__':
    main()