"""
P2PNet Standalone V5 - Overfit Test + Evaluation

1. Overfit on training batches (should get ~100% accuracy)
2. Evaluate on separate validation set
3. Detailed diagnostics to compare

Usage:
    python standalone_p2p_training_v5.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional
import timm


# =============================================================================
# DATASET
# =============================================================================

class SimplePointDataset(Dataset):
    """
    Minimal dataset for point annotations.
    CSV format: images,x,y,species,labels
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_size: int = 512,
        normalize: bool = True,
        max_images: Optional[int] = None,
    ):
        self.image_dir = image_dir
        self.image_size = image_size
        self.normalize = normalize

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        if max_images is not None:
            self.image_names = self.image_names[:max_images]
            self.df = self.df[self.df['images'].isin(self.image_names)]

        self.annotations = {}
        for img_name in self.image_names:
            img_df = self.df[self.df['images'] == img_name]
            points = img_df[['x', 'y']].values.astype(np.float32)
            labels = img_df['labels'].values.astype(np.int64)
            self.annotations[img_name] = {'points': points, 'labels': labels}

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        total_points = sum(len(a['points']) for a in self.annotations.values())
        print(f"Loaded {len(self.image_names)} images with {total_points} total annotations")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        orig_w, orig_h = image.size
        if image.size != (self.image_size, self.image_size):
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        else:
            scale_x, scale_y = 1.0, 1.0

        image = np.array(image, dtype=np.float32) / 255.0
        if self.normalize:
            image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1)

        anno = self.annotations[img_name]
        points = anno['points'].copy()
        labels = anno['labels'].copy()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y

        return image, {
            'points': torch.from_numpy(points),
            'labels': torch.from_numpy(labels),
            'image_name': img_name,
        }


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# =============================================================================
# MODEL - DIRECT FEATURE SAMPLING (V4)
# =============================================================================

class SimpleP2PNetV4(nn.Module):
    """P2PNet with direct feature sampling - NO transformer decoder."""

    def __init__(
        self,
        backbone: str = 'resnet50',
        num_queries: int = 100,
        num_classes: int = 2,
        hidden_dim: int = 256,
        max_offset: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.max_offset = max_offset
        self.hidden_dim = hidden_dim
        self.is_vit = 'vit' in backbone.lower() or 'dino' in backbone.lower()

        if self.is_vit:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

            with torch.no_grad():
                dummy = torch.randn(1, 3, 512, 512)
                feat = self.backbone.forward_features(dummy)

                self.feat_dim = feat.shape[-1]
                num_tokens = feat.shape[1]

                if hasattr(self.backbone, 'num_prefix_tokens'):
                    self.num_prefix_tokens = self.backbone.num_prefix_tokens
                else:
                    self.num_prefix_tokens = 1 if num_tokens in [197, 257, 577, 785, 1025] else 0

                spatial_tokens = num_tokens - self.num_prefix_tokens
                self.spatial_size = int(np.sqrt(spatial_tokens))

            print(f"ViT backbone: {backbone}")
            print(f"  Feature dim: {self.feat_dim}, Prefix tokens: {self.num_prefix_tokens}")
            print(f"  Spatial: {self.spatial_size}x{self.spatial_size}")
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=[-1])

            with torch.no_grad():
                dummy = torch.randn(1, 3, 512, 512)
                feat = self.backbone(dummy)[-1]
                self.feat_dim = feat.shape[1]
                self.spatial_size = feat.shape[2]

            print(f"CNN backbone: {backbone}")
            print(f"  Feature dim: {self.feat_dim}, spatial: {self.spatial_size}x{self.spatial_size}")

        if freeze_backbone:
            print("Freezing backbone!")
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer('reference_points', self._create_grid(num_queries))
        self.input_proj = nn.Linear(self.feat_dim, hidden_dim)
        self.pos_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        self._init_weights()

    def _create_grid(self, n: int) -> torch.Tensor:
        grid_size = int(np.ceil(np.sqrt(n)))
        coords = torch.linspace(0.05, 0.95, grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return grid[:n]

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_embed.weight, std=1.0)
        nn.init.zeros_(self.class_head[-1].weight)
        nn.init.zeros_(self.class_head[-1].bias)
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def _sample_features(self, feature_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        B, H, W, C = feature_map.shape
        N = points.shape[0]

        grid = points * 2 - 1
        grid = grid.view(1, 1, N, 2).expand(B, 1, N, 2)
        feature_map = feature_map.permute(0, 3, 1, 2)

        sampled = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        return sampled

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        device = x.device

        if self.is_vit:
            features = self.backbone.forward_features(x)
            if self.num_prefix_tokens > 0:
                features = features[:, self.num_prefix_tokens:, :]
            features = features.view(B, self.spatial_size, self.spatial_size, self.feat_dim)
        else:
            features = self.backbone(x)[-1]
            features = features.permute(0, 2, 3, 1)

        sampled_features = self._sample_features(features, self.reference_points)
        sampled_features = self.input_proj(sampled_features)

        pos_indices = torch.arange(self.num_queries, device=device)
        pos_embed = self.pos_embed(pos_indices).unsqueeze(0).expand(B, -1, -1)

        combined = torch.cat([sampled_features, pos_embed], dim=-1)

        logits = self.class_head(combined)
        offsets = self.offset_head(combined)

        ref_pts = self.reference_points.unsqueeze(0).expand(B, -1, -1)

        if self.max_offset is not None:
            offsets = torch.tanh(offsets) * self.max_offset
            points = (ref_pts + offsets).clamp(0, 1)
        else:
            points = torch.sigmoid(offsets)

        return {
            'logits': logits,
            'pred_points_normalized': points,
            'offsets': offsets,
            'reference_points': ref_pts,
        }


# =============================================================================
# LOSS
# =============================================================================

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

                status = "✓" if gap > 0.3 else "⚠️" if gap > 0 else "❌"
                collapse = " COLLAPSE!" if score_std < 0.05 else ""

                print(f"\n[Step {self._step}] cls={cls_loss.item():.4f} reg={reg_loss.item():.4f} total={total.item():.4f}")
                print(f"  matched={n_matched} pos_weight={pos_weight:.1f}")
                print(f"  scores: matched={avg_m:.3f} unmatched={avg_u:.3f} gap={gap:.3f} {status}{collapse}")

        return total


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_detailed(model, dataloader, device, image_size=512, threshold=0.5, match_radius=25):
    """
    Detailed evaluation with per-image diagnostics.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: torch device
        image_size: Image size in pixels
        threshold: Confidence threshold for predictions
        match_radius: Radius in pixels for matching pred to GT
    """
    model.eval()

    all_results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    print(f"\n{'='*70}")
    print(f"EVALUATION (threshold={threshold}, match_radius={match_radius}px)")
    print(f"{'='*70}")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)

            logits = outputs['logits']
            points_norm = outputs['pred_points_normalized']
            probs = logits.softmax(-1)[:, :, 1]

            for b in range(len(targets)):
                gt_pts = targets[b]['points']
                img_name = targets[b].get('image_name', f'batch{batch_idx}_sample{b}')

                # Get predictions above threshold
                mask = probs[b] >= threshold
                pred_scores = probs[b, mask]
                pred_pts_norm = points_norm[b, mask]

                # Convert to pixels
                pred_pts_px = pred_pts_norm.clone()
                pred_pts_px[:, 0] *= image_size
                pred_pts_px[:, 1] *= image_size

                gt_pts_px = gt_pts.float().to(device)
                if gt_pts_px.numel() > 0 and gt_pts_px.max() <= 1.0:
                    gt_pts_px = gt_pts_px * image_size

                n_pred = len(pred_pts_px)
                n_gt = len(gt_pts_px)

                # Match predictions to GT
                matched_gt = set()
                matched_pred = set()

                if n_pred > 0 and n_gt > 0:
                    dists = torch.cdist(pred_pts_px, gt_pts_px)

                    # Greedy matching
                    # Sort by distance
                    flat_dists = dists.flatten()
                    sorted_indices = flat_dists.argsort()

                    for idx in sorted_indices:
                        if flat_dists[idx] > match_radius:
                            break
                        pred_i = idx // n_gt
                        gt_i = idx % n_gt

                        if pred_i.item() not in matched_pred and gt_i.item() not in matched_gt:
                            matched_pred.add(pred_i.item())
                            matched_gt.add(gt_i.item())

                    tp = len(matched_pred)
                    fp = n_pred - tp
                    fn = n_gt - len(matched_gt)
                else:
                    tp = 0
                    fp = n_pred
                    fn = n_gt

                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Print per-image results
                precision = tp / max(n_pred, 1)
                recall = tp / max(n_gt, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-6)

                status = "✓" if recall >= 0.9 and precision >= 0.9 else "⚠️" if recall >= 0.5 else "❌"

                print(f"\n{img_name}: GT={n_gt}, Pred={n_pred}, TP={tp}, FP={fp}, FN={fn} {status}")
                print(f"  P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

                # Show top predictions
                if n_pred > 0:
                    top_k = min(5, n_pred)
                    top_scores, top_idx = pred_scores.topk(top_k)
                    top_pts = pred_pts_px[top_idx]

                    print(f"  Top {top_k} predictions:")
                    for i in range(top_k):
                        # Find nearest GT
                        if n_gt > 0:
                            gt_dists = (gt_pts_px - top_pts[i]).pow(2).sum(dim=1).sqrt()
                            nearest_dist = gt_dists.min().item()
                        else:
                            nearest_dist = float('inf')

                        matched = "✓" if nearest_dist <= match_radius else ""
                        print(f"    score={top_scores[i]:.3f}, coord=({top_pts[i, 0]:.1f}, {top_pts[i, 1]:.1f}), nearest_GT={nearest_dist:.1f}px {matched}")

                # Show missed GT
                if fn > 0 and n_gt > 0:
                    print(f"  Missed GT:")
                    for gt_i in range(n_gt):
                        if gt_i not in matched_gt:
                            gt_coord = gt_pts_px[gt_i]
                            if n_pred > 0:
                                pred_dists = (pred_pts_px - gt_coord).pow(2).sum(dim=1).sqrt()
                                nearest_pred_dist = pred_dists.min().item()
                                nearest_pred_idx = pred_dists.argmin().item()
                                nearest_pred_score = pred_scores[nearest_pred_idx].item()
                                print(f"    GT=({gt_coord[0]:.1f}, {gt_coord[1]:.1f}), nearest_pred={nearest_pred_dist:.1f}px (score={nearest_pred_score:.3f})")
                            else:
                                # Check ALL predictions (including below threshold)
                                all_pts_norm = points_norm[b]
                                all_pts_px = all_pts_norm.clone()
                                all_pts_px[:, 0] *= image_size
                                all_pts_px[:, 1] *= image_size
                                all_scores = probs[b]

                                all_dists = (all_pts_px - gt_coord).pow(2).sum(dim=1).sqrt()
                                nearest_idx = all_dists.argmin().item()
                                nearest_dist = all_dists[nearest_idx].item()
                                nearest_score = all_scores[nearest_idx].item()

                                print(f"    GT=({gt_coord[0]:.1f}, {gt_coord[1]:.1f}), best_pred={nearest_dist:.1f}px (score={nearest_score:.3f} < threshold)")

                all_results.append({
                    'image': img_name,
                    'n_gt': n_gt,
                    'n_pred': n_pred,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                })

    # Overall metrics
    overall_precision = total_tp / max(total_tp + total_fp, 1)
    overall_recall = total_tp / max(total_tp + total_fn, 1)
    overall_f1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-6)

    print(f"\n{'='*70}")
    print(f"OVERALL: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision={overall_precision:.3f}, Recall={overall_recall:.3f}, F1={overall_f1:.3f}")
    print(f"{'='*70}\n")

    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'per_image': all_results,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_overfit(model, train_batches, criterion, device, epochs=500, lr=1e-3):
    """Train on fixed batches (overfit test)."""
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_gt = sum(len(t['points']) for _, targets in train_batches for t in targets)
    print(f"\nTraining on {len(train_batches)} batches, {total_gt} total GT points")
    print(f"Epochs: {epochs}, LR: {lr}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, targets in train_batches:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: loss={epoch_loss/len(train_batches):.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"Final loss: {epoch_loss/len(train_batches):.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Training data (overfit on this)
    TRAIN_CSV = "/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_s_detection/train/herdnet_format_512_0_crops.csv"
    TRAIN_IMAGE_DIR = "/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_s_detection/train/crops_512_numNone_overlap0"

    # Validation data (evaluate on this) - SET YOUR VAL PATH HERE
    VAL_CSV = "/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_s_detection/val/herdnet_format_512_0_crops.csv"
    VAL_IMAGE_DIR = "/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_s_detection/val/crops_512_numNone_overlap0"

    # Model config
    # BACKBONE = 'vit_small_patch16_dinov2.lvd142m'  # DINOv2
    BACKBONE = 'vit_small_patch16_dinov3_qkvb.lvd1689m'  # DINOv3
    # BACKBONE = 'resnet50'

    IS_VIT = 'vit' in BACKBONE.lower() or 'dino' in BACKBONE.lower()
    FREEZE_BACKBONE = IS_VIT  # Freeze ViT, train CNN

    # Training config
    # NUM_TRAIN_BATCHES = 5  # Number of batches to overfit on
    # BATCH_SIZE = 4
    # EPOCHS = 500

    # In your training config:
    BATCH_SIZE = 25
    NUM_TRAIN_BATCHES = None  # Use ALL batches, not just 5
    EPOCHS = 50  # Fewer epochs when using full dataset
    LR = 1e-4  # Lower LR for full training

    LR = 1e-3 if FREEZE_BACKBONE else 1e-4

    # Evaluation config
    EVAL_THRESHOLD = 0.5
    MATCH_RADIUS = 25  # pixels

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    train_dataset = SimplePointDataset(
        csv_path=TRAIN_CSV,
        image_dir=TRAIN_IMAGE_DIR,
        image_size=512,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Get fixed batches for overfit test
    import itertools
    train_batches = list(itertools.islice(train_loader, NUM_TRAIN_BATCHES))

    # Load validation data if exists
    val_loader = None
    if os.path.exists(VAL_CSV) and os.path.exists(VAL_IMAGE_DIR):
        val_dataset = SimplePointDataset(
            csv_path=VAL_CSV,
            image_dir=VAL_IMAGE_DIR,
            image_size=512,
            max_images=20,  # Limit for faster eval
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # One at a time for detailed eval
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
    else:
        print(f"\n⚠️ Validation data not found at {VAL_CSV}")
        print("Will only do overfit test")

    # =========================================================================
    # BUILD MODEL
    # =========================================================================

    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)

    model = SimpleP2PNetV4(
        backbone=BACKBONE,
        num_queries=100,
        num_classes=2,
        hidden_dim=256,
        max_offset=0.5,
        pretrained=True,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(DEVICE)

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = MinimalHungarianLoss(
        cost_class=2.0,
        cost_point=5.0,
        cls_weight=1.0,
        reg_weight=10.0,
        image_size=512,
        debug=True,
    )

    # =========================================================================
    # TRAIN (OVERFIT TEST)
    # =========================================================================

    print("\n" + "="*70)
    print("OVERFIT TEST")
    print("="*70)

    train_overfit(
        model=model,
        train_batches=train_batches,
        criterion=criterion,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LR,
    )

    # =========================================================================
    # EVALUATE ON TRAIN BATCHES (should be ~100%)
    # =========================================================================

    print("\n" + "="*70)
    print("EVALUATE ON TRAINING DATA (should be ~100%)")
    print("="*70)

    # Convert train batches to a pseudo-dataloader
    class ListDataLoader:
        def __init__(self, batches):
            self.batches = batches
        def __iter__(self):
            return iter(self.batches)
        def __len__(self):
            return len(self.batches)

    train_eval_loader = ListDataLoader(train_batches)

    train_results = evaluate_detailed(
        model=model,
        dataloader=train_eval_loader,
        device=DEVICE,
        image_size=512,
        threshold=EVAL_THRESHOLD,
        match_radius=MATCH_RADIUS,
    )

    # =========================================================================
    # EVALUATE ON VALIDATION DATA
    # =========================================================================

    if val_loader is not None:
        print("\n" + "="*70)
        print("EVALUATE ON VALIDATION DATA (generalization)")
        print("="*70)

        val_results = evaluate_detailed(
            model=model,
            dataloader=val_loader,
            device=DEVICE,
            image_size=512,
            threshold=EVAL_THRESHOLD,
            match_radius=MATCH_RADIUS,
        )

        # Compare
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Train: P={train_results['precision']:.3f}, R={train_results['recall']:.3f}, F1={train_results['f1']:.3f}")
        print(f"Val:   P={val_results['precision']:.3f}, R={val_results['recall']:.3f}, F1={val_results['f1']:.3f}")

        if train_results['f1'] > 0.95 and val_results['f1'] < 0.5:
            print("\n⚠️ Large gap between train and val - model is overfitting as expected")
            print("   To improve val performance, train on more data with augmentation")
        elif train_results['f1'] < 0.8:
            print("\n❌ Train F1 < 0.8 - model is not learning properly")
            print("   Check: loss function, learning rate, data format")


if __name__ == "__main__":
    main()