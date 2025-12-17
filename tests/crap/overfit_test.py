"""
P2PNet V4 - Direct Feature Sampling

The transformer decoder is causing query collapse with ViT.
New approach: Sample features directly at reference point locations.

For each query:
1. Sample ViT features at the reference point via bilinear interpolation
2. Concatenate with positional encoding
3. Use simple MLP to predict class + offset

This guarantees each query sees different features based on its position.
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
from typing import Dict, List
import timm


# =============================================================================
# DATASET
# =============================================================================

class SimplePointDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, image_size: int = 512, normalize: bool = True):
        self.image_dir = image_dir
        self.image_size = image_size
        self.normalize = normalize

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        self.annotations = {}
        for img_name in self.image_names:
            img_df = self.df[self.df['images'] == img_name]
            points = img_df[['x', 'y']].values.astype(np.float32)
            labels = img_df['labels'].values.astype(np.int64)
            self.annotations[img_name] = {'points': points, 'labels': labels}

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"Loaded {len(self.image_names)} images with {len(self.df)} total annotations")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if image.size != (self.image_size, self.image_size):
            scale_x = self.image_size / image.size[0]
            scale_y = self.image_size / image.size[1]
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
# MODEL V4 - DIRECT FEATURE SAMPLING
# =============================================================================

class SimpleP2PNetV4(nn.Module):
    """
    P2PNet with direct feature sampling - NO transformer decoder.

    For each reference point:
    1. Sample features from ViT/CNN feature map via bilinear interpolation
    2. Process with MLP to get classification + offset

    This guarantees spatial diversity - each query MUST see different features.
    """

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
            print(f"  Feature dim: {self.feat_dim}")
            print(f"  Prefix tokens: {self.num_prefix_tokens}")
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

        # Reference points
        self.register_buffer('reference_points', self._create_grid(num_queries))

        # Feature projection
        self.input_proj = nn.Linear(self.feat_dim, hidden_dim)

        # Positional encoding (simple learned embedding based on reference point index)
        self.pos_embed = nn.Embedding(num_queries, hidden_dim)

        # Prediction heads - simple MLPs
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # features + position
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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

        # Diverse position embeddings
        nn.init.normal_(self.pos_embed.weight, std=1.0)

        # Zero-init final layers
        nn.init.zeros_(self.class_head[-1].weight)
        nn.init.zeros_(self.class_head[-1].bias)
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    def _sample_features(self, feature_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Sample features at given points using bilinear interpolation.

        Args:
            feature_map: [B, H, W, C] - spatial feature map
            points: [N, 2] - normalized coordinates in [0, 1]

        Returns:
            sampled: [B, N, C] - features at each point
        """
        B, H, W, C = feature_map.shape
        N = points.shape[0]

        # Convert [0, 1] to [-1, 1] for grid_sample
        grid = points * 2 - 1  # [N, 2]
        grid = grid.view(1, 1, N, 2).expand(B, 1, N, 2)  # [B, 1, N, 2]

        # Reshape feature map for grid_sample: [B, C, H, W]
        feature_map = feature_map.permute(0, 3, 1, 2)

        # Sample
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [B, C, 1, N]

        sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, N, C]
        return sampled

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        device = x.device

        # Extract features
        if self.is_vit:
            features = self.backbone.forward_features(x)  # [B, num_tokens, feat_dim]

            # Remove prefix tokens
            if self.num_prefix_tokens > 0:
                features = features[:, self.num_prefix_tokens:, :]

            # Reshape to spatial grid
            H = W = self.spatial_size
            features = features.view(B, H, W, self.feat_dim)  # [B, H, W, C]
        else:
            features = self.backbone(x)[-1]  # [B, C, H, W]
            features = features.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Sample features at reference points
        sampled_features = self._sample_features(features, self.reference_points)  # [B, N, feat_dim]

        # Project features
        sampled_features = self.input_proj(sampled_features)  # [B, N, hidden_dim]

        # Get position embeddings
        pos_indices = torch.arange(self.num_queries, device=device)
        pos_embed = self.pos_embed(pos_indices)  # [N, hidden_dim]
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N, hidden_dim]

        # Concatenate features and position
        combined = torch.cat([sampled_features, pos_embed], dim=-1)  # [B, N, hidden_dim*2]

        # Predict
        logits = self.class_head(combined)  # [B, N, num_classes]
        offsets = self.offset_head(combined)  # [B, N, 2]

        # Apply bounded offsets
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

                status = "✓" if gap > 0.1 else "⚠️" if gap > 0 else "❌"
                collapse = "COLLAPSE!" if score_std < 0.05 else ""

                print(f"\n[Step {self._step}] cls={cls_loss.item():.4f} reg={reg_loss.item():.4f} total={total.item():.4f}")
                print(f"  matched={n_matched} pos_weight={pos_weight:.1f}")
                print(f"  scores: matched={avg_m:.3f} unmatched={avg_u:.3f} gap={gap:.3f} {status}")
                print(f"  score_std={score_std:.4f} {collapse}")

        return total


# =============================================================================
# TRAINING
# =============================================================================

def overfit_test(model, dataloader, criterion, device, num_batches=5, epochs=200, lr=1e-4, lr_backbone=None):
    import itertools

    batches = list(itertools.islice(dataloader, num_batches))
    total_gt = sum(len(t['points']) for _, targets in batches for t in targets)
    max_gt = max(len(t['points']) for _, targets in batches for t in targets)

    print(f"\n{'='*60}")
    print(f"OVERFITTING TEST: {num_batches} batches, {total_gt} GT points")
    print(f"Max GT: {max_gt}, Queries: {model.num_queries}")
    print(f"{'='*60}\n")

    model.train()

    if lr_backbone is not None:
        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr},
        ], weight_decay=1e-4)
        print(f"lr={lr} for heads, lr={lr_backbone} for backbone")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        print(f"lr={lr} for all")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, targets in batches:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss={epoch_loss/num_batches:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"\n{'='*60}\nFINAL CHECK\n{'='*60}")

    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(batches):
            images = images.to(device)
            outputs = model(images)

            logits = outputs['logits']
            points = outputs['pred_points_normalized']
            probs = logits.softmax(-1)[:, :, 1]

            for b in range(len(targets)):
                gt_pts = targets[b]['points']
                if len(gt_pts) == 0:
                    continue

                gt_norm = gt_pts.float().to(device) / 512.0
                n_gt = len(gt_pts)

                top_scores, top_idx = probs[b].topk(min(n_gt, model.num_queries))
                top_pts = points[b, top_idx]
                dists = torch.cdist(top_pts, gt_norm)
                min_dists = dists.min(dim=1)[0]
                score_std = probs[b].std().item()

                print(f"\nBatch {i}, Sample {b}: {n_gt} GT")
                print(f"  Top scores: {top_scores[:8].cpu().numpy().round(3)}...")
                print(f"  Score std: {score_std:.4f}")
                print(f"  Dist to GT: {min_dists[:8].cpu().numpy().round(3)}...")

                hits = (min_dists < 0.05).sum().item()
                print(f"  Hits (<25px): {hits}/{n_gt}")

                if top_scores.min() > 0.7 and hits >= n_gt * 0.8 and score_std > 0.1:
                    print("  ✓ GOOD")
                elif score_std < 0.05:
                    print("  ❌ COLLAPSED")
                else:
                    print("  ⚠️ Not fully learned")


# =============================================================================
# MAIN
# =============================================================================

def main():
    CSV_PATH = "/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_s_detection/train/herdnet_format_512_0_crops.csv"
    IMAGE_DIR = "/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_s_detection/train/crops_512_numNone_overlap0"

    CSV_PATH = "/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_s_detection/val/herdnet_format_512_0_crops.csv"
    IMAGE_DIR = "/raid/cwinkelmann/training_data/iguana/2025_11_12/Fernandina_s_detection/val/crops_512_numNone_overlap0"

    BATCH_SIZE = 4
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    dataset = SimplePointDataset(csv_path=CSV_PATH, image_dir=IMAGE_DIR, image_size=512)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    # Model config - TRY DINOV3 WITH THIS APPROACH
    # BACKBONE = 'vit_small_patch16_dinov2.lvd142m'  # DINOv2
    BACKBONE = 'vit_small_patch16_dinov3_qkvb.lvd1689m'  # DINOv3
    # BACKBONE = 'resnet50'

    IS_VIT = 'vit' in BACKBONE.lower() or 'dino' in BACKBONE.lower()
    FREEZE_VIT = True

    model = SimpleP2PNetV4(
        backbone=BACKBONE,
        num_queries=100,
        num_classes=2,
        hidden_dim=256,
        max_offset=0.5,
        pretrained=True,
        freeze_backbone=IS_VIT and FREEZE_VIT,
    ).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = MinimalHungarianLoss(cost_class=2.0, cost_point=5.0, cls_weight=1.0, reg_weight=10.0, image_size=512, debug=True)

    if IS_VIT:
        print("\n*** ViT mode - Direct Feature Sampling ***")
        if FREEZE_VIT:
            print("*** Backbone FROZEN ***")
            overfit_test(model=model, dataloader=dataloader, criterion=criterion, device=DEVICE,
                        num_batches=5, epochs=500, lr=1e-3, lr_backbone=None)
        else:
            overfit_test(model=model, dataloader=dataloader, criterion=criterion, device=DEVICE,
                        num_batches=5, epochs=500, lr=5e-4, lr_backbone=1e-6)
    else:
        overfit_test(model=model, dataloader=dataloader, criterion=criterion, device=DEVICE,
                    num_batches=5, epochs=500, lr=1e-4)


if __name__ == "__main__":
    main()