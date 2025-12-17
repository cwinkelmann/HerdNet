"""
Iguana Presence Classifier

Simple binary classifier: Does this tile contain an iguana?

Approach:
1. Random crops from full images
2. Label as positive if crop contains any iguana point, negative otherwise
3. Use DINOv3 backbone with frozen features + simple classification head
4. Once this works well, localization becomes trivial via patch embeddings

Expected: >95% accuracy on single-iguana tiles should be achievable.
"""

import os
import argparse
import random
import time
from pathlib import Path
from typing import Tuple, List, Optional

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
    raise ImportError("albumentations required: pip install albumentations")


# =============================================================================
# DATASET: Random crops with presence labels
# =============================================================================

class IguanaPresenceDataset(Dataset):
    """
    Dataset that extracts random crops and labels them by iguana presence.

    Positive: Crop contains at least one iguana point (with margin from edge)
    Negative: Crop contains no iguana points

    Balance is controlled by `positive_ratio` - we oversample positives to
    avoid class imbalance issues.
    """

    def __init__(
            self,
            csv_path: str,
            image_dir: str,
            crop_size: int = 512,
            crops_per_image: int = 4,
            positive_ratio: float = 0.5,
            min_edge_margin: int = 20,
            augment: bool = True,
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.positive_ratio = positive_ratio
        self.min_edge_margin = min_edge_margin
        self.augment = augment

        # Load annotations
        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        # Store points per image in original pixel coordinates
        self.annotations = {}
        for name, group in self.df.groupby('images'):
            self.annotations[name] = group[['x', 'y']].values.astype(np.float32)

        # Normalization
        self.mean = np.array([0.430, 0.411, 0.296])
        self.std = np.array([0.213, 0.156, 0.143])

        # Build transforms
        self.photometric_transform = self._build_photometric() if augment else None
        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])

        # Cache image sizes for efficient sampling
        self._image_sizes = {}

        print(f"IguanaPresenceDataset: {len(self.image_names)} images")
        print(f"  Crop size: {crop_size}, Crops per image: {crops_per_image}")
        print(f"  Positive ratio: {positive_ratio}, Edge margin: {min_edge_margin}")
        print(f"  Total crops per epoch: {len(self)}")

    def _build_photometric(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
        ])

    def _get_image_size(self, name: str) -> Tuple[int, int]:
        """Get image dimensions (cached)."""
        if name not in self._image_sizes:
            img_path = os.path.join(self.image_dir, name)
            with Image.open(img_path) as img:
                self._image_sizes[name] = img.size  # (width, height)
        return self._image_sizes[name]

    def _sample_positive_crop(self, points: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int]:
        """Sample a crop that contains at least one point with margin."""
        # Filter points that can have valid crops around them
        valid_points = []
        for pt in points:
            px, py = pt
            # Check if we can place a crop with the point having sufficient margin
            if (px >= self.min_edge_margin and
                    px <= img_w - self.min_edge_margin and
                    py >= self.min_edge_margin and
                    py <= img_h - self.min_edge_margin):
                valid_points.append(pt)

        if not valid_points:
            # Fallback: use any point
            valid_points = points.tolist()

        # Pick a random valid point
        pt = random.choice(valid_points)
        px, py = pt

        # Calculate valid crop range that keeps point inside with margin
        x_min = max(0, int(px - self.crop_size + self.min_edge_margin))
        x_max = min(img_w - self.crop_size, int(px - self.min_edge_margin))
        y_min = max(0, int(py - self.crop_size + self.min_edge_margin))
        y_max = min(img_h - self.crop_size, int(py - self.min_edge_margin))

        # Handle edge cases
        x_min = min(x_min, max(0, img_w - self.crop_size))
        x_max = max(x_max, 0)
        y_min = min(y_min, max(0, img_h - self.crop_size))
        y_max = max(y_max, 0)

        crop_x = random.randint(min(x_min, x_max), max(x_min, x_max))
        crop_y = random.randint(min(y_min, y_max), max(y_min, y_max))

        return crop_x, crop_y

    def _sample_negative_crop(self, points: np.ndarray, img_w: int, img_h: int,
                              max_attempts: int = 50) -> Optional[Tuple[int, int]]:
        """Sample a crop that contains no points."""
        max_x = max(0, img_w - self.crop_size)
        max_y = max(0, img_h - self.crop_size)

        for _ in range(max_attempts):
            crop_x = random.randint(0, max_x) if max_x > 0 else 0
            crop_y = random.randint(0, max_y) if max_y > 0 else 0

            # Check if any point falls inside this crop
            in_crop = (
                    (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
                    (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
            )

            if not in_crop.any():
                return crop_x, crop_y

        return None  # Failed to find negative crop

    def __len__(self):
        return len(self.image_names) * self.crops_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image

        name = self.image_names[img_idx]
        points = self.annotations[name]

        # Load image
        img_path = os.path.join(self.image_dir, name)
        img = np.array(Image.open(img_path).convert('RGB'))
        img_h, img_w = img.shape[:2]

        # Decide positive or negative based on ratio
        want_positive = random.random() < self.positive_ratio

        if want_positive and len(points) > 0:
            crop_x, crop_y = self._sample_positive_crop(points, img_w, img_h)
            label = 1.0
        else:
            # Try to get negative crop
            result = self._sample_negative_crop(points, img_w, img_h)
            if result is not None:
                crop_x, crop_y = result
                label = 0.0
            elif len(points) > 0:
                # Fallback to positive if can't find negative
                crop_x, crop_y = self._sample_positive_crop(points, img_w, img_h)
                label = 1.0
            else:
                # No points at all - random crop is negative
                max_x = max(0, img_w - self.crop_size)
                max_y = max(0, img_h - self.crop_size)
                crop_x = random.randint(0, max_x) if max_x > 0 else 0
                crop_y = random.randint(0, max_y) if max_y > 0 else 0
                label = 0.0

        # Extract crop
        crop = img[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]

        # Handle edge cases (image smaller than crop)
        if crop.shape[0] < self.crop_size or crop.shape[1] < self.crop_size:
            padded = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        # Verify label by checking actual point presence in crop
        if len(points) > 0:
            in_crop = (
                    (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
                    (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
            )
            actual_label = 1.0 if in_crop.any() else 0.0
            label = actual_label  # Use ground truth, not intended label

        # Apply augmentations
        if self.photometric_transform:
            crop = self.photometric_transform(image=crop)['image']

        # Normalize and convert to tensor
        crop = self.normalize_transform(image=crop)['image']

        return crop, torch.tensor(label, dtype=torch.float32)


# =============================================================================
# MODEL: DINOv3 backbone + classification head
# =============================================================================

class IguanaClassifier(nn.Module):
    """
    Simple binary classifier using DINOv3 backbone.

    Uses CLS token from ViT for classification.
    Backbone is frozen by default - we just train the head.
    """

    def __init__(
            self,
            backbone: str = 'vit_large_patch16_dinov3.sat493m',
            freeze_backbone: bool = True,
            hidden_dim: int = 512,
            dropout: float = 0.3,
    ):
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features

        print(f"Backbone: {backbone}")
        print(f"  Feature dim: {self.feat_dim}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen")

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            logits: [B] binary classification logits
        """
        # Get CLS token features
        features = self.backbone(x)  # [B, feat_dim]

        # Classify
        logits = self.head(features).squeeze(-1)  # [B]

        return logits

    def unfreeze_backbone(self, n_blocks: Optional[int] = None):
        """Unfreeze backbone (all or last n blocks)."""
        if n_blocks is None:
            print("Unfreezing entire backbone")
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            if hasattr(self.backbone, 'blocks'):
                total = len(self.backbone.blocks)
                print(f"Unfreezing last {n_blocks} of {total} blocks")
                for i, block in enumerate(self.backbone.blocks):
                    if i >= total - n_blocks:
                        for p in block.parameters():
                            p.requires_grad = True


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, epoch: int = 0):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pos_correct = 0
    pos_total = 0
    neg_correct = 0
    neg_total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        # Per-class accuracy
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_correct += (preds[pos_mask] == labels[pos_mask]).sum().item()
        pos_total += pos_mask.sum().item()
        neg_correct += (preds[neg_mask] == labels[neg_mask]).sum().item()
        neg_total += neg_mask.sum().item()

    return {
        'loss': total_loss / total_samples,
        'acc': total_correct / total_samples,
        'pos_acc': pos_correct / max(pos_total, 1),
        'neg_acc': neg_correct / max(neg_total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    accuracy = (all_preds == all_labels).mean()

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    pos_acc = (all_preds[pos_mask] == all_labels[pos_mask]).mean() if pos_mask.any() else 0
    neg_acc = (all_preds[neg_mask] == all_labels[neg_mask]).mean() if neg_mask.any() else 0

    # Precision, Recall, F1
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        'loss': total_loss / total_samples,
        'acc': accuracy,
        'pos_acc': pos_acc,
        'neg_acc': neg_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_pos': pos_mask.sum(),
        'n_neg': neg_mask.sum(),
    }


def main():
    parser = argparse.ArgumentParser(description="Iguana Presence Classifier")

    # Data
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')

    # Model
    parser.add_argument('--backbone', default='vit_large_patch16_dinov3.sat493m',
                        help='timm model name (DINOv2/v3)')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Dataset
    parser.add_argument('--crop_size', type=int, default=518,
                        help='Crop size (512 for DINOv4 patch14)')
    parser.add_argument('--crops_per_image', type=int, default=16)
    parser.add_argument('--positive_ratio', type=float, default=0.5)
    parser.add_argument('--min_edge_margin', type=int, default=30)

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--unfreeze_epoch', type=int, default=10,
                        help='Epoch to unfreeze last backbone blocks')
    parser.add_argument('--unfreeze_blocks', type=int, default=4,
                        help='Number of backbone blocks to unfreeze')

    # Output
    parser.add_argument('--output_dir', default='./outputs_classifier')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_ds = IguanaPresenceDataset(
        args.train_csv, args.train_image_dir,
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        positive_ratio=args.positive_ratio,
        min_edge_margin=args.min_edge_margin,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if args.val_csv and args.val_image_dir:
        val_ds = IguanaPresenceDataset(
            args.val_csv, args.val_image_dir,
            crop_size=args.crop_size,
            crops_per_image=args.crops_per_image,
            positive_ratio=0.5,  # Balanced for evaluation
            min_edge_margin=args.min_edge_margin,
            augment=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # Model
    model = IguanaClassifier(
        backbone=args.backbone,
        freeze_backbone=True,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}, Trainable: {n_train:,}")

    # Optimizer (only head initially)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    best_f1 = 0
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(args.epochs):
        # Unfreeze backbone at specified epoch
        if epoch == args.unfreeze_epoch and args.unfreeze_blocks > 0:
            print(f"\n*** Unfreezing last {args.unfreeze_blocks} backbone blocks ***")
            model.unfreeze_backbone(args.unfreeze_blocks)

            # Reset optimizer with lower LR for backbone
            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.named_parameters()
                            if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.01},
                {'params': [p for n, p in model.named_parameters()
                            if 'backbone' not in n and p.requires_grad], 'lr': args.lr * 0.1},
            ], weight_decay=args.weight_decay)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-7
            )

        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | "
        log += f"loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} "
        log += f"[pos={train_m['pos_acc']:.3f} neg={train_m['neg_acc']:.3f}]"

        if val_loader:
            val_m = evaluate(model, val_loader, device)
            log += f" | val_acc={val_m['acc']:.3f} P={val_m['precision']:.3f} "
            log += f"R={val_m['recall']:.3f} F1={val_m['f1']:.3f}"

            if val_m['f1'] > best_f1:
                best_f1 = val_m['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_f1': best_f1,
                }, output_dir / 'best.pth')
                log += " ★"

        print(log)

        # Save checkpoint
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / 'latest.pth')

    print("\n" + "=" * 70)
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print("=" * 70)

    # Final detailed evaluation
    if val_loader:
        print("\nFinal evaluation on validation set:")

        # Load best model
        best_path = output_dir / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

        val_m = evaluate(model, val_loader, device)
        print(f"  Accuracy: {val_m['acc']:.4f}")
        print(f"  Positive accuracy: {val_m['pos_acc']:.4f} (n={val_m['n_pos']})")
        print(f"  Negative accuracy: {val_m['neg_acc']:.4f} (n={val_m['n_neg']})")
        print(f"  Precision: {val_m['precision']:.4f}")
        print(f"  Recall: {val_m['recall']:.4f}")
        print(f"  F1: {val_m['f1']:.4f}")


if __name__ == '__main__':
    main()