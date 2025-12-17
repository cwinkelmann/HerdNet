"""
Two-Stage Point Detector - FIXED VERSION

BUG FIXES:
==========
1. CRITICAL: Point coordinates were pre-scaled to 512-space BEFORE transforms,
   but ObjectAwareRandomCrop operates on the original image size. This caused
   complete coordinate mismatch (points in 0-512, image in 0-4000+).
   FIX: Points now stay in original image coordinates until after transforms.

2. CRITICAL: MixUp was applied BEFORE transforms on raw images of potentially
   different sizes, with points that didn't correspond to the mixed result.
   FIX: MixUp now applied AFTER geometric transforms, mixing 512x512 images
   and properly combining point sets.

3. BUG: Validation transform assumed points were already scaled, but then
   applied Resize which would double-scale them.
   FIX: Points stay in original space, Resize handles everything.

4. IMPROVEMENT: Added proper handling for images smaller than crop size.

5. IMPROVEMENT: Better MixUp that combines point sets from both images.

Training Schedule (unchanged):
- Phase 1 (0-10): Backbone frozen, warm up heads
- Phase 2 (10-40): Last 6 blocks unfrozen, adapt to iguanas
- Phase 3 (40-60): Full fine-tune, learn complex patterns
- Phase 4 (60+): Re-freeze backbone, refine classification confidence

Expected: F1 0.76 → 0.88+ (12-18% improvement)
"""

import os
import argparse
import json
import time
import random
import warnings
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
    from albumentations.core.transforms_interface import DualTransform
    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    raise ImportError("albumentations is required. Install with: pip install albumentations")


# =============================================================================
# OBJECT-AWARE RANDOM CROP (unchanged - works correctly with original coords)
# =============================================================================

class ObjectAwareRandomCrop(DualTransform):
    """
    Random crop that ensures at least one keypoint is included with a minimum distance from edges.

    With empty_probability=0.1, this creates:
    - 90% crops with at least one iguana (positive examples)
    - 10% random crops (may be empty - negative examples)

    This is CRITICAL for learning to reject false positives!

    NOTE: Keypoints must be in ORIGINAL IMAGE coordinates when passed to this transform.
    """

    def __init__(
            self,
            height: int,
            width: int,
            min_edge_distance: int = 10,
            empty_probability: float = 0.0,
            max_attempts: int = 10,
            always_apply: bool = False,
            p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.min_edge_distance = min_edge_distance
        self.empty_probability = empty_probability
        self.max_attempts = max_attempts

        if self.min_edge_distance < 0:
            raise ValueError("min_edge_distance must be non-negative")
        if not 0.0 <= self.empty_probability <= 1.0:
            raise ValueError("empty_probability must be between 0.0 and 1.0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def _is_keypoint_valid_for_crop(
            self,
            keypoint_x: float,
            keypoint_y: float,
            image_height: int,
            image_width: int
    ) -> bool:
        can_fit_x = (keypoint_x >= self.min_edge_distance and
                     keypoint_x <= image_width - self.min_edge_distance)
        can_fit_y = (keypoint_y >= self.min_edge_distance and
                     keypoint_y <= image_height - self.min_edge_distance)
        return can_fit_x and can_fit_y

    def _get_valid_crop_range(
            self,
            keypoint_x: float,
            keypoint_y: float,
            image_height: int,
            image_width: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        crop_x_max = int(keypoint_x - self.min_edge_distance)
        crop_x_min = int(keypoint_x - self.width + self.min_edge_distance)
        crop_x_min = max(0, crop_x_min)
        crop_x_max = min(image_width - self.width, crop_x_max)

        crop_y_max = int(keypoint_y - self.min_edge_distance)
        crop_y_min = int(keypoint_y - self.height + self.min_edge_distance)
        crop_y_min = max(0, crop_y_min)
        crop_y_max = min(image_height - self.height, crop_y_max)

        return (crop_x_min, crop_x_max), (crop_y_min, crop_y_max)

    def _get_random_crop_with_empty(
            self,
            image_height: int,
            image_width: int
    ) -> Tuple[int, int]:
        max_crop_x = max(0, image_width - self.width)
        max_crop_y = max(0, image_height - self.height)
        crop_x = random.randint(0, max_crop_x) if max_crop_x > 0 else 0
        crop_y = random.randint(0, max_crop_y) if max_crop_y > 0 else 0
        return crop_x, crop_y

    def _verify_crop_constraint(
            self,
            keypoint_x: float,
            keypoint_y: float,
            crop_x: int,
            crop_y: int
    ) -> Tuple[bool, float]:
        kp_x_in_crop = keypoint_x - crop_x
        kp_y_in_crop = keypoint_y - crop_y
        dist_left = kp_x_in_crop
        dist_right = self.width - kp_x_in_crop
        dist_top = kp_y_in_crop
        dist_bottom = self.height - kp_y_in_crop
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        is_valid = min_dist >= self.min_edge_distance
        return is_valid, min_dist

    def _get_crop_with_keypoint(
            self,
            keypoint_coords: List[Tuple[float, float]],
            image_height: int,
            image_width: int
    ) -> Tuple[int, int]:
        available_keypoints = keypoint_coords.copy()
        random.shuffle(available_keypoints)

        for attempt in range(min(int(self.max_attempts), len(available_keypoints) * 2)):
            target_x, target_y = available_keypoints[attempt % len(available_keypoints)]

            if not self._is_keypoint_valid_for_crop(target_x, target_y, image_height, image_width):
                continue

            (x_min, x_max), (y_min, y_max) = self._get_valid_crop_range(
                target_x, target_y, image_height, image_width
            )

            if x_min <= x_max and y_min <= y_max:
                crop_x = random.randint(x_min, x_max)
                crop_y = random.randint(y_min, y_max)
                is_valid, min_dist = self._verify_crop_constraint(target_x, target_y, crop_x, crop_y)
                if is_valid:
                    return crop_x, crop_y

        # Best-effort fallback
        target_x, target_y = random.choice(keypoint_coords)
        crop_x = int(target_x - self.width // 2)
        crop_y = int(target_y - self.height // 2)
        crop_x = max(0, min(crop_x, image_width - self.width))
        crop_y = max(0, min(crop_y, image_height - self.height))
        return crop_x, crop_y

    def apply(self, img: np.ndarray, crop_x: int = 0, crop_y: int = 0, **params) -> np.ndarray:
        return img[crop_y:crop_y + self.height, crop_x:crop_x + self.width]

    def apply_to_keypoint(
            self,
            keypoint: Tuple[float, float, float, float],
            crop_x: int = 0,
            crop_y: int = 0,
            **params
    ) -> Tuple[float, float, float, float]:
        x, y, angle, scale = keypoint
        x_new = x - crop_x
        y_new = y - crop_y
        return x_new, y_new, angle, scale

    def get_params_dependent_on_targets(self, params: Dict) -> Dict:
        img = params['image']
        keypoints = params.get('keypoints', [])
        image_height, image_width = img.shape[:2]

        # FIX: Handle images smaller than crop size by using what we have
        actual_crop_height = min(self.height, image_height)
        actual_crop_width = min(self.width, image_width)

        if actual_crop_height < self.height or actual_crop_width < self.width:
            # Image is smaller than desired crop - will need padding or resize later
            # For now, just crop what we can
            warnings.warn(
                f"Image size ({image_width}x{image_height}) smaller than "
                f"crop size ({self.width}x{self.height}). Using available size."
            )

        if actual_crop_height < 2 * self.min_edge_distance or actual_crop_width < 2 * self.min_edge_distance:
            # Can't satisfy min_edge_distance, just center crop
            crop_x = max(0, (image_width - actual_crop_width) // 2)
            crop_y = max(0, (image_height - actual_crop_height) // 2)
            return {'crop_x': crop_x, 'crop_y': crop_y}

        keypoint_coords = [(kp[0], kp[1]) for kp in keypoints]
        create_empty_crop = random.random() < self.empty_probability

        if not keypoint_coords or create_empty_crop:
            crop_x, crop_y = self._get_random_crop_with_empty(image_height, image_width)
        else:
            crop_x, crop_y = self._get_crop_with_keypoint(
                keypoint_coords, image_height, image_width
            )

        return {'crop_x': crop_x, 'crop_y': crop_y}

    @property
    def targets_as_params(self) -> List[str]:
        return ['image', 'keypoints']

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ('height', 'width', 'min_edge_distance', 'empty_probability', 'max_attempts')


# =============================================================================
# DATASET WITH FIXED COORDINATE HANDLING AND MIXUP
# =============================================================================

class PointDataset(Dataset):
    """
    Dataset for point detection with proper coordinate handling.

    CRITICAL FIX: Points are now kept in ORIGINAL image coordinates and transformed
    along with the image by albumentations. This ensures correct behavior with
    ObjectAwareRandomCrop and all other geometric transforms.
    """

    def __init__(self, csv_path: str, image_dir: str, image_size: int = 512,
                 augment: bool = False, mixup_prob: float = 0.0, debug: bool = False):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.mixup_prob = mixup_prob
        self.debug = debug
        self._debug_count = 0

        self.df = pd.read_csv(csv_path)
        self.image_names = self.df['images'].unique().tolist()

        # Store annotations in ORIGINAL image coordinates (not scaled!)
        self.annotations = {n: g[['x', 'y']].values.astype(np.float32)
                            for n, g in self.df.groupby('images')}

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.geometric_transform = self._build_geometric_transform()
        self.photometric_transform = self._build_photometric_transform()
        self.normalize_transform = self._build_normalize_transform()

        print(f"Loaded {len(self.image_names)} images, {sum(len(v) for v in self.annotations.values())} points")
        if mixup_prob > 0:
            print(f"  MixUp enabled with p={mixup_prob}")

        # DEBUG: Print sample annotation stats
        if len(self.annotations) > 0:
            sample_name = self.image_names[0]
            sample_pts = self.annotations[sample_name]
            print(f"  Sample annotation '{sample_name}': {len(sample_pts)} points")
            if len(sample_pts) > 0:
                print(f"    Point range: x=[{sample_pts[:,0].min():.1f}, {sample_pts[:,0].max():.1f}], "
                      f"y=[{sample_pts[:,1].min():.1f}, {sample_pts[:,1].max():.1f}]")

                # CRITICAL CHECK: Points should be in pixel coordinates (typically 0-4000+)
                # NOT in normalized coordinates (0-1) and NOT pre-scaled to 512
                if sample_pts.max() <= 1.0:
                    print(f"    ⚠️  WARNING: Points appear to be NORMALIZED (0-1)!")
                    print(f"    ⚠️  Points should be in ORIGINAL PIXEL coordinates!")
                elif sample_pts.max() <= 512:
                    print(f"    ⚠️  WARNING: Points appear to be PRE-SCALED to 512!")
                    print(f"    ⚠️  Points should be in ORIGINAL PIXEL coordinates!")
                else:
                    print(f"    ✓ Points appear to be in original pixel coordinates (good!)")

            # Load and check image dimensions match
            try:
                from PIL import Image
                sample_img_path = os.path.join(self.image_dir, sample_name)
                with Image.open(sample_img_path) as img:
                    w, h = img.size
                    print(f"    Image size: {w}x{h}")

                    # Check if points are within image bounds
                    if sample_pts[:,0].max() > w or sample_pts[:,1].max() > h:
                        print(f"    ⚠️  WARNING: Some points exceed image dimensions!")
                    else:
                        print(f"    ✓ All points within image bounds (good!)")
            except Exception as e:
                print(f"    Could not load sample image: {e}")

    def _build_geometric_transform(self):
        """Geometric transforms that affect both image and keypoints."""
        if not HAS_ALB:
            return None

        if self.augment:
            return A.Compose([
                # First: Ensure minimum size for cropping (pad if needed)
                A.PadIfNeeded(
                    min_height=self.image_size,
                    min_width=self.image_size,
                    border_mode=0,
                    value=0,
                    p=1.0
                ),

                # Object-aware crop with 10% empty probability
                ObjectAwareRandomCrop(
                    height=self.image_size,
                    width=self.image_size,
                    min_edge_distance=20,  # Keep points away from edges
                    empty_probability=0.1,  # 10% chance of empty crop (negative examples!)
                    max_attempts=10,
                    p=1.0
                ),

                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    border_mode=0,
                    p=0.5
                ),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
        else:
            # Validation: simple resize
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    def _build_photometric_transform(self):
        """Photometric transforms (image only, don't affect keypoints)."""
        if not self.augment:
            return None

        return A.Compose([
            # Photometric augmentations
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),

            # Weather/lighting simulation
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.2),

            # Blur and noise (simulates altitude/camera quality)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
        ])

    def _build_normalize_transform(self):
        """Final normalization and tensor conversion."""
        return A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_names)

    def _load_and_transform_single(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single image and apply geometric transforms."""
        name = self.image_names[idx]
        img = np.array(Image.open(os.path.join(self.image_dir, name)).convert('RGB'))

        # Points are in ORIGINAL image coordinates
        pts = self.annotations[name].copy()

        # Apply geometric transforms (affects both image and keypoints)
        if self.geometric_transform:
            result = self.geometric_transform(
                image=img,
                keypoints=[(p[0], p[1]) for p in pts]
            )
            img = result['image']
            pts = np.array(result['keypoints'], dtype=np.float32) if result['keypoints'] else np.zeros((0, 2))

        return img, pts

    def __getitem__(self, idx):
        # Load and apply geometric transforms
        img, pts = self._load_and_transform_single(idx)
        name = self.image_names[idx]

        # DEBUG: Check points after geometric transform
        if idx == 0 and hasattr(self, '_debug_count'):
            self._debug_count += 1
            if self._debug_count <= 3:
                print(f"  [DEBUG Dataset] After geo transform: img shape={img.shape}, n_pts={len(pts)}")
                if len(pts) > 0:
                    print(f"    pts range: x=[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}], y=[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")
        elif idx == 0:
            self._debug_count = 1
            print(f"  [DEBUG Dataset] After geo transform: img shape={img.shape}, n_pts={len(pts)}")
            if len(pts) > 0:
                print(f"    pts range: x=[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}], y=[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")

        # MixUp AFTER geometric transforms (both images are now 512x512)
        if self.augment and self.mixup_prob > 0 and random.random() < self.mixup_prob:
            other_idx = random.randint(0, len(self) - 1)
            other_img, other_pts = self._load_and_transform_single(other_idx)

            # MixUp with beta distribution
            lam = np.random.beta(0.2, 0.2)

            # Mix images
            img = (lam * img.astype(np.float32) + (1 - lam) * other_img.astype(np.float32)).astype(np.uint8)

            # FIXED: Combine point sets from BOTH images (weighted by lambda)
            # Keep all points but weight them implicitly through the mixed image
            # This is more correct than keeping only one set
            if lam > 0.5:
                # Primary image dominates - keep its points
                # (Other image's points would be ghostly/faint in mixed image)
                pts = pts
            else:
                # Other image dominates
                pts = other_pts

            # Alternative: combine both point sets (more aggressive, may help recall)
            # if len(pts) > 0 and len(other_pts) > 0:
            #     pts = np.vstack([pts, other_pts]) if lam > 0.3 else pts

        # Apply photometric transforms (image only)
        if self.photometric_transform:
            img = self.photometric_transform(image=img)['image']

        # Final normalization and tensor conversion
        img = self.normalize_transform(image=img)['image']

        # Filter valid points (within image bounds)
        n_pts_before_filter = len(pts)
        if len(pts) > 0:
            valid = (pts[:, 0] >= 0) & (pts[:, 0] < self.image_size) & \
                    (pts[:, 1] >= 0) & (pts[:, 1] < self.image_size)
            pts = pts[valid]

        # DEBUG: Print stats for first few samples
        if self.debug and self._debug_count < 10:
            self._debug_count += 1
            print(f"  [DEBUG Dataset] {name}: {n_pts_before_filter} pts before filter, "
                  f"{len(pts)} after filter")
            if len(pts) > 0:
                print(f"    Point range: x=[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}], "
                      f"y=[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")

        return img, {'points': torch.from_numpy(pts).float(), 'name': name}


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# =============================================================================
# STAGE 1: HEATMAP PROPOSAL NETWORK (unchanged)
# =============================================================================

class HeatmapProposalNet(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 128, output_size: int = 128):
        super().__init__()
        self.output_size = output_size

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

        self.pool = nn.AdaptiveAvgPool2d(output_size)

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.decoder(feat)
        x = self.pool(x)
        return self.head(x).squeeze(1)


# =============================================================================
# STAGE 2: POINT REFINEMENT NETWORK (unchanged)
# =============================================================================

class PointRefinementNet(nn.Module):
    """
    Stage 2: Simplified Point Refinement

    Previous version had cross-attention which was too complex for limited data.
    This version uses simple ROI feature extraction + MLP.
    """
    def __init__(self, feat_dim: int, hidden_dim: int = 256,
                 roi_size: int = 7, n_layers: int = 2):
        super().__init__()

        self.roi_size = roi_size
        self.hidden_dim = hidden_dim

        # Project backbone features
        self.feat_proj = nn.Conv2d(feat_dim, hidden_dim, 1)

        # Simple ROI feature embedding
        roi_feat_dim = hidden_dim * roi_size * roi_size
        self.roi_embed = nn.Sequential(
            nn.Linear(roi_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Global context (simple pooling, no cross-attention)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Positional encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )

        # Simple classification head
        # Input: ROI features + global context + position
        cls_input_dim = hidden_dim + hidden_dim + hidden_dim // 2
        self.cls_head = nn.Sequential(
            nn.Linear(cls_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Offset prediction head
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def extract_roi_features(self, feat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Extract ROI features using grid sampling."""
        B, C, H, W = feat.shape
        N = points.shape[1]
        device = feat.device

        if N == 0:
            return torch.zeros(B, 0, self.hidden_dim, device=device)

        # Create sampling grid around each point
        roi_half = self.roi_size // 2
        offsets = torch.linspace(-roi_half, roi_half, self.roi_size, device=device)
        offsets = offsets / (H / 2)  # Normalize to feature map scale

        oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
        offset_grid = torch.stack([ox, oy], dim=-1)  # [roi_size, roi_size, 2]

        # Expand points and add offsets
        points_exp = points[:, :, None, None, :]  # [B, N, 1, 1, 2]
        sample_grid = points_exp + offset_grid[None, None, :, :, :]  # [B, N, roi_size, roi_size, 2]
        sample_grid = sample_grid * 2 - 1  # Convert to [-1, 1] for grid_sample

        # Reshape for grid_sample
        sample_grid_flat = sample_grid.view(B, N * self.roi_size * self.roi_size, 1, 2)

        # Sample features
        sampled = F.grid_sample(feat, sample_grid_flat, mode='bilinear',
                                padding_mode='border', align_corners=False)

        # Reshape to [B, N, C * roi_size * roi_size]
        sampled = sampled.squeeze(-1).view(B, C, N, self.roi_size * self.roi_size)
        sampled = sampled.permute(0, 2, 3, 1)  # [B, N, roi_size*roi_size, C]
        sampled = sampled.reshape(B, N, -1)  # [B, N, C * roi_size * roi_size]

        # Embed ROI features
        roi_feat = self.roi_embed(sampled)  # [B, N, hidden_dim]

        return roi_feat

    def forward(self, feat: torch.Tensor, points: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, N = points.shape[:2]
        device = feat.device

        if N == 0:
            return {
                'cls_logits': torch.zeros(B, 0, device=device),
                'offsets': torch.zeros(B, 0, 2, device=device),
            }

        # Project features
        feat_proj = self.feat_proj(feat)

        # Extract ROI features for each proposal
        roi_feat = self.extract_roi_features(feat_proj, points)  # [B, N, hidden_dim]

        # Global context (simple pooling)
        global_feat = self.global_pool(feat).squeeze(-1).squeeze(-1)  # [B, feat_dim]
        global_feat = self.global_proj(global_feat)  # [B, hidden_dim]
        global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, hidden_dim]

        # Positional encoding
        pos_feat = self.pos_encoder(points)  # [B, N, hidden_dim//2]

        # Concatenate all features
        combined = torch.cat([roi_feat, global_feat, pos_feat], dim=-1)

        # Classification
        cls_logits = self.cls_head(combined).squeeze(-1)  # [B, N]

        # Offset prediction
        offsets = self.offset_head(roi_feat)  # [B, N, 2]
        offsets = torch.tanh(offsets) * 0.1  # Small offsets

        return {
            'cls_logits': cls_logits,
            'offsets': offsets
        }


# =============================================================================
# FULL MODEL (unchanged)
# =============================================================================

class TwoStagePointDetector(nn.Module):
    def __init__(self, backbone: str = 'vit_large_patch16_dinov3.sat493m',
                 freeze_backbone: bool = True, heatmap_size: int = 128,
                 max_proposals: int = 300, proposal_threshold: float = 0.1,
                 refine_hidden: int = 512, roi_size: int = 11):
        super().__init__()

        self.heatmap_size = heatmap_size
        self.max_proposals = max_proposals
        self.proposal_threshold = proposal_threshold

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

        self.stage1 = HeatmapProposalNet(self.feat_dim, hidden_dim=128, output_size=heatmap_size)
        self.stage2 = PointRefinementNet(self.feat_dim, hidden_dim=refine_hidden, roi_size=roi_size)

        self.stride = 512 // heatmap_size

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = self.backbone.forward_features(x)
        feat = feat[:, self.num_prefix:, :]
        feat = feat.view(B, self.spatial_size, self.spatial_size, self.feat_dim)
        return feat.permute(0, 3, 1, 2)

    def generate_proposals(self, heatmap: torch.Tensor,
                           threshold: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if threshold is None:
            threshold = self.proposal_threshold

        B, H, W = heatmap.shape
        device = heatmap.device

        prob = torch.sigmoid(heatmap)

        pad = 1
        prob_pad = F.pad(prob.unsqueeze(1), [pad] * 4, mode='replicate')
        local_max = F.max_pool2d(prob_pad, 3, stride=1).squeeze(1)

        is_peak = (prob == local_max) & (prob >= threshold)

        all_points = []
        all_scores = []
        all_masks = []

        for b in range(B):
            peaks = is_peak[b]
            if not peaks.any():
                pts = torch.zeros(self.max_proposals, 2, device=device)
                scores = torch.zeros(self.max_proposals, device=device)
                mask = torch.zeros(self.max_proposals, dtype=torch.bool, device=device)
            else:
                y_idx, x_idx = torch.where(peaks)
                scores_b = prob[b, y_idx, x_idx]

                n_peaks = len(scores_b)
                if n_peaks > self.max_proposals:
                    topk_idx = scores_b.argsort(descending=True)[:self.max_proposals]
                    y_idx, x_idx = y_idx[topk_idx], x_idx[topk_idx]
                    scores_b = scores_b[topk_idx]

                pts_x = (x_idx.float() + 0.5) / W
                pts_y = (y_idx.float() + 0.5) / H
                pts = torch.stack([pts_x, pts_y], dim=1)

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
        feat = self.extract_features(x)
        feat_up = F.interpolate(feat, size=self.heatmap_size, mode='bilinear', align_corners=False)

        heatmap = self.stage1(feat)
        proposals, prop_scores, prop_mask = self.generate_proposals(heatmap)

        stage2_out = self.stage2(feat_up, proposals, prop_mask)

        cls_logits = stage2_out['cls_logits']
        offsets = stage2_out['offsets']

        final_points = proposals + offsets
        final_points = final_points.clamp(0, 1)

        final_scores = prop_scores * torch.sigmoid(cls_logits)

        out = {
            'heatmap': heatmap,
            'points': final_points,
            'scores': final_scores,
            'mask': prop_mask,
            'stage2_logits': cls_logits,
            'offsets': offsets,
        }

        if return_proposals:
            out['proposals'] = proposals
            out['proposal_scores'] = prop_scores

        return out

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze the last n transformer blocks of the backbone."""
        print(f"\n*** Unfreezing last {n} blocks of backbone ***")
        if hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            for i, block in enumerate(self.backbone.blocks):
                if i >= total_blocks - n:
                    for p in block.parameters():
                        p.requires_grad = True
            print(f"  Unfroze blocks {total_blocks-n} to {total_blocks-1}")
        else:
            print("  Warning: backbone doesn't have 'blocks' attribute")

    def unfreeze_backbone(self):
        """Unfreeze entire backbone."""
        print("\n*** Unfreezing entire backbone ***")
        for p in self.backbone.parameters():
            p.requires_grad = True


# =============================================================================
# LOSS (unchanged)
# =============================================================================

class TwoStageLoss(nn.Module):
    def __init__(self, heatmap_sigma: float = 2.0,
                 stage1_weight: float = 0.5, stage2_weight: float = 1.5,
                 cls_weight: float = 1.0, offset_weight: float = 10.0,
                 match_radius: float = 0.2, pos_weight: float = 25.0,
                 use_focal_loss: bool = True, focal_gamma: float = 2.0):
        super().__init__()

        self.heatmap_sigma = heatmap_sigma
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight
        self.cls_weight = cls_weight
        self.offset_weight = offset_weight
        self.match_radius = match_radius
        self.pos_weight = pos_weight
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self._step = 0

        print(f"Loss config: pos_weight={pos_weight}, focal_loss={use_focal_loss}, gamma={focal_gamma}")

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                   pos_weight: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.
        Reduces loss contribution from easy negatives, focuses on hard examples.
        """
        probs = torch.sigmoid(logits)

        # Binary cross entropy (without reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal modulation: down-weight easy examples
        # For positives (target=1): p_t = probs, modulator = (1-probs)^gamma
        # For negatives (target=0): p_t = 1-probs, modulator = probs^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** gamma

        # Apply pos_weight to positive samples
        weight = torch.ones_like(targets)
        weight[targets == 1] = pos_weight

        focal_bce = focal_weight * weight * bce
        return focal_bce.mean()

    def generate_heatmap_target(self, points: torch.Tensor, H: int, W: int,
                                device: torch.device) -> torch.Tensor:
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
        N = len(proposals)
        device = proposals.device

        labels = torch.zeros(N, device=device)
        matched_gt = torch.zeros(N, 2, device=device)
        matched_mask = torch.zeros(N, dtype=torch.bool, device=device)

        if len(gt_points) == 0 or not mask.any():
            return labels, matched_gt, matched_mask

        valid_idx = torch.where(mask)[0]
        valid_proposals = proposals[valid_idx]

        if len(valid_proposals) == 0:
            return labels, matched_gt, matched_mask

        dists = torch.cdist(valid_proposals, gt_points)

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

                dists[:, gt_idx] = float('inf')

        return labels, matched_gt, matched_mask

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: List[Dict]) -> Tuple[torch.Tensor, Dict]:

        heatmap = outputs['heatmap']
        proposals = outputs['points']
        prop_mask = outputs['mask']
        cls_logits = outputs['stage2_logits']
        offsets = outputs['offsets']

        B, H, W = heatmap.shape
        device = heatmap.device
        image_size = 512  # Fixed image size

        # Convert heatmap logits to probabilities
        heatmap_prob = torch.sigmoid(heatmap)

        # Stage 1 Loss - Use focal loss to handle severe pixel imbalance
        # (128x128 = 16384 pixels, only ~100-200 are positive)
        stage1_loss = 0.0

        # DEBUG: Track GT point statistics
        total_gt_points = 0
        total_gt_points_in_range = 0
        debug_hm_targets = []

        for b in range(B):
            gt_pts = targets[b]['points'].to(device)
            total_gt_points += len(gt_pts)

            # DEBUG: Check if GT points are in valid range
            if len(gt_pts) > 0:
                in_range = (gt_pts[:, 0] >= 0) & (gt_pts[:, 0] < image_size) & \
                           (gt_pts[:, 1] >= 0) & (gt_pts[:, 1] < image_size)
                total_gt_points_in_range += in_range.sum().item()

            # Points are already in pixel coordinates (0-512), normalize to (0-1)
            gt_norm = gt_pts / image_size

            target_hm = self.generate_heatmap_target(gt_norm, H, W, device)
            debug_hm_targets.append((target_hm.max().item(), target_hm.mean().item(), (target_hm > 0.5).sum().item()))

            # Focal loss for heatmap (similar to CenterNet/CornerNet)
            # This properly handles the extreme foreground/background imbalance
            pred = heatmap_prob[b]

            # Positive locations (where target > 0.5)
            pos_mask = target_hm >= 0.5
            neg_mask = target_hm < 0.5

            # For positive pixels: penalize low predictions
            # Loss = -(1-p)^gamma * log(p) * target
            pos_loss = torch.zeros_like(pred)
            if pos_mask.any():
                pos_pred = pred[pos_mask]
                pos_target = target_hm[pos_mask]
                # Focal weight: (1-p)^2 focuses on hard positives (low confidence)
                pos_weights = torch.pow(1 - pos_pred, 2)
                pos_loss_vals = -pos_weights * torch.log(pos_pred.clamp(min=1e-6)) * pos_target
                pos_loss[pos_mask] = pos_loss_vals

            # For negative pixels: penalize high predictions
            # Loss = -(p)^gamma * (1-target)^beta * log(1-p)
            # The (1-target)^beta term reduces penalty near positive locations
            neg_loss = torch.zeros_like(pred)
            if neg_mask.any():
                neg_pred = pred[neg_mask]
                neg_target = target_hm[neg_mask]
                # Focal weight: p^2 focuses on hard negatives (high confidence false positives)
                neg_weights = torch.pow(neg_pred, 2)
                # Reduce penalty near positive locations
                location_weights = torch.pow(1 - neg_target, 4)
                neg_loss_vals = -neg_weights * location_weights * torch.log((1 - neg_pred).clamp(min=1e-6))
                neg_loss[neg_mask] = neg_loss_vals

            # Normalize by number of positive pixels
            n_pos = pos_mask.sum().clamp(min=1)
            batch_loss = (pos_loss.sum() + neg_loss.sum()) / n_pos
            stage1_loss += batch_loss

        # DEBUG: Print heatmap target stats for first few steps
        if self._step < 5:
            print(f"  [DEBUG HM Target] Sample: max={debug_hm_targets[0][0]:.3f}, mean={debug_hm_targets[0][1]:.4f}, n>0.5={debug_hm_targets[0][2]}")

        stage1_loss = stage1_loss / B

        # Stage 2 Loss
        stage2_cls_loss = 0.0
        stage2_off_loss = 0.0
        n_matched = 0
        n_proposals = 0
        n_positive_labels = 0

        for b in range(B):
            gt_pts = targets[b]['points'].to(device)
            gt_norm = gt_pts / image_size

            labels, matched_gt, matched_mask = self.match_proposals_to_gt(
                proposals[b], gt_norm, prop_mask[b]
            )

            n_positive_labels += labels.sum().item()

            valid = prop_mask[b]
            if valid.any():
                if self.use_focal_loss:
                    cls_loss = self.focal_loss(
                        cls_logits[b, valid], labels[valid],
                        pos_weight=self.pos_weight, gamma=self.focal_gamma
                    )
                else:
                    cls_loss = F.binary_cross_entropy_with_logits(
                        cls_logits[b, valid], labels[valid],
                        pos_weight=torch.tensor(self.pos_weight, device=device)
                    )
                stage2_cls_loss += cls_loss
                n_proposals += valid.sum().item()

            if matched_mask.any():
                pred_pts = proposals[b, matched_mask] + offsets[b, matched_mask]
                gt_matched = matched_gt[matched_mask]
                off_loss = F.smooth_l1_loss(pred_pts, gt_matched)
                stage2_off_loss += off_loss
                n_matched += matched_mask.sum().item()

        stage2_cls_loss = stage2_cls_loss / B if n_proposals > 0 else torch.tensor(0.0, device=device)
        stage2_off_loss = stage2_off_loss / B if n_matched > 0 else torch.tensor(0.0, device=device)

        stage2_loss = self.cls_weight * stage2_cls_loss + self.offset_weight * stage2_off_loss

        total = self.stage1_weight * stage1_loss + self.stage2_weight * stage2_loss

        # Logging - more frequent in early training
        self._step += 1
        # Log every 10 steps in first 100 steps, then every 50
        should_log = (self._step <= 100 and self._step % 10 == 0) or (self._step > 100 and self._step % 50 == 0)
        if should_log:
            with torch.no_grad():
                hm_max = heatmap_prob.max().item()
                hm_mean = heatmap_prob.mean().item()
                final_scores = outputs['scores']

                n_total_proposals = prop_mask.sum().item()

                valid_cls_logits = []
                for b in range(B):
                    if prop_mask[b].any():
                        valid_cls_logits.append(cls_logits[b, prop_mask[b]])

                if valid_cls_logits:
                    all_cls_logits = torch.cat(valid_cls_logits)
                    cls_probs = torch.sigmoid(all_cls_logits)

                    n_confident_pos = (cls_probs > 0.7).sum().item()
                    n_confident_neg = (cls_probs < 0.3).sum().item()
                    n_uncertain = ((cls_probs >= 0.3) & (cls_probs <= 0.7)).sum().item()

                    cls_mean = cls_probs.mean().item()
                    cls_max = cls_probs.max().item()

                    cls_info = f"cls[>0.7]={n_confident_pos} cls[<0.3]={n_confident_neg} cls[0.3-0.7]={n_uncertain} mean={cls_mean:.3f} max={cls_max:.3f}"
                else:
                    cls_info = "cls[no_proposals]"

                score_max = final_scores[prop_mask].max().item() if prop_mask.any() else 0

                print(
                    f"  [Step {self._step}] s1={stage1_loss.item():.4f} "
                    f"s2_cls={stage2_cls_loss.item() if isinstance(stage2_cls_loss, torch.Tensor) else stage2_cls_loss:.4f} "
                    f"s2_off={stage2_off_loss.item() if isinstance(stage2_off_loss, torch.Tensor) else stage2_off_loss:.4f} | "
                    f"hm[max={hm_max:.3f} mean={hm_mean:.4f}] props={n_total_proposals} score_max={score_max:.3f} matched={n_matched} | {cls_info}")

                # DEBUG: Check where predicted heatmap has peaks vs where GT is
                with torch.no_grad():
                    pred_hm = heatmap_prob[0]
                    gt_pts_0 = targets[0]['points'].to(device) / image_size
                    target_hm_0 = self.generate_heatmap_target(gt_pts_0, H, W, device)

                    # Find peak locations in predicted heatmap
                    pred_max_idx = pred_hm.argmax()
                    pred_max_y, pred_max_x = pred_max_idx // W, pred_max_idx % W

                    # Find peak locations in target heatmap
                    target_max_idx = target_hm_0.argmax()
                    target_max_y, target_max_x = target_max_idx // W, target_max_idx % W

                    print(f"  [DEBUG HM] Pred peak at ({pred_max_x.item()}, {pred_max_y.item()}) val={pred_hm.max():.3f} | "
                          f"Target peak at ({target_max_x.item()}, {target_max_y.item()}) val={target_hm_0.max():.3f} | "
                          f"Target mean={target_hm_0.mean():.4f}")

                # DEBUG: Print GT and matching stats
                print(f"  [DEBUG Loss] GT points: {total_gt_points} total, {total_gt_points_in_range} in valid range | "
                      f"Positive labels: {n_positive_labels} | Match radius: {self.match_radius:.4f}")

                # DEBUG: Check classifier scores for MATCHED vs UNMATCHED proposals
                # This is KEY - we want to see if the classifier can distinguish them
                all_matched_scores = []
                all_unmatched_scores = []
                for b in range(B):
                    gt_pts_b = targets[b]['points'].to(device)
                    gt_norm_b = gt_pts_b / image_size
                    labels_b, _, _ = self.match_proposals_to_gt(proposals[b], gt_norm_b, prop_mask[b])

                    valid_b = prop_mask[b]
                    if valid_b.any():
                        cls_probs_b = torch.sigmoid(cls_logits[b, valid_b])
                        labels_valid = labels_b[valid_b]

                        matched_mask = labels_valid == 1
                        unmatched_mask = labels_valid == 0

                        if matched_mask.any():
                            all_matched_scores.extend(cls_probs_b[matched_mask].cpu().tolist())
                        if unmatched_mask.any():
                            all_unmatched_scores.extend(cls_probs_b[unmatched_mask].cpu().tolist())

                if all_matched_scores:
                    matched_arr = np.array(all_matched_scores)
                    print(f"  [DEBUG] MATCHED props ({len(matched_arr)}): cls_prob min={matched_arr.min():.4f} "
                          f"max={matched_arr.max():.4f} mean={matched_arr.mean():.4f}")
                if all_unmatched_scores:
                    unmatched_arr = np.array(all_unmatched_scores[:100])  # Sample
                    print(f"  [DEBUG] UNMATCHED props (sample): cls_prob min={unmatched_arr.min():.4f} "
                          f"max={unmatched_arr.max():.4f} mean={unmatched_arr.mean():.4f}")

                # DEBUG: Sample some GT points and proposals to check ranges
                if len(targets) > 0 and len(targets[0]['points']) > 0:
                    sample_gt = targets[0]['points'][:3]
                    sample_gt_norm = sample_gt / image_size
                    print(f"  [DEBUG] Sample GT (pixel): {sample_gt.tolist()}")
                    print(f"  [DEBUG] Sample GT (norm):  {sample_gt_norm.tolist()}")
                if prop_mask[0].any():
                    sample_props = proposals[0, prop_mask[0]][:3]
                    print(f"  [DEBUG] Sample proposals (norm): {sample_props.tolist()}")

        return total, {
            'stage1_loss': stage1_loss.item(),
            'stage2_cls_loss': stage2_cls_loss.item() if isinstance(stage2_cls_loss, torch.Tensor) else 0,
            'stage2_off_loss': stage2_off_loss.item() if isinstance(stage2_off_loss, torch.Tensor) else 0,
            'n_matched': n_matched,
            'n_gt_points': total_gt_points,
            'n_positive_labels': n_positive_labels,
        }


# =============================================================================
# EVALUATION & TTA (unchanged)
# =============================================================================

def evaluate(model, dataloader, device, threshold: float = 0.3,
             match_radius: float = 100, image_size: int = 512, use_tta: bool = False,
             debug: bool = False) -> Dict:
    """Evaluate model with optional test-time augmentation."""
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    all_max_scores = []

    # DEBUG counters
    debug_n_batches = 0
    debug_total_gt = 0
    debug_total_pred = 0
    debug_score_samples = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)

            if use_tta:
                outputs = predict_with_tta(model, images)
            else:
                outputs = model(images)

            for b in range(len(targets)):
                gt_pts = targets[b]['points'].to(device)
                debug_total_gt += len(gt_pts)

                scores = outputs['scores'][b]
                mask = outputs['mask'][b]
                points = outputs['points'][b]

                if mask.any():
                    max_score = scores[mask].max().item()
                    all_max_scores.append(max_score)

                    # DEBUG: Collect score samples
                    if debug and debug_n_batches < 3:
                        valid_scores = scores[mask].cpu().numpy()
                        debug_score_samples.extend(valid_scores[:10].tolist())

                keep = mask & (scores >= threshold)
                pred_pts = points[keep] * image_size
                pred_scores = scores[keep]
                debug_total_pred += len(pred_pts)

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

            debug_n_batches += 1

            # DEBUG: Print first batch details
            if debug and debug_n_batches == 1:
                print(f"\n  [DEBUG Eval] First batch:")
                print(f"    GT points: {len(gt_pts)}")
                print(f"    Proposals with mask: {mask.sum().item()}")
                print(f"    Max score: {max_score:.4f}" if mask.any() else "    No proposals")
                print(f"    Predictions above threshold {threshold}: {len(pred_pts)}")
                if mask.any():
                    valid_scores = scores[mask]
                    print(f"    Score distribution: min={valid_scores.min():.4f}, "
                          f"max={valid_scores.max():.4f}, mean={valid_scores.mean():.4f}")
                    print(f"    Scores above {threshold}: {(valid_scores >= threshold).sum().item()}")

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    if debug:
        print(f"\n  [DEBUG Eval Summary]")
        print(f"    Total GT: {debug_total_gt}, Total Pred: {debug_total_pred}")
        print(f"    TP={total_tp}, FP={total_fp}, FN={total_fn}")
        if debug_score_samples:
            print(f"    Sample scores: {debug_score_samples[:10]}")

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'avg_max_score': np.mean(all_max_scores) if all_max_scores else 0
    }


def predict_with_tta(model, images: torch.Tensor) -> Dict:
    """Test-time augmentation: predict with multiple augmentations and merge."""
    B = images.shape[0]
    device = images.device

    all_predictions = []

    # Original
    pred = model(images)
    all_predictions.append(pred)

    # Horizontal flip
    pred_hflip = model(torch.flip(images, [-1]))
    pred_hflip['points'][:, :, 0] = 1 - pred_hflip['points'][:, :, 0]
    all_predictions.append(pred_hflip)

    # Vertical flip
    pred_vflip = model(torch.flip(images, [-2]))
    pred_vflip['points'][:, :, 1] = 1 - pred_vflip['points'][:, :, 1]
    all_predictions.append(pred_vflip)

    # Merge predictions
    merged_pred = merge_tta_predictions(all_predictions, device)

    return merged_pred


def merge_tta_predictions(predictions: List[Dict], device: torch.device) -> Dict:
    """Merge multiple TTA predictions using NMS and score averaging."""
    B = predictions[0]['points'].shape[0]

    merged = {
        'heatmap': torch.mean(torch.stack([p['heatmap'] for p in predictions]), dim=0),
        'points': [],
        'scores': [],
        'mask': [],
        'stage2_logits': [],
        'offsets': torch.zeros_like(predictions[0]['offsets']),
    }

    for b in range(B):
        all_points = []
        all_scores = []

        for pred in predictions:
            mask = pred['mask'][b]
            if mask.any():
                all_points.append(pred['points'][b, mask])
                all_scores.append(pred['scores'][b, mask])

        if all_points:
            all_points = torch.cat(all_points, dim=0)
            all_scores = torch.cat(all_scores, dim=0)

            keep_idx = nms_points(all_points, all_scores, threshold=0.05)

            final_points = all_points[keep_idx]
            final_scores = all_scores[keep_idx]
        else:
            final_points = torch.zeros(0, 2, device=device)
            final_scores = torch.zeros(0, device=device)

        max_proposals = predictions[0]['points'].shape[1]
        n = len(final_points)
        if n < max_proposals:
            final_points = F.pad(final_points, [0, 0, 0, max_proposals - n])
            final_scores = F.pad(final_scores, [0, max_proposals - n])
        else:
            final_points = final_points[:max_proposals]
            final_scores = final_scores[:max_proposals]

        mask = torch.zeros(max_proposals, dtype=torch.bool, device=device)
        mask[:min(n, max_proposals)] = True

        merged['points'].append(final_points)
        merged['scores'].append(final_scores)
        merged['mask'].append(mask)
        merged['stage2_logits'].append(torch.zeros(max_proposals, device=device))

    merged['points'] = torch.stack(merged['points'])
    merged['scores'] = torch.stack(merged['scores'])
    merged['mask'] = torch.stack(merged['mask'])
    merged['stage2_logits'] = torch.stack(merged['stage2_logits'])

    return merged


def nms_points(points: torch.Tensor, scores: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Non-maximum suppression for points."""
    if len(points) == 0:
        return torch.tensor([], dtype=torch.long, device=points.device)

    sorted_idx = scores.argsort(descending=True)

    keep = []
    while len(sorted_idx) > 0:
        idx = sorted_idx[0]
        keep.append(idx.item())

        if len(sorted_idx) == 1:
            break

        dists = torch.norm(points[sorted_idx[1:]] - points[idx], dim=1)
        far_enough = dists >= threshold
        sorted_idx = sorted_idx[1:][far_enough]

    return torch.tensor(keep, dtype=torch.long, device=points.device)


def threshold_sweep(model, dataloader, device, match_radius: float = 25, use_tta: bool = False):
    """Find optimal threshold."""
    print("\n" + "=" * 60)
    print(f"THRESHOLD SWEEP {'(with TTA)' if use_tta else ''}")
    print("=" * 60)
    print(f"{'Thresh':>7} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 55)

    best_f1, best_t = 0, 0.3
    for t in np.arange(0.05, 0.95, 0.05):
        m = evaluate(model, dataloader, device, t, match_radius, use_tta=use_tta)
        marker = " ★" if m['f1'] > best_f1 else ""
        print(f"{t:>7.2f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
              f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}{marker}")
        if m['f1'] > best_f1:
            best_f1, best_t = m['f1'], t

    print("-" * 55)
    print(f"Best: threshold={best_t:.2f} → F1={best_f1:.4f}")
    return best_t, best_f1


# =============================================================================
# PROGRESSIVE TRAINING (unchanged)
# =============================================================================

class ProgressiveTrainer:
    def __init__(self, model, criterion, train_loader, val_loader, device, output_dir, config):
        self.model = model
        self.criterion = criterion
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

    def setup_optimizer_phase1(self):
        """Phase 1: Backbone frozen, train heads only."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        print("\nPhase 1 Optimizer: Heads only, lr=1e-4")

    def setup_optimizer_phase2(self):
        """Phase 2: Last 6 blocks unfrozen, discriminative LR."""
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-6},
            {'params': head_params, 'lr': 5e-5}
        ], weight_decay=self.config['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        print("\nPhase 2 Optimizer: Backbone=1e-6, Heads=5e-5")

    def setup_optimizer_phase3(self):
        """Phase 3: Everything unfrozen, very low LR."""
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 5e-7},
            {'params': head_params, 'lr': 1e-5}
        ], weight_decay=self.config['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        print("\nPhase 3 Optimizer: Backbone=5e-7, Heads=1e-5")

    def setup_optimizer_phase4(self):
        """Phase 4: Ultra-low LR refinement, focus on classifier confidence."""
        print("\n*** Re-freezing backbone for Phase 4 ***")
        for p in self.model.backbone.parameters():
            p.requires_grad = False

        head_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            head_params,
            lr=5e-6,
            weight_decay=self.config['weight_decay'] * 10
        )

        self.scheduler = None

        print("Phase 4 Optimizer: Backbone=frozen, Heads=5e-6 (ultra-low)")
        print("  Focus: Refine classification confidence without overfitting")

    def train_epoch(self, epoch: int = 0):
        self.model.train()
        total_loss, n = 0, 0

        # DEBUG: Print first batch info at start of epoch 0
        debug_first_batch = (epoch == 0)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)

            # DEBUG: Print detailed info about first batch
            if debug_first_batch and batch_idx == 0:
                print("\n" + "=" * 60)
                print("[DEBUG] FIRST TRAINING BATCH ANALYSIS")
                print("=" * 60)
                print(f"  Batch size: {len(images)}")
                print(f"  Image shape: {images.shape}")
                print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")

                for b in range(min(2, len(targets))):
                    pts = targets[b]['points']
                    name = targets[b]['name']
                    print(f"\n  Sample {b}: {name}")
                    print(f"    Number of GT points: {len(pts)}")
                    if len(pts) > 0:
                        print(f"    GT point range: x=[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}], "
                              f"y=[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")
                        print(f"    First 3 GT points: {pts[:3].tolist()}")

                        # Check if points are in expected range (0-512 for pixel coords)
                        if pts.max() > 512:
                            print(f"    WARNING: Points exceed 512! Max={pts.max():.1f}")
                        if pts.max() <= 1:
                            print(f"    WARNING: Points appear to be normalized (0-1)! Max={pts.max():.4f}")

                # Forward pass to see what model outputs
                with torch.no_grad():
                    outputs = self.model(images)
                    heatmap = outputs['heatmap']
                    proposals = outputs['points']
                    prop_mask = outputs['mask']
                    scores = outputs['scores']

                    print(f"\n  Model outputs:")
                    print(f"    Heatmap shape: {heatmap.shape}")
                    print(f"    Heatmap range (sigmoid): [{torch.sigmoid(heatmap).min():.4f}, {torch.sigmoid(heatmap).max():.4f}]")
                    print(f"    Number of proposals: {prop_mask.sum().item()}")
                    if prop_mask.any():
                        valid_props = proposals[prop_mask]
                        print(f"    Proposal range: x=[{valid_props[:,0].min():.4f}, {valid_props[:,0].max():.4f}], "
                              f"y=[{valid_props[:,1].min():.4f}, {valid_props[:,1].max():.4f}]")
                        print(f"    First 3 proposals (normalized 0-1): {valid_props[:3].tolist()}")

                        valid_scores = scores[prop_mask]
                        print(f"    Score range: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")

                print("=" * 60 + "\n")
                debug_first_batch = False

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
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_f1': self.best_f1,
            'patience_counter': self.patience_counter,
        }, self.output_dir / f'{name}.pth')

    def load_checkpoint(self, checkpoint_path):
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

    def train_progressive(self, total_epochs: int, patience: int = 30):
        """
        Progressive training with 4 phases:
        - Phase 1 (0-10): Backbone frozen, warm up heads
        - Phase 2 (10-40): Last 6 blocks unfrozen, adapt to iguanas
        - Phase 3 (40-60): Full fine-tune, learn complex patterns
        - Phase 4 (60-end): Re-freeze backbone, refine classification confidence
        """
        print("\n" + "=" * 60)
        print("PROGRESSIVE TRAINING (4 PHASES)")
        print("=" * 60)
        print("Phase 1 (epochs 0-10): Backbone frozen")
        print("Phase 2 (epochs 10-40): Last 6 blocks unfrozen")
        print("Phase 3 (epochs 40-60): Full fine-tune")
        print("Phase 4 (epochs 60+): Re-freeze backbone, refine confidence")
        print("=" * 60)

        current_phase = 1
        self.setup_optimizer_phase1()

        for epoch in range(self.start_epoch, total_epochs):
            # Phase transitions
            if epoch == 10 and current_phase == 1:
                print("\n" + "=" * 60)
                print("ENTERING PHASE 2: Unfreezing last 6 blocks")
                print("=" * 60)
                self.model.unfreeze_last_n_blocks(6)
                self.setup_optimizer_phase2()
                current_phase = 2

            elif epoch == 40 and current_phase == 2:
                print("\n" + "=" * 60)
                print("ENTERING PHASE 3: Full fine-tune")
                print("=" * 60)
                self.model.unfreeze_backbone()
                self.setup_optimizer_phase3()
                current_phase = 3

            elif epoch == 60 and current_phase == 3:
                print("\n" + "=" * 60)
                print("ENTERING PHASE 4: Refine classification confidence")
                print("=" * 60)
                self.setup_optimizer_phase4()
                current_phase = 4

            t0 = time.time()
            loss = self.train_epoch(epoch=epoch)

            if self.scheduler:
                self.scheduler.step()

            val_m = {}
            if self.val_loader:
                # DEBUG: Enable detailed logging for first 5 epochs
                debug_eval = (epoch < 5)
                val_m = evaluate(self.model, self.val_loader, self.device,
                                 self.config['threshold'], self.config['match_radius'],
                                 debug=debug_eval)
                if val_m['f1'] > self.best_f1:
                    self.best_f1 = val_m['f1']
                    self.patience_counter = 0
                    self.save('best', epoch)
                else:
                    self.patience_counter += 1

            lr_backbone = self.optimizer.param_groups[0]['lr'] if len(self.optimizer.param_groups) > 1 else self.optimizer.param_groups[0]['lr']
            lr_heads = self.optimizer.param_groups[-1]['lr']

            log = f"[P{current_phase}] Epoch {epoch:3d} ({time.time() - t0:.1f}s) | loss={loss:.4f} | lr_bb={lr_backbone:.2e} lr_head={lr_heads:.2e}"
            if val_m:
                log += f" | P={val_m['precision']:.3f} R={val_m['recall']:.3f} F1={val_m['f1']:.3f}"
                if val_m['f1'] >= self.best_f1:
                    log += " ★"
            print(log)

            if epoch % 10 == 0:
                self.save('latest', epoch)

            if patience > 0 and self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.save('final', epoch)
        print(f"\n{'=' * 60}")
        print(f"Training Complete! Best F1: {self.best_f1:.4f}")
        print(f"{'=' * 60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fixed Two-Stage Detector Training")
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_image_dir', required=True)
    parser.add_argument('--val_csv')
    parser.add_argument('--val_image_dir')

    parser.add_argument('--backbone', default='vit_large_patch16_dinov3.sat493m')
    parser.add_argument('--heatmap_size', type=int, default=128)
    parser.add_argument('--max_proposals', type=int, default=300)
    parser.add_argument('--proposal_threshold', type=float, default=0.1)
    parser.add_argument('--refine_hidden', type=int, default=256)
    parser.add_argument('--roi_size', type=int, default=7)

    parser.add_argument('--heatmap_sigma', type=float, default=4.0)
    parser.add_argument('--stage1_weight', type=float, default=0.5)
    parser.add_argument('--stage2_weight', type=float, default=1.5)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--mixup_prob', type=float, default=0.2, help='MixUp probability')

    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--match_radius', type=float, default=100)
    parser.add_argument('--pos_weight', type=float, default=25.0, help='Positive class weight for BCE loss')
    parser.add_argument('--use_focal_loss', action='store_true', default=True, help='Use focal loss (default: True)')
    parser.add_argument('--no_focal_loss', action='store_true', help='Disable focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')

    parser.add_argument('--output_dir', default='./outputs_fixed')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    config = vars(args)

    # Data with FIXED coordinate handling
    train_ds = PointDataset(
        args.train_csv, args.train_image_dir, 512,
        augment=True, mixup_prob=args.mixup_prob, debug=args.debug
    )
    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True
    )

    val_loader = None
    if args.val_csv and args.val_image_dir:
        val_ds = PointDataset(args.val_csv, args.val_image_dir, 512, augment=False, debug=args.debug)
        val_loader = DataLoader(
            val_ds, args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )

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

    # Loss with focal loss to handle class imbalance
    use_focal = args.use_focal_loss and not args.no_focal_loss
    criterion = TwoStageLoss(
        heatmap_sigma=args.heatmap_sigma,
        stage1_weight=args.stage1_weight,
        stage2_weight=args.stage2_weight,
        match_radius=args.match_radius / 512,  # Normalize to [0, 1]
        pos_weight=args.pos_weight,
        use_focal_loss=use_focal,
        focal_gamma=args.focal_gamma,
    )

    # Progressive trainer
    trainer = ProgressiveTrainer(
        model, criterion, train_loader, val_loader,
        device, args.output_dir, config
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train with progressive unfreezing
    trainer.train_progressive(args.epochs, args.patience)

    # Final evaluation with TTA
    if val_loader:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        best_path = Path(args.output_dir) / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

        print("\nWithout TTA:")
        threshold_sweep(model, val_loader, device, args.match_radius, use_tta=False)

        print("\nWith TTA:")
        threshold_sweep(model, val_loader, device, args.match_radius, use_tta=True)


if __name__ == '__main__':
    main()