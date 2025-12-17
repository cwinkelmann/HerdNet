#!/usr/bin/env python3
"""
Pretraining script for Iguana Classifier using GeoTIFF orthomosaics.

Features:
- Streams tiles directly from GeoTIFFs using windowed reads (memory efficient)
- Reads annotations from GeoJSON files (transforms geo-coords to pixel coords)
- Coverage tracking ensures all parts of orthomosaics are seen during training
- Supports multiple GeoTIFFs and GeoJSONs

Usage:
    python pretrain_geotiff.py \
        --geotiffs ortho1.tif ortho2.tif \
        --geojsons annot1.geojson annot2.geojson \
        --output_dir ./pretrain_outputs

The script pairs GeoTIFFs with GeoJSONs by order (first tif with first geojson, etc.)
or by matching filenames if --match_by_name is specified.
"""

import os
import argparse
import random
import time
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image
import timm

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import rowcol
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio not installed. Install with: pip install rasterio")

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("WARNING: geopandas not installed. Install with: pip install geopandas")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    raise ImportError("albumentations required: pip install albumentations")


# =============================================================================
# GEOTIFF TILE DATASET
# =============================================================================

class GeoTiffTileDataset(Dataset):
    """
    Dataset that samples random tiles from GeoTIFF orthomosaics.
    
    Features:
    - Windowed reads for memory efficiency (doesn't load full image)
    - Converts GeoJSON annotations to pixel coordinates
    - Coverage tracking ensures all areas are eventually seen
    - Balanced sampling between positive (with annotations) and negative tiles
    
    Args:
        geotiff_paths: List of paths to GeoTIFF files
        geojson_paths: List of paths to GeoJSON annotation files (matched by order)
        crop_size: Size of tiles to extract (default: 518 for DINOv2)
        tiles_per_epoch: Number of tiles to sample per epoch
        positive_ratio: Ratio of positive (has annotations) tiles to sample
        coverage_grid_size: Size of coverage tracking grid cells
        augment: Enable data augmentation
        patch_size: ViT patch size for creating patch labels
    """
    
    def __init__(
        self,
        geotiff_paths: List[str],
        geojson_paths: List[str],
        crop_size: int = 518,
        tiles_per_epoch: int = 10000,
        positive_ratio: float = 0.5,
        coverage_grid_size: int = 256,
        augment: bool = True,
        patch_size: int = 14,
        min_edge_margin: int = 30,
        seed: int = 42,
    ):
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required for GeoTIFF support")
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas is required for GeoJSON support")
        
        self.crop_size = crop_size
        self.tiles_per_epoch = tiles_per_epoch
        self.positive_ratio = positive_ratio
        self.coverage_grid_size = coverage_grid_size
        self.augment = augment
        self.patch_size = patch_size
        self.grid_size = crop_size // patch_size
        self.min_edge_margin = min_edge_margin
        self.rng = np.random.RandomState(seed)
        
        # Normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Load GeoTIFF metadata (don't load pixels yet)
        self.geotiffs = []
        self.annotations = {}  # geotiff_idx -> list of (px, py) pixel coordinates
        self.image_bounds = {}  # geotiff_idx -> (width, height)
        
        print(f"\nLoading {len(geotiff_paths)} GeoTIFFs and {len(geojson_paths)} GeoJSONs...")
        
        for idx, (tif_path, geojson_path) in enumerate(zip(geotiff_paths, geojson_paths)):
            self._load_geotiff_metadata(idx, tif_path, geojson_path)
        
        # Coverage tracking: grid of visited cells per image
        self.coverage_grids = {}
        self._init_coverage_grids()
        
        # Precompute sampling weights based on image sizes
        self._compute_sampling_weights()
        
        # Build augmentation pipeline
        self.photometric_transform = self._build_photometric() if augment else None
        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])
        
        # Statistics
        total_points = sum(len(pts) for pts in self.annotations.values())
        total_area = sum(w * h for w, h in self.image_bounds.values())
        print(f"\nDataset summary:")
        print(f"  GeoTIFFs: {len(self.geotiffs)}")
        print(f"  Total annotations: {total_points}")
        print(f"  Total area: {total_area / 1e6:.1f} megapixels")
        print(f"  Tiles per epoch: {tiles_per_epoch}")
        print(f"  Positive ratio: {positive_ratio}")
        print(f"  Coverage grid size: {coverage_grid_size}")
    
    def _load_geotiff_metadata(self, idx: int, tif_path: str, geojson_path: str):
        """Load GeoTIFF metadata and corresponding GeoJSON annotations."""
        tif_path = Path(tif_path)
        geojson_path = Path(geojson_path)
        
        if not tif_path.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
        
        # Load GeoTIFF metadata
        with rasterio.open(tif_path) as src:
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            n_bands = src.count
        
        self.geotiffs.append({
            'path': str(tif_path),
            'width': width,
            'height': height,
            'transform': transform,
            'crs': crs,
            'bounds': bounds,
            'n_bands': n_bands,
        })
        self.image_bounds[idx] = (width, height)
        
        print(f"\n  [{idx}] {tif_path.name}")
        print(f"      Size: {width} x {height} ({width * height / 1e6:.1f} MP)")
        print(f"      CRS: {crs}")
        print(f"      Bands: {n_bands}")
        
        # Load GeoJSON annotations
        gdf = gpd.read_file(geojson_path)
        
        # Reproject to GeoTIFF CRS if needed
        if gdf.crs != crs:
            print(f"      Reprojecting annotations from {gdf.crs} to {crs}")
            gdf = gdf.to_crs(crs)
        
        # Convert geo-coordinates to pixel coordinates
        points = []
        for geom in gdf.geometry:
            if geom is None:
                continue
            if geom.geom_type == 'Point':
                # Convert geo coords to pixel coords
                px, py = ~transform * (geom.x, geom.y)
                px, py = int(px), int(py)
                if 0 <= px < width and 0 <= py < height:
                    points.append((px, py))
            elif geom.geom_type == 'MultiPoint':
                for pt in geom.geoms:
                    px, py = ~transform * (pt.x, pt.y)
                    px, py = int(px), int(py)
                    if 0 <= px < width and 0 <= py < height:
                        points.append((px, py))
        
        self.annotations[idx] = np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)
        print(f"      Annotations: {len(points)} points")
    
    def _init_coverage_grids(self):
        """Initialize coverage tracking grids for each image."""
        for idx, (width, height) in self.image_bounds.items():
            n_cells_x = max(1, width // self.coverage_grid_size)
            n_cells_y = max(1, height // self.coverage_grid_size)
            self.coverage_grids[idx] = np.zeros((n_cells_y, n_cells_x), dtype=np.int32)
    
    def _compute_sampling_weights(self):
        """Compute sampling weights for each image based on area."""
        total_area = sum(w * h for w, h in self.image_bounds.values())
        self.image_weights = {
            idx: (w * h) / total_area 
            for idx, (w, h) in self.image_bounds.items()
        }
        self.image_indices = list(self.image_weights.keys())
        self.image_probs = [self.image_weights[i] for i in self.image_indices]
    
    def _build_photometric(self):
        """Build photometric augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(
                max_holes=8, max_height=64, max_width=64,
                min_holes=1, min_height=16, min_width=16,
                fill_value=0, p=0.3,
            ),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1, num_shadows_upper=3,
                shadow_dimension=5, p=0.2,
            ),
        ])
    
    def get_coverage_stats(self) -> Dict:
        """Get coverage statistics for all images."""
        stats = {}
        for idx, grid in self.coverage_grids.items():
            total_cells = grid.size
            visited_cells = (grid > 0).sum()
            coverage_pct = 100 * visited_cells / total_cells
            stats[idx] = {
                'total_cells': total_cells,
                'visited_cells': int(visited_cells),
                'coverage_pct': coverage_pct,
                'min_visits': int(grid.min()),
                'max_visits': int(grid.max()),
                'mean_visits': float(grid.mean()),
            }
        return stats
    
    def reset_coverage(self):
        """Reset coverage tracking (call at start of epoch if desired)."""
        for grid in self.coverage_grids.values():
            grid.fill(0)
    
    def _sample_positive_tile(self, img_idx: int) -> Tuple[int, int]:
        """Sample a tile that contains at least one annotation."""
        points = self.annotations[img_idx]
        width, height = self.image_bounds[img_idx]
        
        if len(points) == 0:
            return self._sample_random_tile(img_idx)
        
        # Pick a random point
        pt_idx = self.rng.randint(len(points))
        px, py = points[pt_idx]
        
        # Sample tile position that includes this point with margin
        x_min = max(0, int(px - self.crop_size + self.min_edge_margin))
        x_max = min(width - self.crop_size, int(px - self.min_edge_margin))
        y_min = max(0, int(py - self.crop_size + self.min_edge_margin))
        y_max = min(height - self.crop_size, int(py - self.min_edge_margin))
        
        x_min = min(x_min, max(0, width - self.crop_size))
        x_max = max(x_max, 0)
        y_min = min(y_min, max(0, height - self.crop_size))
        y_max = max(y_max, 0)
        
        crop_x = self.rng.randint(min(x_min, x_max), max(x_min, x_max) + 1)
        crop_y = self.rng.randint(min(y_min, y_max), max(y_min, y_max) + 1)
        
        return crop_x, crop_y
    
    def _sample_negative_tile(self, img_idx: int, max_attempts: int = 50) -> Optional[Tuple[int, int]]:
        """Sample a tile that contains no annotations."""
        points = self.annotations[img_idx]
        width, height = self.image_bounds[img_idx]
        
        if len(points) == 0:
            return self._sample_random_tile(img_idx)
        
        max_x = max(0, width - self.crop_size)
        max_y = max(0, height - self.crop_size)
        
        for _ in range(max_attempts):
            crop_x = self.rng.randint(0, max_x + 1) if max_x > 0 else 0
            crop_y = self.rng.randint(0, max_y + 1) if max_y > 0 else 0
            
            # Check if any point falls in this tile
            in_tile = (
                (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
                (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
            )
            
            if not in_tile.any():
                return crop_x, crop_y
        
        return None
    
    def _sample_random_tile(self, img_idx: int) -> Tuple[int, int]:
        """Sample a completely random tile."""
        width, height = self.image_bounds[img_idx]
        max_x = max(0, width - self.crop_size)
        max_y = max(0, height - self.crop_size)
        
        crop_x = self.rng.randint(0, max_x + 1) if max_x > 0 else 0
        crop_y = self.rng.randint(0, max_y + 1) if max_y > 0 else 0
        
        return crop_x, crop_y
    
    def _sample_coverage_aware_tile(self, img_idx: int) -> Tuple[int, int]:
        """Sample tile preferring less-visited areas."""
        width, height = self.image_bounds[img_idx]
        grid = self.coverage_grids[img_idx]
        
        # Find cells with minimum visits
        min_visits = grid.min()
        candidates = np.argwhere(grid == min_visits)
        
        if len(candidates) == 0:
            return self._sample_random_tile(img_idx)
        
        # Pick a random least-visited cell
        cell_idx = self.rng.randint(len(candidates))
        cell_y, cell_x = candidates[cell_idx]
        
        # Sample random position within this cell
        cell_x_px = cell_x * self.coverage_grid_size
        cell_y_px = cell_y * self.coverage_grid_size
        
        # Add some randomness within the cell
        offset_x = self.rng.randint(0, min(self.coverage_grid_size, width - cell_x_px - self.crop_size + 1))
        offset_y = self.rng.randint(0, min(self.coverage_grid_size, height - cell_y_px - self.crop_size + 1))
        
        crop_x = min(cell_x_px + offset_x, width - self.crop_size)
        crop_y = min(cell_y_px + offset_y, height - self.crop_size)
        
        return max(0, crop_x), max(0, crop_y)
    
    def _update_coverage(self, img_idx: int, crop_x: int, crop_y: int):
        """Update coverage grid for sampled tile."""
        grid = self.coverage_grids[img_idx]
        
        # Find which cells this tile overlaps
        cell_x_start = crop_x // self.coverage_grid_size
        cell_x_end = (crop_x + self.crop_size) // self.coverage_grid_size
        cell_y_start = crop_y // self.coverage_grid_size
        cell_y_end = (crop_y + self.crop_size) // self.coverage_grid_size
        
        # Clamp to grid bounds
        cell_x_end = min(cell_x_end + 1, grid.shape[1])
        cell_y_end = min(cell_y_end + 1, grid.shape[0])
        
        grid[cell_y_start:cell_y_end, cell_x_start:cell_x_end] += 1
    
    def _read_tile(self, img_idx: int, crop_x: int, crop_y: int) -> np.ndarray:
        """Read a tile from GeoTIFF using windowed read."""
        tif_info = self.geotiffs[img_idx]
        
        with rasterio.open(tif_info['path']) as src:
            # Create window
            window = Window(crop_x, crop_y, self.crop_size, self.crop_size)
            
            # Read RGB bands (assuming first 3 bands are RGB)
            # Handle both RGB and RGBA images
            n_bands = min(3, src.count)
            data = src.read(list(range(1, n_bands + 1)), window=window)
            
            # Handle edge cases (tile extends beyond image)
            if data.shape[1] < self.crop_size or data.shape[2] < self.crop_size:
                padded = np.zeros((3, self.crop_size, self.crop_size), dtype=data.dtype)
                padded[:n_bands, :data.shape[1], :data.shape[2]] = data
                data = padded
            
            # Transpose to HWC format
            data = np.transpose(data, (1, 2, 0))
            
            # Handle single-band or different band counts
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
            elif data.shape[2] == 2:
                data = np.concatenate([data, data[:, :, :1]], axis=2)
            
            # Normalize to 0-255 uint8 if needed
            if data.dtype != np.uint8:
                if data.max() > 255:
                    data = (data / data.max() * 255).astype(np.uint8)
                else:
                    data = data.astype(np.uint8)
        
        return data
    
    def _get_points_in_tile(self, img_idx: int, crop_x: int, crop_y: int) -> np.ndarray:
        """Get all annotation points within a tile (in tile coordinates)."""
        points = self.annotations[img_idx]
        
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        in_tile = (
            (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
            (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
        )
        
        points_in_tile = points[in_tile].copy()
        points_in_tile[:, 0] -= crop_x
        points_in_tile[:, 1] -= crop_y
        
        return points_in_tile
    
    def _create_patch_labels(self, points_in_tile: np.ndarray) -> torch.Tensor:
        """Create binary patch labels from points."""
        patch_labels = torch.zeros(self.grid_size, self.grid_size)
        
        if len(points_in_tile) == 0:
            return patch_labels
        
        for pt in points_in_tile:
            px, py = pt
            patch_x = int(px / self.patch_size)
            patch_y = int(py / self.patch_size)
            
            patch_x = max(0, min(patch_x, self.grid_size - 1))
            patch_y = max(0, min(patch_y, self.grid_size - 1))
            
            patch_labels[patch_y, patch_x] = 1.0
        
        return patch_labels
    
    def __len__(self):
        return self.tiles_per_epoch
    
    def __getitem__(self, idx):
        # Select image based on area-weighted probability
        img_idx = self.rng.choice(self.image_indices, p=self.image_probs)
        
        # Decide positive or negative
        want_positive = self.rng.random() < self.positive_ratio
        
        # Sample tile position
        if want_positive and len(self.annotations[img_idx]) > 0:
            crop_x, crop_y = self._sample_positive_tile(img_idx)
        else:
            result = self._sample_negative_tile(img_idx)
            if result is not None:
                crop_x, crop_y = result
            elif len(self.annotations[img_idx]) > 0:
                crop_x, crop_y = self._sample_positive_tile(img_idx)
            else:
                crop_x, crop_y = self._sample_random_tile(img_idx)
        
        # Update coverage
        self._update_coverage(img_idx, crop_x, crop_y)
        
        # Read tile
        tile = self._read_tile(img_idx, crop_x, crop_y)
        
        # Get points in tile
        points_in_tile = self._get_points_in_tile(img_idx, crop_x, crop_y)
        label = 1.0 if len(points_in_tile) > 0 else 0.0
        
        # Create patch labels
        patch_labels = self._create_patch_labels(points_in_tile)
        
        # Apply augmentation
        if self.photometric_transform:
            tile = self.photometric_transform(image=tile)['image']
        
        # Normalize
        tile = self.normalize_transform(image=tile)['image']
        
        return tile, {
            'label': torch.tensor(label, dtype=torch.float32),
            'patch_labels': patch_labels,
            'points': torch.from_numpy(points_in_tile).float(),
            'img_idx': img_idx,
            'crop_x': crop_x,
            'crop_y': crop_y,
        }


class GeoTiffValidationDataset(Dataset):
    """
    Deterministic tiled validation dataset for GeoTIFFs.
    
    Pre-computes all tile positions for reproducible evaluation.
    """
    
    def __init__(
        self,
        geotiff_paths: List[str],
        geojson_paths: List[str],
        crop_size: int = 518,
        overlap: int = 0,
        patch_size: int = 14,
    ):
        if not HAS_RASTERIO or not HAS_GEOPANDAS:
            raise ImportError("rasterio and geopandas are required")
        
        self.crop_size = crop_size
        self.stride = crop_size - overlap
        self.patch_size = patch_size
        self.grid_size = crop_size // patch_size
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.normalize_transform = A.Compose([
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ])
        
        # Load GeoTIFFs and annotations
        self.geotiffs = []
        self.annotations = {}
        
        print(f"\nLoading validation GeoTIFFs...")
        for idx, (tif_path, geojson_path) in enumerate(zip(geotiff_paths, geojson_paths)):
            self._load_geotiff(idx, tif_path, geojson_path)
        
        # Pre-compute all tile positions
        self.tiles = []
        for idx, tif_info in enumerate(self.geotiffs):
            width, height = tif_info['width'], tif_info['height']
            
            n_tiles_x = max(1, (width - overlap) // self.stride)
            n_tiles_y = max(1, (height - overlap) // self.stride)
            
            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    crop_x = tx * self.stride
                    crop_y = ty * self.stride
                    self.tiles.append((idx, crop_x, crop_y))
        
        # Count positive/negative
        n_pos = sum(1 for idx, cx, cy in self.tiles if self._has_points(idx, cx, cy))
        print(f"\nValidation tiles: {len(self.tiles)} (pos: {n_pos}, neg: {len(self.tiles) - n_pos})")
    
    def _load_geotiff(self, idx: int, tif_path: str, geojson_path: str):
        """Load GeoTIFF metadata and annotations."""
        with rasterio.open(tif_path) as src:
            width, height = src.width, src.height
            transform = src.transform
            crs = src.crs
        
        self.geotiffs.append({
            'path': str(tif_path),
            'width': width,
            'height': height,
            'transform': transform,
            'crs': crs,
        })
        
        # Load annotations
        gdf = gpd.read_file(geojson_path)
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        
        points = []
        for geom in gdf.geometry:
            if geom is None:
                continue
            if geom.geom_type == 'Point':
                px, py = ~transform * (geom.x, geom.y)
                if 0 <= px < width and 0 <= py < height:
                    points.append((int(px), int(py)))
        
        self.annotations[idx] = np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)
        print(f"  [{idx}] {Path(tif_path).name}: {width}x{height}, {len(points)} annotations")
    
    def _has_points(self, img_idx: int, crop_x: int, crop_y: int) -> bool:
        points = self.annotations[img_idx]
        if len(points) == 0:
            return False
        in_tile = (
            (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
            (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
        )
        return in_tile.any()
    
    def _get_points_in_tile(self, img_idx: int, crop_x: int, crop_y: int) -> np.ndarray:
        points = self.annotations[img_idx]
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        in_tile = (
            (points[:, 0] >= crop_x) & (points[:, 0] < crop_x + self.crop_size) &
            (points[:, 1] >= crop_y) & (points[:, 1] < crop_y + self.crop_size)
        )
        pts = points[in_tile].copy()
        pts[:, 0] -= crop_x
        pts[:, 1] -= crop_y
        return pts
    
    def _read_tile(self, img_idx: int, crop_x: int, crop_y: int) -> np.ndarray:
        tif_info = self.geotiffs[img_idx]
        with rasterio.open(tif_info['path']) as src:
            window = Window(crop_x, crop_y, self.crop_size, self.crop_size)
            n_bands = min(3, src.count)
            data = src.read(list(range(1, n_bands + 1)), window=window)
            
            if data.shape[1] < self.crop_size or data.shape[2] < self.crop_size:
                padded = np.zeros((3, self.crop_size, self.crop_size), dtype=data.dtype)
                padded[:n_bands, :data.shape[1], :data.shape[2]] = data
                data = padded
            
            data = np.transpose(data, (1, 2, 0))
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
            if data.dtype != np.uint8:
                data = (data / max(data.max(), 1) * 255).astype(np.uint8)
        return data
    
    def _create_patch_labels(self, points_in_tile: np.ndarray) -> torch.Tensor:
        patch_labels = torch.zeros(self.grid_size, self.grid_size)
        for pt in points_in_tile:
            px, py = pt
            patch_x = max(0, min(int(px / self.patch_size), self.grid_size - 1))
            patch_y = max(0, min(int(py / self.patch_size), self.grid_size - 1))
            patch_labels[patch_y, patch_x] = 1.0
        return patch_labels
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        img_idx, crop_x, crop_y = self.tiles[idx]
        
        tile = self._read_tile(img_idx, crop_x, crop_y)
        points_in_tile = self._get_points_in_tile(img_idx, crop_x, crop_y)
        label = 1.0 if len(points_in_tile) > 0 else 0.0
        patch_labels = self._create_patch_labels(points_in_tile)
        
        tile = self.normalize_transform(image=tile)['image']
        
        return tile, {
            'label': torch.tensor(label, dtype=torch.float32),
            'patch_labels': patch_labels,
            'points': torch.from_numpy(points_in_tile).float(),
            'img_idx': img_idx,
            'crop_x': crop_x,
            'crop_y': crop_y,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    images = torch.stack([b[0] for b in batch])
    targets = {
        'label': torch.stack([b[1]['label'] for b in batch]),
        'patch_labels': torch.stack([b[1]['patch_labels'] for b in batch]),
        'points': [b[1]['points'] for b in batch],
        'img_idx': [b[1]['img_idx'] for b in batch],
        'crop_x': [b[1]['crop_x'] for b in batch],
        'crop_y': [b[1]['crop_y'] for b in batch],
    }
    return images, targets


# =============================================================================
# MODEL (same as iguana_classifier.py)
# =============================================================================

class IguanaClassifier(nn.Module):
    """Binary classifier using DINOv2 backbone."""
    
    def __init__(
        self,
        backbone: str = 'vit_large_patch14_reg4_dinov2.lvd142m',
        freeze_backbone: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)
        
        print(f"Backbone: {backbone}")
        print(f"  Feature dim: {self.feat_dim}, Patch size: {self.patch_size}")
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen")
        
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.patch_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        for head in [self.head, self.patch_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x, return_patches=False, upsample_patches=False):
        B, C, H, W = x.shape
        grid_size = H // self.patch_size
        
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.head(cls_token).squeeze(-1)
        
        if return_patches:
            patch_tokens = features[:, self.num_prefix_tokens:]
            patch_tokens = patch_tokens.view(B, grid_size, grid_size, self.feat_dim)
            patch_logits = self.patch_head(patch_tokens).squeeze(-1)
            
            if upsample_patches:
                patch_logits = F.interpolate(
                    patch_logits.unsqueeze(1), size=(H, W),
                    mode='bilinear', align_corners=False
                ).squeeze(1)
            
            return logits, patch_logits
        
        return logits
    
    def unfreeze_backbone(self, n_blocks=None):
        if n_blocks is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            if hasattr(self.backbone, 'blocks'):
                total = len(self.backbone.blocks)
                for i, block in enumerate(self.backbone.blocks):
                    if i >= total - n_blocks:
                        for p in block.parameters():
                            p.requires_grad = True


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, epoch=0, 
                patch_loss_weight=1.0, pos_weight=3.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tile_loss = 0
    total_patch_loss = 0
    total_correct = 0
    total_samples = 0
    
    tile_pos_weight = torch.tensor([pos_weight], device=device)
    
    pos_correct = 0
    pos_total = 0
    neg_correct = 0
    neg_total = 0
    
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        labels = targets['label'].to(device)
        patch_labels = targets['patch_labels'].to(device) if patch_loss_weight > 0 else None
        
        optimizer.zero_grad()
        
        if patch_labels is not None and patch_loss_weight > 0:
            tile_logits, patch_logits = model(images, return_patches=True)
            
            tile_loss = F.binary_cross_entropy_with_logits(
                tile_logits, labels,
                pos_weight=tile_pos_weight.expand_as(tile_logits)
            )
            
            n_pos_patches = patch_labels.sum()
            n_neg_patches = patch_labels.numel() - n_pos_patches
            patch_pos_weight = torch.clamp(n_neg_patches / (n_pos_patches + 1e-6), 1.0, 50.0).to(device)
            
            patch_loss = F.binary_cross_entropy_with_logits(
                patch_logits, patch_labels,
                pos_weight=patch_pos_weight.expand_as(patch_logits)
            )
            
            loss = tile_loss + patch_loss_weight * patch_loss
            total_tile_loss += tile_loss.item() * len(labels)
            total_patch_loss += patch_loss.item() * len(labels)
            logits = tile_logits
        else:
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels,
                pos_weight=tile_pos_weight.expand_as(logits)
            )
            total_tile_loss += loss.item() * len(labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)
        
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_correct += (preds[pos_mask] == labels[pos_mask]).sum().item()
        pos_total += pos_mask.sum().item()
        neg_correct += (preds[neg_mask] == labels[neg_mask]).sum().item()
        neg_total += neg_mask.sum().item()
    
    return {
        'loss': total_loss / total_samples,
        'tile_loss': total_tile_loss / total_samples,
        'patch_loss': total_patch_loss / total_samples if total_patch_loss > 0 else 0,
        'acc': total_correct / total_samples,
        'pos_acc': pos_correct / max(pos_total, 1),
        'neg_acc': neg_correct / max(neg_total, 1),
        'pos_total': pos_total,
        'neg_total': neg_total,
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.3):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, targets in loader:
        images = images.to(device)
        labels = targets['label'].to(device)
        
        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
        
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = (all_preds == all_labels).mean()
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    f3 = 10 * precision * recall / max(9 * precision + recall, 1e-6)
    
    return {
        'loss': total_loss / total_samples,
        'acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f3': f3,
        'all_probs': all_probs,
        'all_labels': all_labels,
    }


def find_optimal_threshold(all_probs, all_labels, beta=3.0):
    """Find threshold that maximizes F-beta."""
    best_thresh, best_fbeta, best_metrics = 0.5, 0, {}
    
    for thresh in np.arange(0.05, 0.95, 0.025):
        preds = (all_probs > thresh).astype(float)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        
        beta_sq = beta ** 2
        fbeta = (1 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-6)
        
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_thresh = thresh
            best_metrics = {'precision': precision, 'recall': recall, 'f_beta': fbeta}
    
    return best_thresh, best_fbeta, best_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pretrain Iguana Classifier on GeoTIFFs")
    
    # Data
    parser.add_argument('--geotiffs', nargs='+', required=True,
                        help='List of GeoTIFF files')
    parser.add_argument('--geojsons', nargs='+', required=True,
                        help='List of GeoJSON annotation files (matched by order)')
    parser.add_argument('--val_geotiffs', nargs='+', default=None,
                        help='Validation GeoTIFFs (optional)')
    parser.add_argument('--val_geojsons', nargs='+', default=None,
                        help='Validation GeoJSONs (optional)')
    
    # Model
    parser.add_argument('--backbone', default='vit_large_patch14_reg4_dinov2.lvd142m')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Dataset
    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--tiles_per_epoch', type=int, default=10000)
    parser.add_argument('--positive_ratio', type=float, default=0.5)
    parser.add_argument('--coverage_grid_size', type=int, default=256,
                        help='Size of coverage tracking grid cells')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pos_weight', type=float, default=3.0)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--patch_loss_weight', type=float, default=1.0)
    parser.add_argument('--unfreeze_epoch', type=int, default=10)
    parser.add_argument('--unfreeze_blocks', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', default='./pretrain_outputs')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--early_stopping', type=int, default=15)
    parser.add_argument('--min_epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.geotiffs) != len(args.geojsons):
        raise ValueError(f"Number of GeoTIFFs ({len(args.geotiffs)}) must match "
                        f"number of GeoJSONs ({len(args.geojsons)})")
    
    if args.val_geotiffs and args.val_geojsons:
        if len(args.val_geotiffs) != len(args.val_geojsons):
            raise ValueError("Number of validation GeoTIFFs must match GeoJSONs")
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training dataset
    train_ds = GeoTiffTileDataset(
        geotiff_paths=args.geotiffs,
        geojson_paths=args.geojsons,
        crop_size=args.crop_size,
        tiles_per_epoch=args.tiles_per_epoch,
        positive_ratio=args.positive_ratio,
        coverage_grid_size=args.coverage_grid_size,
        augment=True,
        seed=args.seed,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )
    
    # Validation dataset
    val_loader = None
    if args.val_geotiffs and args.val_geojsons:
        val_ds = GeoTiffValidationDataset(
            geotiff_paths=args.val_geotiffs,
            geojson_paths=args.val_geojsons,
            crop_size=args.crop_size,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn,
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
    print(f"\nParameters: {n_params:,}, Trainable: {n_train:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Resume
    start_epoch = 0
    best_f3 = 0
    epochs_without_improvement = 0
    backbone_unfrozen = False
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_f3 = ckpt.get('best_f3', 0)
            epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
            backbone_unfrozen = ckpt.get('backbone_unfrozen', False)
            
            if backbone_unfrozen:
                model.unfreeze_backbone(args.unfreeze_blocks)
                optimizer = torch.optim.AdamW([
                    {'params': [p for n, p in model.named_parameters()
                               if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.01},
                    {'params': [p for n, p in model.named_parameters()
                               if 'backbone' not in n and p.requires_grad], 'lr': args.lr * 0.1},
                ], weight_decay=args.weight_decay)
            
            print(f"  Resuming from epoch {start_epoch}, best F3: {best_f3:.4f}")
    
    # Training loop
    print("\n" + "=" * 70)
    print(f"PRETRAINING (F3 optimization, pos_weight={args.pos_weight})")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        # Unfreeze backbone
        if epoch == args.unfreeze_epoch and args.unfreeze_blocks > 0 and not backbone_unfrozen:
            print(f"\n*** Unfreezing last {args.unfreeze_blocks} backbone blocks ***")
            model.unfreeze_backbone(args.unfreeze_blocks)
            backbone_unfrozen = True
            
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
        train_m = train_epoch(model, train_loader, optimizer, device, epoch,
                              patch_loss_weight=args.patch_loss_weight,
                              pos_weight=args.pos_weight)
        scheduler.step()
        
        # Log training metrics
        log = f"Epoch {epoch:3d} ({time.time() - t0:.1f}s) | "
        log += f"loss={train_m['loss']:.4f} "
        if train_m['patch_loss'] > 0:
            log += f"(tile={train_m['tile_loss']:.3f} patch={train_m['patch_loss']:.3f}) "
        log += f"acc={train_m['acc']:.3f} "
        log += f"[pos={train_m['pos_acc']:.3f}({train_m['pos_total']}) neg={train_m['neg_acc']:.3f}({train_m['neg_total']})]"
        
        # Validation
        if val_loader:
            val_m = evaluate(model, val_loader, device, threshold=args.threshold)
            log += f" | P={val_m['precision']:.3f} R={val_m['recall']:.3f} "
            log += f"F1={val_m['f1']:.3f} F3={val_m['f3']:.3f}"
            
            if val_m['f3'] > best_f3:
                best_f3 = val_m['f3']
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f3': best_f3,
                    'backbone': args.backbone,
                    'hidden_dim': args.hidden_dim,
                    'dropout': args.dropout,
                    'backbone_unfrozen': backbone_unfrozen,
                }, output_dir / 'best.pth')
                log += " ★"
            else:
                epochs_without_improvement += 1
                if args.early_stopping > 0:
                    log += f" (no improvement: {epochs_without_improvement}/{args.early_stopping})"
        
        print(log)
        
        # Coverage stats every 5 epochs
        if epoch % 5 == 0:
            coverage = train_ds.get_coverage_stats()
            print("  Coverage stats:")
            for idx, stats in coverage.items():
                print(f"    [{idx}] {stats['coverage_pct']:.1f}% visited, "
                      f"visits: {stats['min_visits']}-{stats['max_visits']} (mean: {stats['mean_visits']:.1f})")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f3': best_f3,
            'backbone': args.backbone,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'backbone_unfrozen': backbone_unfrozen,
            'epochs_without_improvement': epochs_without_improvement,
        }, output_dir / 'latest.pth')
        
        # Early stopping
        if args.early_stopping > 0 and epoch >= args.min_epochs:
            if epochs_without_improvement >= args.early_stopping:
                print(f"\n*** Early stopping triggered ***")
                break
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best F3: {best_f3:.4f}")
    print("=" * 70)
    
    # Final coverage report
    print("\nFinal coverage report:")
    coverage = train_ds.get_coverage_stats()
    for idx, stats in coverage.items():
        tif_name = Path(train_ds.geotiffs[idx]['path']).name
        print(f"  {tif_name}:")
        print(f"    Coverage: {stats['coverage_pct']:.1f}%")
        print(f"    Visits per cell: {stats['min_visits']}-{stats['max_visits']} (mean: {stats['mean_visits']:.1f})")
    
    # Save final coverage stats
    with open(output_dir / 'coverage_stats.json', 'w') as f:
        json.dump({
            str(Path(train_ds.geotiffs[idx]['path']).name): stats
            for idx, stats in coverage.items()
        }, f, indent=2)


if __name__ == '__main__':
    main()
