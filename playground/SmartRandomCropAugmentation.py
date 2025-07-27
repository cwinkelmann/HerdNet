import pandas as pd
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


class PointTileAugmentation:
    """
    Tile augmentation system specifically designed for point annotations.
    Supports controlled empty crop generation, intelligent point-aware sampling, and zoom functionality.
    """

    def __init__(
            self,
            tile_size: int = 512,
            empty_prob: float = 0.2,  # Probability of returning empty crops
            zoom_limit: float = 0.3,  # Range for zoom factor (0.3 = 70% to 130%)
            rotation_limit: float = 45.0,
            scale_limit: float = 0.2,
            brightness_limit: float = 0.2,
            contrast_limit: float = 0.2,
            flip_horizontal: bool = True,
            flip_vertical: bool = True,
            blur_limit: Tuple[int, int] = (3, 7),
            noise_limit: float = 0.1,
            min_points_in_tile: int = 1,  # Minimum points required for non-empty tiles
            point_margin: int = 50,  # Margin around points for crop positioning
            zoom_prob: float = 0.5,  # Probability of applying zoom
    ):
        self.tile_size = tile_size
        self.empty_prob = empty_prob
        self.zoom_limit = zoom_limit
        self.zoom_prob = zoom_prob
        self.rotation_limit = rotation_limit
        self.scale_limit = scale_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.blur_limit = blur_limit
        self.noise_limit = noise_limit
        self.min_points_in_tile = min_points_in_tile
        self.point_margin = point_margin

        # Create albumentations transform for geometric augmentations
        self.geometric_transform = A.Compose([
            A.Rotate(limit=rotation_limit, p=0.8),
            A.RandomScale(scale_limit=scale_limit, p=0.5),
            A.HorizontalFlip(p=0.5 if flip_horizontal else 0),
            A.VerticalFlip(p=0.5 if flip_vertical else 0),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        # Color and noise augmentations
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.8
            ),
            A.GaussianBlur(blur_limit=blur_limit, p=0.3),
            A.GaussNoise(var_limit=(0, noise_limit * 255), p=0.2),
        ])

    def calculate_zoom_crop_size(self, zoom_factor: float) -> int:
        """
        Calculate the initial crop size needed for zoom effect.

        Args:
            zoom_factor: Zoom factor (>1 = zoom in, <1 = zoom out)

        Returns:
            Initial crop size needed
        """
        # Add safety margin for rotations and other transforms
        safety_factor = 1.4  # 40% extra for rotation buffer
        base_size = int(self.tile_size * safety_factor)

        # Adjust crop size based on zoom factor
        # zoom_factor > 1: smaller initial crop (zoom in effect)
        # zoom_factor < 1: larger initial crop (zoom out effect)
        zoom_crop_size = int(base_size / zoom_factor)

        return zoom_crop_size

    def get_point_aware_crop_region(
            self,
            image_shape: Tuple[int, int],
            points: List[Tuple[float, float]],
            crop_size: int,
            force_empty: bool = False
    ) -> Tuple[int, int, int, int]:
        """
        Get crop region that intelligently considers point locations.

        Args:
            image_shape: (height, width) of the image
            points: List of (x, y) point coordinates
            crop_size: Size of the crop region
            force_empty: If True, deliberately avoid points

        Returns:
            (x1, y1, x2, y2) crop coordinates
        """
        h, w = image_shape[:2]

        if force_empty or len(points) == 0:
            # Generate random crop that avoids points (if any exist)
            return self._get_empty_crop_region(image_shape, points, crop_size)

        # Choose a random point to center the crop around
        target_point = random.choice(points)
        x_center, y_center = target_point

        # Add some randomness around the point
        offset_range = crop_size * 0.3
        x_center += random.uniform(-offset_range, offset_range)
        y_center += random.uniform(-offset_range, offset_range)

        # Calculate crop boundaries
        half_crop = crop_size // 2
        x1 = int(x_center - half_crop)
        y1 = int(y_center - half_crop)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Ensure crop is within image bounds
        if x1 < 0:
            x2 -= x1
            x1 = 0
        elif x2 > w:
            x1 -= (x2 - w)
            x2 = w

        if y1 < 0:
            y2 -= y1
            y1 = 0
        elif y2 > h:
            y1 -= (y2 - h)
            y2 = h

        # Final bounds check
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)

        return x1, y1, x2, y2

    def _get_empty_crop_region(
            self,
            image_shape: Tuple[int, int],
            points: List[Tuple[float, float]],
            crop_size: int
    ) -> Tuple[int, int, int, int]:
        """Generate a crop region that avoids existing points."""
        h, w = image_shape[:2]

        # Try multiple random positions and pick one with minimal point overlap
        best_crop = None
        min_points = float('inf')

        for _ in range(20):  # Try 20 random positions
            max_y = max(0, h - crop_size)
            max_x = max(0, w - crop_size)

            x1 = random.randint(0, max_x) if max_x > 0 else 0
            y1 = random.randint(0, max_y) if max_y > 0 else 0
            x2 = x1 + crop_size
            y2 = y1 + crop_size

            # Count points in this region
            points_in_crop = self._count_points_in_region(points, (x1, y1, x2, y2))

            if points_in_crop < min_points:
                min_points = points_in_crop
                best_crop = (x1, y1, x2, y2)

                # If we found a truly empty region, use it
                if points_in_crop == 0:
                    break

        return best_crop if best_crop else (0, 0, crop_size, crop_size)

    def _count_points_in_region(
            self,
            points: List[Tuple[float, float]],
            region: Tuple[int, int, int, int]
    ) -> int:
        """Count how many points fall within a given region."""
        x1, y1, x2, y2 = region
        count = 0

        for x, y in points:
            if x1 <= x <= x2 and y1 <= y <= y2:
                count += 1

        return count

    def apply_zoom_effect(
            self,
            cropped_region: np.ndarray,
            zoom_factor: float
    ) -> np.ndarray:
        """
        Apply zoom effect by scaling the cropped region to tile size.

        Args:
            cropped_region: Initial large crop from image
            zoom_factor: Zoom factor applied

        Returns:
            Scaled region at tile size
        """
        # Resize to final tile size
        scaled_region = cv2.resize(cropped_region, (self.tile_size, self.tile_size),
                                   interpolation=cv2.INTER_LINEAR)

        return scaled_region

    def augment_tile(
            self,
            image: np.ndarray,
            points: List[Tuple[float, float, int]],  # (x, y, label)
            force_empty: bool = None,
            return_debug_info: bool = False
    ) -> Tuple[np.ndarray, List[Tuple[float, float, int]], bool, Optional[Dict]]:
        """
        Apply tile augmentation to image and points with zoom functionality.

        Args:
            image: Input image
            points: List of (x, y, label) tuples
            force_empty: Override empty probability decision
            return_debug_info: If True, return debug information

        Returns:
            (augmented_tile, transformed_points, is_empty, debug_info)
        """
        debug_info = {} if return_debug_info else None

        # Decide whether to generate empty crop
        if force_empty is None:
            force_empty = random.random() < self.empty_prob

        # Decide whether to apply zoom
        apply_zoom = random.random() < self.zoom_prob
        zoom_factor = 1.0

        if apply_zoom:
            zoom_factor = random.uniform(1 - self.zoom_limit, 1 + self.zoom_limit)

        if return_debug_info:
            debug_info.update({
                'force_empty': force_empty,
                'apply_zoom': apply_zoom,
                'zoom_factor': zoom_factor
            })

        # Calculate initial crop size based on zoom
        initial_crop_size = self.calculate_zoom_crop_size(zoom_factor)

        # Get crop region
        crop_coords = self.get_point_aware_crop_region(
            image.shape,
            [(p[0], p[1]) for p in points],
            initial_crop_size,
            force_empty=force_empty
        )

        x1, y1, x2, y2 = crop_coords

        if return_debug_info:
            debug_info['crop_coords'] = crop_coords
            debug_info['initial_crop_size'] = initial_crop_size

        # Extract initial crop
        initial_crop = image[y1:y2, x1:x2]

        # Apply zoom effect (scale to tile size)
        tile = self.apply_zoom_effect(initial_crop, zoom_factor)

        # Filter points that are in the crop region and scale them to tile coordinates
        points_in_crop = []
        for x, y, label in points:
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Adjust coordinates to tile space
                adj_x = (x - x1) * (self.tile_size / (x2 - x1))
                adj_y = (y - y1) * (self.tile_size / (y2 - y1))
                points_in_crop.append((adj_x, adj_y, label))

        # Apply geometric augmentations
        keypoints_for_transform = [(p[0], p[1]) for p in points_in_crop]

        if len(keypoints_for_transform) > 0:
            transform_result = self.geometric_transform(
                image=tile,
                keypoints=keypoints_for_transform
            )
            tile = transform_result['image']

            # Update points with transformed coordinates
            transformed_points = []
            for i, (x, y) in enumerate(transform_result['keypoints']):
                if i < len(points_in_crop):
                    label = points_in_crop[i][2]
                    # Filter out points that went outside the tile
                    if 0 <= x < self.tile_size and 0 <= y < self.tile_size:
                        transformed_points.append((x, y, label))
        else:
            # No points to transform
            transformed_points = []
            transform_result = self.geometric_transform(image=tile, keypoints=[])
            tile = transform_result['image']

        # Apply color augmentations
        tile = self.color_transform(image=tile)['image']

        # Determine if tile is empty
        is_empty = len(transformed_points) < self.min_points_in_tile

        if return_debug_info:
            debug_info.update({
                'points_in_initial_crop': len(points_in_crop),
                'points_after_transform': len(transformed_points),
                'is_empty': is_empty
            })

        if return_debug_info:
            return tile, transformed_points, is_empty, debug_info
        else:
            return tile, transformed_points, is_empty, None


def create_sample_image_and_points():
    """Create a sample image with some point annotations for testing."""
    # Create a sample image (simulating a nature/animal scene)
    image = np.random.randint(50, 200, (2000, 3000, 3), dtype=np.uint8)

    # Add some colored regions to make it more interesting
    cv2.circle(image, (500, 400), 100, (100, 150, 100), -1)  # Green blob
    cv2.circle(image, (1500, 800), 80, (150, 100, 100), -1)  # Blue blob
    cv2.circle(image, (2200, 1200), 120, (100, 100, 150), -1)  # Red blob
    cv2.rectangle(image, (800, 600), (1200, 1000), (120, 120, 80), -1)  # Rectangle

    # Add some texture
    noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Sample points (simulating animal detections)
    sample_points = [
        (520, 420, 3),  # Point in green blob
        (480, 390, 3),  # Another point in green blob
        (1520, 820, 2),  # Point in blue blob
        (2220, 1220, 2),  # Point in red blob
        (1000, 800, 1),  # Point in rectangle
        (1100, 900, 1),  # Another point in rectangle
        (300, 300, 3),  # Random point
        (2500, 1500, 2),  # Edge point
    ]

    return image, sample_points


def visualize_augmentation_results(
        original_image: np.ndarray,
        original_points: List[Tuple[float, float, int]],
        augmenter: PointTileAugmentation,
        num_examples: int = 6
):
    """Visualize before/after augmentation results with proper comparison."""

    # Color map for different labels
    color_map = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    fig, axes = plt.subplots(2, num_examples, figsize=(24, 8))
    fig.suptitle('Point Annotation Tile Augmentation Results (Top: Original Crop, Bottom: Augmented)', fontsize=16)

    for i in range(num_examples):
        # Force different behaviors for demonstration
        if i < 2:
            force_empty = False  # Ensure we get non-empty tiles
        elif i == num_examples - 1:
            force_empty = True  # Force one empty tile
        else:
            force_empty = None  # Let probability decide

        # Apply augmentation with debug info to get the exact crop coordinates
        tile, transformed_points, is_empty, debug_info = augmenter.augment_tile(
            original_image, original_points,
            force_empty=force_empty,
            return_debug_info=True
        )

        # Get the exact same crop region that was used for augmentation
        crop_coords = debug_info['crop_coords']
        zoom_factor = debug_info['zoom_factor']
        x1, y1, x2, y2 = crop_coords

        # Original image subplot (top row) - show EXACT same crop that was augmented
        ax_orig = axes[0, i]
        orig_crop = original_image[y1:y2, x1:x2]

        # Apply the same zoom scaling to show what the original looked like
        orig_crop_scaled = cv2.resize(orig_crop, (augmenter.tile_size, augmenter.tile_size))
        ax_orig.imshow(cv2.cvtColor(orig_crop_scaled, cv2.COLOR_BGR2RGB))

        # Draw original points in the crop with proper scaling
        for x, y, label in original_points:
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Scale coordinates to tile space (same as in augmentation)
                adj_x = (x - x1) * (augmenter.tile_size / (x2 - x1))
                adj_y = (y - y1) * (augmenter.tile_size / (y2 - y1))
                color = [c / 255.0 for c in color_map.get(label, (255, 255, 255))]
                ax_orig.scatter(adj_x, adj_y, c=[color], s=100, marker='o', edgecolors='white', linewidth=2)

        zoom_text = f"zoom {zoom_factor:.2f}" if debug_info['apply_zoom'] else "no zoom"
        ax_orig.set_title(f'Original {i + 1} ({zoom_text})')
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        # Augmented image subplot (bottom row)
        ax_aug = axes[1, i]
        ax_aug.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))

        # Draw transformed points
        for x, y, label in transformed_points:
            color = [c / 255.0 for c in color_map.get(label, (255, 255, 255))]
            ax_aug.scatter(x, y, c=[color], s=100, marker='o', edgecolors='white', linewidth=2)

        status = "EMPTY" if is_empty else f"{len(transformed_points)} points"
        ax_aug.set_title(f'Augmented {i + 1} ({status})')
        ax_aug.set_xticks([])
        ax_aug.set_yticks([])

    plt.tight_layout()
    return fig


def test_zoom_functionality():
    """Test the zoom functionality specifically."""
    print("Testing Zoom Functionality")
    print("=" * 40)

    # Create test data
    image, points = create_sample_image_and_points()

    # Test different zoom settings
    zoom_limits = [0.0, 0.2, 0.4, 0.6]

    for zoom_limit in zoom_limits:
        augmenter = PointTileAugmentation(
            zoom_limit=zoom_limit,
            zoom_prob=1.0,  # Always apply zoom for testing
            empty_prob=0.0,  # No empty tiles for this test
            tile_size=512
        )

        print(f"\nZoom limit: ±{zoom_limit:.1f}")
        print(f"Range: {1 - zoom_limit:.1f}x to {1 + zoom_limit:.1f}x")

        # Generate 10 tiles and show zoom factors
        zoom_factors = []
        for _ in range(10):
            _, _, _, debug_info = augmenter.augment_tile(
                image, points, return_debug_info=True
            )
            zoom_factors.append(debug_info['zoom_factor'])

        print(f"Sample zoom factors: {[f'{z:.2f}' for z in zoom_factors[:5]]}")
        print(f"Crop sizes: {[augmenter.calculate_zoom_crop_size(z) for z in zoom_factors[:3]]}")


def demonstrate_zoom_effects():
    """Demonstrate different zoom effects visually."""
    print("\nDemonstrating Zoom Effects")
    print("=" * 30)

    # Create test data
    image, points = create_sample_image_and_points()

    # Create augmenter with high zoom probability
    augmenter = PointTileAugmentation(
        zoom_limit=0.4,  # Large zoom range
        zoom_prob=1.0,  # Always zoom
        empty_prob=0.0,  # No empty tiles
        rotation_limit=0,  # Disable rotation for clearer zoom demonstration
        flip_horizontal=False,
        flip_vertical=False,
        tile_size=512
    )

    # Show zoom effects
    fig = visualize_augmentation_results(image, points, augmenter, num_examples=6)

    return fig


def test_empty_probability_control():
    """Test the empty probability control mechanism."""
    print("Testing Empty Probability Control")
    print("=" * 40)

    # Create test data
    image, points = create_sample_image_and_points()

    # Test different empty probabilities
    empty_probs = [0.0, 0.2, 0.5, 0.8, 1.0]

    for empty_prob in empty_probs:
        augmenter = PointTileAugmentation(empty_prob=empty_prob, tile_size=512)

        # Generate 100 tiles and count empty ones
        empty_count = 0
        total_tiles = 100

        for _ in range(total_tiles):
            _, transformed_points, is_empty, _ = augmenter.augment_tile(image, points)
            if is_empty:
                empty_count += 1

        actual_empty_rate = empty_count / total_tiles
        print(f"empty_prob={empty_prob:.1f}: {empty_count:2d}/100 empty tiles (actual rate: {actual_empty_rate:.2f})")


def demonstrate_augmentation_pipeline():
    """Demonstrate the complete augmentation pipeline."""
    print("\nDemonstrating Complete Augmentation Pipeline")
    print("=" * 45)

    # Create sample data
    image, points = create_sample_image_and_points()

    print(f"Original image shape: {image.shape}")
    print(f"Number of points: {len(points)}")
    print(f"Point coordinates: {points}")

    # Create augmenter with comprehensive settings
    augmenter = PointTileAugmentation(
        tile_size=512,
        empty_prob=0.3,  # 30% chance of empty tiles
        zoom_limit=0.3,  # ±30% zoom range
        zoom_prob=0.7,  # 70% chance of zoom
        rotation_limit=45.0,
        scale_limit=0.2,
        brightness_limit=0.2,
        contrast_limit=0.2,
        min_points_in_tile=1
    )

    # Test the augmentation pipeline
    results = []
    for i in range(10):
        tile, transformed_points, is_empty, debug_info = augmenter.augment_tile(
            image, points, return_debug_info=True
        )
        results.append({
            'tile_id': i,
            'num_points': len(transformed_points),
            'is_empty': is_empty,
            'zoom_applied': debug_info['apply_zoom'],
            'zoom_factor': debug_info['zoom_factor'],
            'tile_shape': tile.shape
        })

        if i < 3:  # Show details for first 3 tiles
            print(f"\nTile {i + 1}:")
            print(f"  Points in tile: {len(transformed_points)}")
            print(f"  Is empty: {is_empty}")
            print(f"  Zoom applied: {debug_info['apply_zoom']}")
            if debug_info['apply_zoom']:
                print(f"  Zoom factor: {debug_info['zoom_factor']:.2f}")
            print(f"  Point coordinates: {[(round(x, 1), round(y, 1), l) for x, y, l in transformed_points]}")

    # Summary statistics
    empty_tiles = sum(1 for r in results if r['is_empty'])
    zoom_tiles = sum(1 for r in results if r['zoom_applied'])

    print(f"\nSummary (10 tiles):")
    print(f"  Empty tiles: {empty_tiles}/10 ({empty_tiles * 10}%)")
    print(f"  Non-empty tiles: {10 - empty_tiles}/10")
    print(f"  Zoom applied: {zoom_tiles}/10 ({zoom_tiles * 10}%)")

    # Create visualization
    fig = visualize_augmentation_results(image, points, augmenter, num_examples=6)

    return image, points, augmenter, fig


if __name__ == "__main__":
    # Run tests and demonstrations
    test_empty_probability_control()
    test_zoom_functionality()

    # Demonstrate the complete pipeline
    image, points, augmenter, fig = demonstrate_augmentation_pipeline()

    # Show zoom effects specifically
    zoom_fig = demonstrate_zoom_effects()

    print("\n" + "=" * 60)
    print("ENHANCED FEATURES:")
    print("=" * 60)
    print("""
✅ ZOOM FUNCTIONALITY:
- zoom_limit: Controls zoom range (e.g., 0.3 = 70% to 130%)
- zoom_prob: Probability of applying zoom (0.0 to 1.0)
- Simulates different viewing distances/scales
- zoom < 1.0: zoom out (see more context)
- zoom > 1.0: zoom in (see more detail)

✅ PROPER VISUALIZATION:
- Top row shows EXACT original crop that gets augmented
- Bottom row shows the augmented result
- True before/after comparison
- Shows zoom factor in titles

✅ ENHANCED CONTROL:
- return_debug_info=True provides detailed transformation info
- Force specific behaviors for testing
- Comprehensive statistics and monitoring

USAGE EXAMPLE:
augmenter = PointTileAugmentation(
    tile_size=512,
    empty_prob=0.2,      # 20% empty tiles
    zoom_limit=0.3,      # ±30% zoom range  
    zoom_prob=0.5,       # 50% chance of zoom
    rotation_limit=45.0
)

tile, points, is_empty, debug = augmenter.augment_tile(
    image, points, return_debug_info=True
)
    """)

    plt.show()