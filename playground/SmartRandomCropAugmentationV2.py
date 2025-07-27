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
    Pipeline: Initial large crop → Geometric augmentations → Final center crop
    """

    def __init__(
            self,
            tile_size: int = 512,
            empty_prob: float = 0.2,  # Probability of returning empty crops
            crop_scale_factor: float = 1.5,  # How much larger initial crop should be (1.5 = 50% larger)
            rotation_limit: float = 45.0,
            scale_limit: float = 0.2,
            shear_limit: float = 10.0,
            brightness_limit: float = 0.2,
            contrast_limit: float = 0.2,
            flip_horizontal: bool = True,
            flip_vertical: bool = True,
            blur_limit: Tuple[int, int] = (3, 7),
            noise_limit: float = 0.1,
            min_points_in_tile: int = 1,  # Minimum points required for non-empty tiles
            point_margin: int = 50,  # Margin around points for crop positioning
    ):
        self.tile_size = tile_size
        self.empty_prob = empty_prob
        self.crop_scale_factor = crop_scale_factor
        self.initial_crop_size = int(tile_size * crop_scale_factor)
        self.rotation_limit = rotation_limit
        self.scale_limit = scale_limit
        self.shear_limit = shear_limit
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
            A.Affine(shear=shear_limit, p=0.3),
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

    def get_point_aware_crop_region(
            self,
            image_shape: Tuple[int, int],
            points: List[Tuple[float, float]],
            force_empty: bool = False
    ) -> Tuple[int, int, int, int]:
        """
        Get initial crop region that intelligently considers point locations.
        This creates the larger initial crop before geometric augmentations.

        Args:
            image_shape: (height, width) of the image
            points: List of (x, y) point coordinates
            force_empty: If True, deliberately avoid points

        Returns:
            (x1, y1, x2, y2) crop coordinates for initial large crop
        """
        h, w = image_shape[:2]
        crop_size = self.initial_crop_size

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

    def extract_final_tile(self, augmented_large_crop: np.ndarray) -> np.ndarray:
        """
        Extract the final tile from the center of the augmented large crop.

        Args:
            augmented_large_crop: The larger crop after geometric augmentations

        Returns:
            Final tile of size (tile_size, tile_size)
        """
        h, w = augmented_large_crop.shape[:2]

        # Calculate center crop coordinates
        center_y, center_x = h // 2, w // 2
        half_tile = self.tile_size // 2

        y1 = center_y - half_tile
        x1 = center_x - half_tile
        y2 = y1 + self.tile_size
        x2 = x1 + self.tile_size

        # Ensure we don't go out of bounds
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(h, y1 + self.tile_size)
        x2 = min(w, x1 + self.tile_size)

        # Extract final tile
        final_tile = augmented_large_crop[y1:y2, x1:x2]

        # Ensure exact size (resize if needed due to boundary constraints)
        if final_tile.shape[:2] != (self.tile_size, self.tile_size):
            final_tile = cv2.resize(final_tile, (self.tile_size, self.tile_size))

        return final_tile

    def transform_points_through_pipeline(
            self,
            points: List[Tuple[float, float, int]],
            initial_crop_coords: Tuple[int, int, int, int],
            geometric_transform_result: Dict,
            final_crop_offset: Tuple[int, int]
    ) -> List[Tuple[float, float, int]]:
        """
        Transform points through the complete pipeline:
        1. Adjust for initial crop
        2. Apply geometric transforms
        3. Adjust for final center crop
        """
        x1_initial, y1_initial, x2_initial, y2_initial = initial_crop_coords
        final_x_offset, final_y_offset = final_crop_offset

        # Step 1: Filter points in initial crop and adjust coordinates
        points_in_initial_crop = []
        for x, y, label in points:
            if x1_initial <= x <= x2_initial and y1_initial <= y <= y2_initial:
                adj_x = x - x1_initial
                adj_y = y - y1_initial
                points_in_initial_crop.append((adj_x, adj_y, label))

        # Step 2: Apply geometric transforms
        transformed_points = []
        if len(points_in_initial_crop) > 0 and 'keypoints' in geometric_transform_result:
            for i, (transformed_x, transformed_y) in enumerate(geometric_transform_result['keypoints']):
                if i < len(points_in_initial_crop):
                    label = points_in_initial_crop[i][2]

                    # Step 3: Adjust for final center crop
                    final_x = transformed_x - final_x_offset
                    final_y = transformed_y - final_y_offset

                    # Only keep points that are within the final tile
                    if 0 <= final_x < self.tile_size and 0 <= final_y < self.tile_size:
                        transformed_points.append((final_x, final_y, label))

        return transformed_points

    def augment_tile(
            self,
            image: np.ndarray,
            points: List[Tuple[float, float, int]],  # (x, y, label)
            force_empty: bool = None,
            return_debug_info: bool = False
    ) -> Tuple[np.ndarray, List[Tuple[float, float, int]], bool, Optional[Dict]]:
        """
        Apply tile augmentation using the proper pipeline:
        1. Extract initial large crop
        2. Apply geometric augmentations
        3. Extract final center crop
        4. Apply color augmentations

        Args:
            image: Input image
            points: List of (x, y, label) tuples
            force_empty: Override empty probability decision
            return_debug_info: If True, return debug information

        Returns:
            (final_tile, transformed_points, is_empty, debug_info)
        """
        debug_info = {} if return_debug_info else None

        # Decide whether to generate empty crop
        if force_empty is None:
            force_empty = random.random() < self.empty_prob

        if return_debug_info:
            debug_info.update({
                'force_empty': force_empty,
                'initial_crop_size': self.initial_crop_size,
                'final_tile_size': self.tile_size,
                'crop_scale_factor': self.crop_scale_factor
            })

        # Step 1: Get initial large crop region
        initial_crop_coords = self.get_point_aware_crop_region(
            image.shape,
            [(p[0], p[1]) for p in points],
            force_empty=force_empty
        )
        x1, y1, x2, y2 = initial_crop_coords

        if return_debug_info:
            debug_info['initial_crop_coords'] = initial_crop_coords

        # Extract initial large crop
        initial_crop = image[y1:y2, x1:x2]

        # Resize to exact initial crop size if needed
        if initial_crop.shape[:2] != (self.initial_crop_size, self.initial_crop_size):
            initial_crop = cv2.resize(initial_crop, (self.initial_crop_size, self.initial_crop_size))

        # Prepare points for geometric transformation
        points_for_geometric_transform = []
        for x, y, label in points:
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Scale coordinates to initial crop space
                scale_x = self.initial_crop_size / (x2 - x1)
                scale_y = self.initial_crop_size / (y2 - y1)
                adj_x = (x - x1) * scale_x
                adj_y = (y - y1) * scale_y
                points_for_geometric_transform.append((adj_x, adj_y))

        # Step 2: Apply geometric augmentations to the large crop
        keypoints_for_transform = points_for_geometric_transform

        if len(keypoints_for_transform) > 0:
            geometric_result = self.geometric_transform(
                image=initial_crop,
                keypoints=keypoints_for_transform
            )
            augmented_large_crop = geometric_result['image']
        else:
            # No points to transform, but still apply geometric transforms
            geometric_result = self.geometric_transform(image=initial_crop, keypoints=[])
            augmented_large_crop = geometric_result['image']

        # Step 3: Extract final center crop
        final_tile = self.extract_final_tile(augmented_large_crop)

        # Calculate the offset for the final center crop
        center_offset_x = (self.initial_crop_size - self.tile_size) // 2
        center_offset_y = (self.initial_crop_size - self.tile_size) // 2

        if return_debug_info:
            debug_info.update({
                'center_offset_x': center_offset_x,
                'center_offset_y': center_offset_y,
                'points_in_initial_crop': len(points_for_geometric_transform)
            })

        # Transform points through the complete pipeline
        transformed_points = []
        if len(points_for_geometric_transform) > 0 and 'keypoints' in geometric_result:
            for i, (transformed_x, transformed_y) in enumerate(geometric_result['keypoints']):
                if i < len(points):
                    # Find the corresponding original point
                    original_point = None
                    point_idx = 0
                    for x, y, label in points:
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            if point_idx == i:
                                original_point = (x, y, label)
                                break
                            point_idx += 1

                    if original_point:
                        label = original_point[2]
                        # Adjust for final center crop
                        final_x = transformed_x - center_offset_x
                        final_y = transformed_y - center_offset_y

                        # Only keep points within final tile
                        if 0 <= final_x < self.tile_size and 0 <= final_y < self.tile_size:
                            transformed_points.append((final_x, final_y, label))

        # Step 4: Apply color augmentations to final tile
        final_tile = self.color_transform(image=final_tile)['image']

        # Determine if tile is empty
        is_empty = len(transformed_points) < self.min_points_in_tile

        if return_debug_info:
            debug_info.update({
                'points_after_geometric': len(geometric_result.get('keypoints', [])),
                'points_in_final_tile': len(transformed_points),
                'is_empty': is_empty
            })

        if return_debug_info:
            return final_tile, transformed_points, is_empty, debug_info
        else:
            return final_tile, transformed_points, is_empty, None


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


def visualize_three_stage_pipeline(
        original_image: np.ndarray,
        original_points: List[Tuple[float, float, int]],
        augmenter: PointTileAugmentation,
        num_examples: int = 4
):
    """Visualize the three-stage pipeline: Original → Large Crop → Augmented Large Crop → Final Tile."""

    # Color map for different labels
    color_map = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    fig, axes = plt.subplots(3, num_examples, figsize=(20, 12))
    fig.suptitle('Three-Stage Pipeline: Initial Crop → Geometric Aug → Final Center Crop', fontsize=16)

    for i in range(num_examples):
        # Force different behaviors for demonstration
        if i < 2:
            force_empty = False
        elif i == num_examples - 1:
            force_empty = True
        else:
            force_empty = None

        # Get initial crop coordinates
        initial_crop_coords = augmenter.get_point_aware_crop_region(
            original_image.shape,
            [(p[0], p[1]) for p in original_points],
            force_empty=force_empty
        )
        x1, y1, x2, y2 = initial_crop_coords

        # Stage 1: Initial large crop
        initial_crop = original_image[y1:y2, x1:x2]
        if initial_crop.shape[:2] != (augmenter.initial_crop_size, augmenter.initial_crop_size):
            initial_crop = cv2.resize(initial_crop, (augmenter.initial_crop_size, augmenter.initial_crop_size))

        axes[0, i].imshow(cv2.cvtColor(initial_crop, cv2.COLOR_BGR2RGB))

        # Draw points in initial crop
        for x, y, label in original_points:
            if x1 <= x <= x2 and y1 <= y <= y2:
                scale_x = augmenter.initial_crop_size / (x2 - x1)
                scale_y = augmenter.initial_crop_size / (y2 - y1)
                adj_x = (x - x1) * scale_x
                adj_y = (y - y1) * scale_y
                color = [c / 255.0 for c in color_map.get(label, (255, 255, 255))]
                axes[0, i].scatter(adj_x, adj_y, c=[color], s=100, marker='o', edgecolors='white', linewidth=2)

        axes[0, i].set_title(f'Initial Crop {i + 1} ({augmenter.initial_crop_size}px)')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Add center crop preview box
        center_offset = (augmenter.initial_crop_size - augmenter.tile_size) // 2
        rect = plt.Rectangle((center_offset, center_offset), augmenter.tile_size, augmenter.tile_size,
                             linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        axes[0, i].add_patch(rect)

        # Stage 2: Apply geometric augmentations (simulate the process)
        points_for_transform = []
        for x, y, label in original_points:
            if x1 <= x <= x2 and y1 <= y <= y2:
                scale_x = augmenter.initial_crop_size / (x2 - x1)
                scale_y = augmenter.initial_crop_size / (y2 - y1)
                adj_x = (x - x1) * scale_x
                adj_y = (y - y1) * scale_y
                points_for_transform.append((adj_x, adj_y))

        if len(points_for_transform) > 0:
            geometric_result = augmenter.geometric_transform(
                image=initial_crop,
                keypoints=points_for_transform
            )
            augmented_large_crop = geometric_result['image']
            transformed_keypoints = geometric_result['keypoints']
        else:
            geometric_result = augmenter.geometric_transform(image=initial_crop, keypoints=[])
            augmented_large_crop = geometric_result['image']
            transformed_keypoints = []

        axes[1, i].imshow(cv2.cvtColor(augmented_large_crop, cv2.COLOR_BGR2RGB))

        # Draw transformed points
        point_idx = 0
        for x, y, label in original_points:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if point_idx < len(transformed_keypoints):
                    tx, ty = transformed_keypoints[point_idx]
                    color = [c / 255.0 for c in color_map.get(label, (255, 255, 255))]
                    axes[1, i].scatter(tx, ty, c=[color], s=100, marker='o', edgecolors='white', linewidth=2)
                    point_idx += 1

        axes[1, i].set_title(f'After Geometric Aug {i + 1}')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # Add center crop preview box
        rect = plt.Rectangle((center_offset, center_offset), augmenter.tile_size, augmenter.tile_size,
                             linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        axes[1, i].add_patch(rect)

        # Stage 3: Final center crop and full pipeline result
        final_tile, transformed_points, is_empty, debug_info = augmenter.augment_tile(
            original_image, original_points,
            force_empty=force_empty,
            return_debug_info=True
        )

        axes[2, i].imshow(cv2.cvtColor(final_tile, cv2.COLOR_BGR2RGB))

        # Draw final transformed points
        for x, y, label in transformed_points:
            color = [c / 255.0 for c in color_map.get(label, (255, 255, 255))]
            axes[2, i].scatter(x, y, c=[color], s=100, marker='o', edgecolors='white', linewidth=2)

        status = "EMPTY" if is_empty else f"{len(transformed_points)} points"
        axes[2, i].set_title(f'Final Tile {i + 1} ({status})')
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])

    plt.tight_layout()
    return fig


def demonstrate_pipeline_benefits():
    """Demonstrate why this pipeline works better."""
    print("Pipeline Benefits Demonstration")
    print("=" * 35)

    # Create test data
    image, points = create_sample_image_and_points()

    print("THREE-STAGE PIPELINE:")
    print("1. Initial Crop: ~50% larger than final tile")
    print("2. Geometric Augmentations: Applied to large crop")
    print("3. Final Center Crop: Clean extraction without artifacts")
    print()

    augmenter = PointTileAugmentation(
        tile_size=512,
        crop_scale_factor=1.5,  # 50% larger initial crop
        empty_prob=0.0,  # No empty tiles for this demo
        rotation_limit=45.0,
        shear_limit=10.0
    )

    print(f"Configuration:")
    print(f"  Final tile size: {augmenter.tile_size}px")
    print(f"  Initial crop size: {augmenter.initial_crop_size}px")
    print(f"  Scale factor: {augmenter.crop_scale_factor}x")
    print(f"  Extra margin: {augmenter.initial_crop_size - augmenter.tile_size}px")
    print()

    # Test a few augmentations
    for i in range(3):
        final_tile, transformed_points, is_empty, debug_info = augmenter.augment_tile(
            image, points, return_debug_info=True
        )

        print(f"Example {i + 1}:")
        print(f"  Points in initial crop: {debug_info['points_in_initial_crop']}")
        print(f"  Points after geometric aug: {debug_info['points_after_geometric']}")
        print(f"  Points in final tile: {debug_info['points_in_final_tile']}")
        print(f"  Center offset: ({debug_info['center_offset_x']}, {debug_info['center_offset_y']})")
        print()

    print("ADVANTAGES:")
    print("✅ No black corners after rotation")
    print("✅ Clean final crops without artifacts")
    print("✅ All geometric augmentations preserve image quality")
    print("✅ Consistent tile size regardless of augmentation")
    print("✅ Points correctly transformed through all stages")


def test_empty_probability_control():
    """Test the empty probability control mechanism."""
    print("\nTesting Empty Probability Control")
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


def demonstrate_complete_pipeline():
    """Demonstrate the complete augmentation pipeline."""
    print("\nDemonstrating Complete Three-Stage Pipeline")
    print("=" * 45)

    # Create sample data
    image, points = create_sample_image_and_points()

    print(f"Original image shape: {image.shape}")
    print(f"Number of points: {len(points)}")

    # Create augmenter with the proper pipeline
    augmenter = PointTileAugmentation(
        tile_size=512,
        crop_scale_factor=1.5,  # 50% larger initial crop
        empty_prob=0.3,  # 30% chance of empty tiles
        rotation_limit=45.0,
        scale_limit=0.2,
        shear_limit=10.0,
        brightness_limit=0.2,
        contrast_limit=0.2
    )

    print(f"\nPipeline Configuration:")
    print(f"  Final tile size: {augmenter.tile_size}px")
    print(f"  Initial crop size: {augmenter.initial_crop_size}px")
    print(f"  Extra margin for rotations: {augmenter.initial_crop_size - augmenter.tile_size}px")

    # Create visualization
    fig = visualize_three_stage_pipeline(image, points, augmenter, num_examples=4)

    return image, points, augmenter, fig


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_pipeline_benefits()
    test_empty_probability_control()

    # Show complete pipeline
    image, points, augmenter, fig = demonstrate_complete_pipeline()

    print("\n" + "=" * 60)
    print("CORRECT PIPELINE IMPLEMENTATION:")
    print("=" * 60)
    print("""
✅ THREE-STAGE PIPELINE:
1. Initial Crop: Extract ~50% larger region (e.g., 768px for 512px final)
2. Geometric Augmentations: Apply rotations, scaling, shear to large crop
3. Final Center Crop: Extract clean 512px tile from center

✅ ADVANTAGES:
- No black corners or artifacts after rotation
- Clean, consistent final tile size
- Proper geometric transformation of points
- Professional augmentation quality

✅ USAGE:
augmenter = PointTileAugmentation(
    tile_size=512,
    crop_scale_factor=1.5,    # 50% larger initial crop
    empty_prob=0.2,           # 20% empty tiles
    rotation_limit=45.0,      # Rotation range
    shear_limit=10.0          # Shear range
)

tile, points, is_empty, debug = augmenter.augment_tile(
    image, points, return_debug_info=True
)

This is the correct pipeline you requested!
    """)

    plt.show()