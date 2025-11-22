"""
Simple test script with fake data to demonstrate ObjectAwareRandomCrop.
Generates synthetic images with colored circles as "animals" and applies the augmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import sys

from ObjectAwareRandomCrop_CORRECT import ObjectAwareRandomCrop

# Try to import albumentations
try:
    import albumentations as A

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Note: Albumentations not found, using direct transform application")


def generate_fake_image(width=800, height=600, num_animals=5):
    """
    Generate a fake image with colored circles representing animals.
    Returns image and keypoints (center of each circle).
    """
    # Create gradient background (like sky/grass)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (blue)
    for i in range(height // 2):
        blue = int(200 - (i / (height // 2)) * 50)
        image[i, :] = [100, 150, blue]

    # Grass gradient (green)
    for i in range(height // 2, height):
        green = int(150 + ((i - height // 2) / (height // 2)) * 50)
        image[i, :] = [50, green, 80]

    # Generate random "animals" (colored circles)
    keypoints = []
    animal_colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
    ]

    np.random.seed(42)
    for i in range(num_animals):
        # Random position
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)

        # Random size
        radius = np.random.randint(20, 40)

        # Draw circle (animal)
        color = animal_colors[i % len(animal_colors)]
        cv2.circle(image, (x, y), radius, color, -1)

        # Add border
        cv2.circle(image, (x, y), radius, (0, 0, 0), 2)

        # Add label
        label = f"A{i + 1}"
        cv2.putText(image, label, (x - 10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        keypoints.append((x, y, 0, 1))  # (x, y, angle, scale)

    return image, keypoints


def visualize_augmentation(original_img, original_kps, augmented_img, augmented_kps,
                           crop_params=None, transform=None):
    """
    Visualize original and augmented images side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Original image
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')

    # Draw keypoints
    for i, (x, y, _, _) in enumerate(original_kps):
        ax1.plot(x, y, 'o', color='yellow', markersize=12,
                 markeredgecolor='black', markeredgewidth=2)
        ax1.text(x + 5, y - 5, f'A{i + 1}', color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Draw crop region if available
    if crop_params and transform:
        crop_x = crop_params.get('crop_x', 0)
        crop_y = crop_params.get('crop_y', 0)

        rect = patches.Rectangle(
            (crop_x, crop_y), transform.width, transform.height,
            linewidth=3, edgecolor='lime', facecolor='none', linestyle='--'
        )
        ax1.add_patch(rect)

        # Draw safe zone
        if transform.min_edge_distance > 0:
            inner_rect = patches.Rectangle(
                (crop_x + transform.min_edge_distance,
                 crop_y + transform.min_edge_distance),
                transform.width - 2 * transform.min_edge_distance,
                transform.height - 2 * transform.min_edge_distance,
                linewidth=2, edgecolor='cyan', facecolor='none', linestyle=':'
            )
            ax1.add_patch(inner_rect)

        ax1.text(crop_x + 5, crop_y - 10, 'Crop Region',
                 color='lime', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax1.set_xlim(0, original_img.shape[1])
    ax1.set_ylim(original_img.shape[0], 0)
    ax1.grid(True, alpha=0.3)

    # Augmented image
    ax2.imshow(cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('After ObjectAwareRandomCrop', fontsize=14, fontweight='bold')

    # Draw transformed keypoints
    for i, (x, y, _, _) in enumerate(augmented_kps):
        ax2.plot(x, y, 'o', color='yellow', markersize=12,
                 markeredgecolor='black', markeredgewidth=2)
        ax2.text(x + 5, y - 5, f'A{i + 1}', color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Calculate distance to edges
        if transform:
            dist_left = x
            dist_right = transform.width - x
            dist_top = y
            dist_bottom = transform.height - y
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            # Show distance for each keypoint
            status = '✓' if min_dist >= transform.min_edge_distance else '·'
            ax2.text(x + 5, y + 15, f'{status} {min_dist:.0f}px',
                     color='lime' if min_dist >= transform.min_edge_distance else 'orange',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Draw safe zone boundary
    if transform and transform.min_edge_distance > 0:
        boundary_rect = patches.Rectangle(
            (transform.min_edge_distance, transform.min_edge_distance),
            transform.width - 2 * transform.min_edge_distance,
            transform.height - 2 * transform.min_edge_distance,
            linewidth=2, edgecolor='cyan', facecolor='none', linestyle=':'
        )
        ax2.add_patch(boundary_rect)
        ax2.text(10, 25, f'Safe Zone\n(≥{transform.min_edge_distance}px)',
                 color='cyan', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))

    ax2.set_xlim(0, augmented_img.shape[1])
    ax2.set_ylim(augmented_img.shape[0], 0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("ObjectAwareRandomCrop - Fake Data Test")
    print("=" * 70)

    # Generate fake data
    print("\n1. Generating fake image with 5 'animals'...")
    image, keypoints = generate_fake_image(width=800, height=600, num_animals=5)
    print(f"   Image shape: {image.shape}")
    print(f"   Number of animals: {len(keypoints)}")
    print(f"   Animal positions: {[(int(x), int(y)) for x, y, _, _ in keypoints]}")

    # Create transform
    print("\n2. Creating ObjectAwareRandomCrop transform...")
    transform = ObjectAwareRandomCrop(
        height=400,
        width=400,
        min_edge_distance=30,
        empty_probability=0.0,  # Always include animals for this demo
        max_attempts=20,
        p=1.0
    )
    print(f"   Crop size: {transform.width}x{transform.height}")
    print(f"   Min edge distance: {transform.min_edge_distance}px")

    # Apply transform multiple times
    print("\n3. Applying augmentation 3 times...")

    if HAS_ALBUMENTATIONS:
        # Use Albumentations pipeline
        aug = A.Compose([
            transform,
        ], keypoint_params=A.KeypointParams(format='xysa', remove_invisible=True))

        for i in range(3):
            print(f"\n   --- Sample {i + 1} ---")
            result = aug(image=image.copy(), keypoints=keypoints)

            augmented_img = result['image']
            augmented_kps = result['keypoints']

            print(f"   Animals in crop: {len(augmented_kps)}/{len(keypoints)}")

            # Get crop params for visualization
            params = transform.get_params_dependent_on_targets({
                'image': image,
                'keypoints': keypoints
            })

            # Visualize
            fig = visualize_augmentation(
                image, keypoints,
                augmented_img, augmented_kps,
                crop_params=params, transform=transform
            )

            filename = f'outputs_demo_sample_{i + 1}.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"   Saved: demo_sample_{i + 1}.png")
            plt.close(fig)

    else:
        # Direct application without Albumentations
        for i in range(3):
            print(f"\n   --- Sample {i + 1} ---")

            # Get crop parameters
            params = transform.get_params_dependent_on_targets({
                'image': image,
                'keypoints': keypoints
            })

            # Apply crop
            augmented_img = transform.apply(
                image.copy(),
                crop_x=params['crop_x'],
                crop_y=params['crop_y']
            )

            # Transform keypoints
            augmented_kps = []
            for kp in keypoints:
                new_kp = transform.apply_to_keypoint(
                    kp,
                    crop_x=params['crop_x'],
                    crop_y=params['crop_y']
                )
                # Only keep if within bounds
                if 0 <= new_kp[0] < transform.width and 0 <= new_kp[1] < transform.height:
                    augmented_kps.append(new_kp)

            print(f"   Animals in crop: {len(augmented_kps)}/{len(keypoints)}")

            # Visualize
            fig = visualize_augmentation(
                image, keypoints,
                augmented_img, augmented_kps,
                crop_params=params, transform=transform
            )

            filename = f'outputs_demo_sample_{i + 1}.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"   Saved: demo_sample_{i + 1}.png")
            plt.close(fig)

    print("\n" + "=" * 70)
    print("✅ Test complete! Check the demo_sample_*.png files.")
    print("=" * 70)

    print("\nKey observations:")
    print("- The crop is positioned to keep at least one animal well-framed")
    print("- Selected animal is ≥30px from all crop edges (✓)")
    print("- Other animals may be closer to edges (·) - this is OK!")
    print("- Cyan dotted line shows the 'safe zone'")


if __name__ == "__main__":
    main()