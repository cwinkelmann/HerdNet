import random
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from albumentations.core.transforms_interface import DualTransform

from utils.augmentations import ObjectAwareRandomCrop

# Example usage with visualization
if __name__ == "__main__":
    import albumentations as A
    import matplotlib.pyplot as plt
    import cv2


    def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=5):
        """Draw keypoints on image."""
        img_copy = image.copy()
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(img_copy, (int(x), int(y)), radius, color, -1)
            cv2.putText(img_copy, str(i), (int(x) + 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        return img_copy


    def draw_crop_area(image, crop_x, crop_y, width, height, color=(255, 255, 0), thickness=5):
        """Draw crop area on image."""
        img_copy = image.copy()
        cv2.rectangle(img_copy, (crop_x, crop_y), (crop_x + width, crop_y + height), color, thickness)
        return img_copy


    # Create a large synthetic image (5000x4000 as in your example)
    image = np.zeros((4000, 5000, 3), dtype=np.uint8)

    # Add some background patterns to see the structure
    for i in range(0, 5000, 200):
        cv2.line(image, (i, 0), (i, 4000), (30, 30, 30), 2)
    for i in range(0, 4000, 200):
        cv2.line(image, (0, i), (5000, i), (30, 30, 30), 2)

    # Add some colored regions scattered around
    cv2.rectangle(image, (1000, 800), (1500, 1200), (100, 150, 200), -1)
    cv2.rectangle(image, (3000, 2500), (3800, 3200), (200, 100, 150), -1)
    cv2.rectangle(image, (500, 3000), (1200, 3600), (150, 200, 100), -1)
    cv2.rectangle(image, (2100, 1500), (2700, 1800), (110, 250, 200), -1)

    # Add some noise
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)

    # Create sparse keypoints in the middle area (like your example)
    keypoints = [
        (2500, 2000, 0, 1),  # center-ish
        (2200, 1800, 0, 1),  # near center
        (2800, 2200, 0, 1),  # near center
        (2600, 1900, 0, 1),  # near center
    ]

    print(f"Original image shape: {image.shape}")
    print(f"Keypoints: {[(int(kp[0]), int(kp[1])) for kp in keypoints]}")

    first_crop_length = 1000

    # Create the transform with a reasonable crop size
    transform = A.Compose([
        ObjectAwareRandomCrop(
            height=first_crop_length,  # Much smaller than the image
            width=first_crop_length,
            attempts=10,
            empty_probability=0.1,
            p=1.0
        ),
        # Rotate
        A.Rotate(limit=180, p=0.5),  # Optional rotation for variety
        ObjectAwareRandomCrop(
            height=512,  # Much smaller than the image
            width=512,
            attempts=10,
            p=1.0,
            edge_black_blobs=True,  # Optional: add black edges to the crop
        )
    ], keypoint_params=A.KeypointParams(format='xy'))

    # Apply the transform
    transformed = transform(image=image, keypoints=keypoints)
    cropped_image = transformed['image']
    cropped_keypoints = transformed['keypoints']

    # Get crop parameters for visualization
    crop_params = transform.transforms[0].get_params_dependent_on_targets({'image': image, 'keypoints': keypoints})

    crop_x, crop_y = transform.transforms[0].last_crop
    crop_x, crop_y = crop_params['crop_x'], crop_params['crop_y']

    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Original image (downsampled for display) with keypoints and crop area
    scale_factor = 0.2  # Downsample for display
    small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    small_keypoints = [(kp[0] * scale_factor, kp[1] * scale_factor) for kp in [(k[0], k[1]) for k in keypoints]]

    img_with_kp = draw_keypoints(small_image, small_keypoints, color=(0, 255, 0), radius=15)
    img_with_crop = draw_crop_area(img_with_kp,
                                   int(crop_x * scale_factor),
                                   int(crop_y * scale_factor),
                                   int(first_crop_length * scale_factor),
                                   int(first_crop_length * scale_factor),
                                   color=(255, 255, 0), thickness=8)

    axes[0].imshow(cv2.cvtColor(img_with_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f'Original Image (5000x4000, shown at 20%)\nGreen=Keypoints, Yellow=Crop Area\nCrop at ({crop_x}, {crop_y})',
        fontsize=12)
    axes[0].axis('off')

    # Cropped image with preserved keypoints
    cropped_keypoint_coords = [(kp[0], kp[1]) for kp in cropped_keypoints]
    cropped_with_kp = draw_keypoints(cropped_image, cropped_keypoint_coords, color=(0, 255, 0), radius=10)

    axes[1].imshow(cv2.cvtColor(cropped_with_kp, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Cropped Image ({first_crop_length}x{first_crop_length})\nKeypoints preserved: {len(cropped_keypoints)}/{len(keypoints)}',
                      fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Print detailed information
    print(f"\nCropped image shape: {cropped_image.shape}")
    print(f"Crop position: ({crop_x}, {crop_y})")
    print(f"\nOriginal keypoints: {[(int(kp[0]), int(kp[1])) for kp in keypoints]}")
    print(f"Cropped keypoints: {[(int(kp[0]), int(kp[1])) for kp in cropped_keypoints]}")
    print(f"Keypoints preserved: {len(cropped_keypoints)}/{len(keypoints)}")

    # Show which keypoints are in the crop area
    print(f"\nKeypoints in crop area:")
    for i, (orig_kp, crop_kp) in enumerate(zip(keypoints, cropped_keypoints)):
        if 0 <= crop_kp[0] <= first_crop_length and 0 <= crop_kp[1] <= first_crop_length:
            print(
                f"  Keypoint {i}: Original({int(orig_kp[0])}, {int(orig_kp[1])}) -> Cropped({int(crop_kp[0])}, {int(crop_kp[1])})")