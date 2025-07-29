import random
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

import pandas as pd
from albumentations.core.transforms_interface import DualTransform
import albumentations as A
import matplotlib.pyplot as plt
import cv2

from utils.augmentations import ObjectAwareRandomCrop


def draw_keypoints(image, keypoints, keypoint_labels=None, color=(0, 255, 0), radius=5):
    """Draw keypoints on image."""
    img_copy = image.copy()
    if keypoint_labels is None:
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(img_copy, (int(x), int(y)), radius, color, -1)
            cv2.putText(img_copy, str(i), (int(x) + 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    else:
        for ((x, y), label) in zip(keypoints, keypoint_labels):
            cv2.circle(img_copy, (int(x), int(y)), radius, color, -1)
            cv2.putText(img_copy, str(label), (int(x) + 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    return img_copy


def draw_crop_area(image, crop_x, crop_y, width, height, color=(255, 255, 0), thickness=5):
    """Draw crop area on image."""
    img_copy = image.copy()
    cv2.rectangle(img_copy, (crop_x, crop_y), (crop_x + width, crop_y + height), color, thickness)
    return img_copy

if __name__ == "__main__":


    # read the image
    image = cv2.imread("/home/cwinkelmann/work/Herdnet/data/floreana_sample/train/Default/FMO04___DJI_0906.JPG")
    keypoints = pd.read_csv('/home/cwinkelmann/work/Herdnet/data/floreana_sample/train/herdnet_format.csv')
    keypoints = keypoints[keypoints.species == 'iguana_point'].reset_index(drop=False).rename(columns={'index': 'id'})

    keypoints = keypoints[['x', 'y', 'id']].values.tolist()


    first_crop_length = 1000
    second_crop_length = 512

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

        A.Rotate(limit=180, p=1),  # Optional rotation for variety
        ObjectAwareRandomCrop(
            height=second_crop_length,  # Much smaller than the image
            width=second_crop_length,
            attempts=10,
            p=1.0,
            edge_black_blobs=True,  # Optional: add black edges to the crop
        ),
        # TODO hide, half of the iguana by add a square mask k deegrees from the point
    ], keypoint_params=A.KeypointParams(format='xy'))

    # Apply the transform
    transformed = transform(image=image, keypoints=keypoints)
    cropped_image = transformed['image']
    cropped_keypoints = transformed['keypoints']

    # Get crop parameters for visualization
    crop_x, crop_y = transform.transforms[0].last_crop

    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Original image (downsampled for display) with keypoints and crop area
    scale_factor = 0.2  # Downsample for display
    small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    small_keypoints = [(kp[0] * scale_factor, kp[1] * scale_factor) for kp in [(k[0], k[1]) for k in keypoints]]
    small_keypoints_labels = [(k[2]) for k in keypoints]

    img_with_kp = draw_keypoints(small_image, small_keypoints, keypoint_labels=small_keypoints_labels, color=(0, 255, 0), radius=5)
    img_with_crop = draw_crop_area(img_with_kp,
                                   int(crop_x * scale_factor),
                                   int(crop_y * scale_factor),
                                   int(first_crop_length * scale_factor),
                                   int(first_crop_length * scale_factor),
                                   color=(255, 255, 0), thickness=3)

    axes[0].imshow(cv2.cvtColor(img_with_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f'Original Image (5000x4000, shown at 20%)\nGreen=Keypoints, Yellow=Crop Area\nCrop at ({crop_x}, {crop_y})',
        fontsize=12)
    axes[0].axis('off')

    # Cropped image with preserved keypoints
    cropped_keypoint_coords = [(kp[0], kp[1]) for kp in cropped_keypoints]
    cropped_keypoint_labels = [kp[2] for kp in cropped_keypoints]
    cropped_with_kp = draw_keypoints(cropped_image, cropped_keypoint_coords, keypoint_labels=cropped_keypoint_labels, color=(0, 255, 0), radius=10)

    axes[1].imshow(cv2.cvtColor(cropped_with_kp, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Cropped Image ({second_crop_length}x{second_crop_length})\nKeypoints preserved: {len(cropped_keypoints)}/{len(keypoints)}',
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