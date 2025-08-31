import random
import numpy as np
import cv2
from albumentations.core.transforms_interface import DualTransform

import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from albumentations.core.transforms_interface import DualTransform
# TODO move these over to transforms
class ObjectAwareRandomCrop(DualTransform):
    """
    Random crop that guarantees at least one keypoint remains within the cropped area.

    This transformation attempts to find a valid crop position multiple times before falling back
    to a crop that ensures at least one keypoint is included.

    Args:
        height (int): Height of the crop.
        width (int): Width of the crop.
        attempts (int): Number of random attempts to try before using guaranteed method. Default: 10.
        always_apply (bool): Whether to always apply this transform. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(
            self,
            height: int,
            width: int,
            attempts: int = 10,
            always_apply: bool = False,
            p: float = 1.0,
            edge_black_blobs = False,
            empty_probability=0.0
    ):
        super().__init__(always_apply, p)
        self.last_crop = None
        self.height = height
        self.width = width
        self.attempts = attempts
        self.empty_probability = empty_probability
        self.edge_black_blobs = edge_black_blobs

        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")

    def _has_keypoint_in_crop(
            self,
            crop_x: int,
            crop_y: int,
            keypoints: List[Tuple[float, float]]
    ) -> bool:
        """Check if at least one keypoint is within the crop area."""
        crop_x_max = crop_x + self.width
        crop_y_max = crop_y + self.height

        for x, y in keypoints:
            if crop_x <= x <= crop_x_max and crop_y <= y <= crop_y_max:
                return True
        return False

    def _get_guaranteed_crop_position(
            self,
            image_height: int,
            image_width: int,
            keypoints: List[Tuple[float, float]]
    ) -> Tuple[int, int]:
        """Get a crop position that guarantees at least one keypoint is included."""
        if not keypoints:
            # If no keypoints, just do a random crop
            max_crop_x = image_width - self.width
            max_crop_y = image_height - self.height
            return random.randint(0, max_crop_x), random.randint(0, max_crop_y)

        # Pick a random keypoint to center the crop around
        target_keypoint = random.choice(keypoints)
        target_x, target_y = target_keypoint

        # Center the crop on this keypoint
        crop_x = int(target_x - self.width // 2)
        crop_y = int(target_y - self.height // 2)

        # Ensure crop stays within image bounds
        crop_x = max(0, min(crop_x, image_width - self.width))
        crop_y = max(0, min(crop_y, image_height - self.height))

        return crop_x, crop_y

    def apply(self, img: np.ndarray, crop_x: int = 0, crop_y: int = 0, **params) -> np.ndarray:
        """Apply the crop to the image."""
        return img[crop_y:crop_y + self.height, crop_x:crop_x + self.width]

    def apply_to_keypoint(self, keypoint: Tuple[float, float, float, float], crop_x: int = 0, crop_y: int = 0,
                          **params) -> Tuple[float, float, float, float]:
        """Apply the crop to keypoints."""
        x, y, angle, scale = keypoint

        # Adjust keypoint coordinates relative to the crop
        x_new = x - crop_x
        y_new = y - crop_y

        return x_new, y_new, angle, scale

    def get_params_dependent_on_targets(self, params: Dict) -> Dict:
        """Generate parameters for the transformation."""
        img = params['image']
        keypoints = params.get('keypoints', [])
        image_height, image_width = img.shape[:2]
        return_iguana = random.uniform(0, 1) > self.empty_probability
        # Validate crop size
        if self.height > image_height or self.width > image_width:
            raise ValueError(
                f"Crop size ({self.width}x{self.height}) is larger than image size ({image_width}x{image_height})")

        # Extract x,y coordinates from keypoints
        keypoint_coords = [(kp[0], kp[1]) for kp in keypoints]
        if len(keypoint_coords) > 1:
            pass
        if not keypoint_coords:
            # If no keypoints, fall back to regular random crop
            max_crop_x = image_width - self.width
            max_crop_y = image_height - self.height
            crop_x = random.randint(0, max_crop_x)
            crop_y = random.randint(0, max_crop_y)
            self.last_crop = (crop_x, crop_y)

            return {'crop_x': crop_x, 'crop_y': crop_y}

        # Try random positions first
        for _ in range(self.attempts):
            max_crop_x = image_width - self.width
            max_crop_y = image_height - self.height

            crop_x = random.randint(0, max_crop_x)
            crop_y = random.randint(0, max_crop_y)

            if self._has_keypoint_in_crop(crop_x, crop_y, keypoint_coords) and return_iguana:
                self.last_crop = (crop_x, crop_y)
                return {'crop_x': crop_x, 'crop_y': crop_y}
            elif not self._has_keypoint_in_crop(crop_x, crop_y, keypoint_coords) and not return_iguana:
                self.last_crop = (crop_x, crop_y)
                return {'crop_x': crop_x, 'crop_y': crop_y}
            # the other cases are not of interest that much

        # If random attempts failed, use guaranteed method
        crop_x, crop_y = self._get_guaranteed_crop_position(image_height, image_width, keypoint_coords)
        self.last_crop = (crop_x, crop_y)
        return {'crop_x': crop_x, 'crop_y': crop_y}

    @property
    def targets_as_params(self) -> List[str]:
        return ['image', 'keypoints']

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ('height', 'width', 'attempts')



class ObjectAwareRandomCropEdgeBlackout(ObjectAwareRandomCrop):
    """
    Object-aware random crop that guarantees at least one keypoint remains within the cropped area.
    This version is designed to be used with a different set of parameters or configurations.
    FIXME this is quickfix
    """

    def __init__(
            self,
            height: int,
            width: int,
            attempts: int = 10,
            always_apply: bool = False,
            p: float = 1.0,
            edge_black_blobs=False,
            empty_probability=0.0
    ):
        super().__init__(height, width, attempts, always_apply, p, edge_black_blobs, empty_probability)






class PasspartoutAugmentation(DualTransform):
    def __init__(
        self,
        inner_radius_ratio: float = 0.3,
        outer_radius_ratio: float = 0.7,
        center_x_ratio: float = 0.5,
        center_y_ratio: float = 0.5,
        keypoint_threshold: float = 0.5,
        gaussian_sigma_ratio: float = 0.3,
        intensity: float = 0.8,
        randomize_center: bool = True,
        randomize_radius: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        # Parameter validation
        if not (0.0 <= inner_radius_ratio < outer_radius_ratio <= 1.0):
            raise ValueError("inner_radius_ratio must be < outer_radius_ratio and both ∈ [0.0, 1.0]")
        if not (0.0 <= keypoint_threshold <= 1.0):
            raise ValueError("keypoint_threshold must be ∈ [0.0, 1.0]")
        if not (0.0 <= intensity <= 1.0):
            raise ValueError("intensity must be ∈ [0.0, 1.0]")

        self.inner_radius_ratio = inner_radius_ratio
        self.outer_radius_ratio = outer_radius_ratio
        self.center_x_ratio = center_x_ratio
        self.center_y_ratio = center_y_ratio
        self.keypoint_threshold = keypoint_threshold
        self.gaussian_sigma_ratio = gaussian_sigma_ratio
        self.intensity = intensity
        self.randomize_center = randomize_center
        self.randomize_radius = randomize_radius

        self.mask_cache = None  # to avoid recomputing for same shape

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]
        diag = np.sqrt(h ** 2 + w ** 2)

        # Possibly randomize center and radius
        cx = np.random.uniform(0.3, 0.7) if self.randomize_center else self.center_x_ratio
        cy = np.random.uniform(0.3, 0.7) if self.randomize_center else self.center_y_ratio
        inner_ratio = np.random.uniform(0.2, self.inner_radius_ratio) if self.randomize_radius else self.inner_radius_ratio
        outer_ratio = np.random.uniform(self.outer_radius_ratio, 1.0) if self.randomize_radius else self.outer_radius_ratio

        inner_radius = inner_ratio * diag
        outer_radius = outer_ratio * diag
        center = (int(cx * w), int(cy * h))

        return {
            "center": center,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "mask": self._make_vignette_mask(h, w, center, inner_radius, outer_radius)
        }

    def _make_vignette_mask(self, h, w, center, inner_r, outer_r):
        """Creates a soft circular mask for vignette effect: bright center → dark edges."""
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = np.ones((h, w), dtype=np.float32)

        # Create a transition zone from inner_r to outer_r
        transition_zone = np.clip((dist - inner_r) / (outer_r - inner_r), 0, 1)

        sigma = self.gaussian_sigma_ratio * (outer_r - inner_r)

        # Apply smooth fall-off using a Gaussian curve
        blend = np.exp(-0.5 * (transition_zone * (outer_r - inner_r) / sigma) ** 2)

        # Blend intensity toward the outer edge
        mask = 1.0 - self.intensity * (1.0 - blend)

        # Outside outer radius, darken completely
        mask[dist >= outer_r] = 1.0 - self.intensity

        return mask.astype(np.float32)

    def apply(self, image, **params):
        mask = params["mask"]
        if image.ndim == 2:
            return (image * mask).astype(image.dtype)
        else:
            return (image * mask[..., None]).astype(image.dtype)

    def apply_to_keypoint(self, keypoint, **params):
        x, y = keypoint[:2]
        mask = params["mask"]
        h, w = mask.shape
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= x_int < w and 0 <= y_int < h:
            if mask[y_int, x_int] < self.keypoint_threshold:
                # Mark as invisible (so it can be filtered later)
                return (-1, -1) + keypoint[2:]  # assume format (x, y, angle, scale) or (x, y)
        return keypoint

    def apply_to_mask(self, mask, **params):
        return mask  # mask is unchanged

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "inner_radius_ratio",
            "outer_radius_ratio",
            "center_x_ratio",
            "center_y_ratio",
            "keypoint_threshold",
            "gaussian_sigma_ratio",
            "intensity",
            "randomize_center",
            "randomize_radius"
        )
