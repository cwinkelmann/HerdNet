import random

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from albumentations.core.transforms_interface import DualTransform
# TODO move these over to transforms
# class ObjectAwareRandomCrop(DualTransform):
#     """
#     Random crop that guarantees at least one keypoint remains within the cropped area.
#
#     This transformation attempts to find a valid crop position multiple times before falling back
#     to a crop that ensures at least one keypoint is included.
#
#     Args:
#         height (int): Height of the crop.
#         width (int): Width of the crop.
#         attempts (int): Number of random attempts to try before using guaranteed method. Default: 10.
#         always_apply (bool): Whether to always apply this transform. Default: False.
#         p (float): Probability of applying the transform. Default: 1.0.
#     """
#
#     def __init__(
#             self,
#             height: int,
#             width: int,
#             attempts: int = 10,
#             always_apply: bool = False,
#             p: float = 1.0,
#             edge_black_blobs = False,
#             empty_probability=0.0
#     ):
#         super().__init__(always_apply, p)
#         self.last_crop = None
#         self.height = height
#         self.width = width
#         self.attempts = attempts
#         self.empty_probability = empty_probability
#         self.edge_black_blobs = edge_black_blobs
#
#         if self.attempts < 1:
#             raise ValueError("attempts must be at least 1")
#
#     def _has_keypoint_in_crop(
#             self,
#             crop_x: int,
#             crop_y: int,
#             keypoints: List[Tuple[float, float]]
#     ) -> bool:
#         """Check if at least one keypoint is within the crop area."""
#         crop_x_max = crop_x + self.width
#         crop_y_max = crop_y + self.height
#
#         for x, y in keypoints:
#             if crop_x <= x <= crop_x_max and crop_y <= y <= crop_y_max:
#             # if crop_x <= x < crop_x_max and crop_y <= y < crop_y_max:
#
#                 return True
#         return False
#
#     def _get_guaranteed_crop_position(
#             self,
#             image_height: int,
#             image_width: int,
#             keypoints: List[Tuple[float, float]]
#     ) -> Tuple[int, int]:
#         """Get a crop position that guarantees at least one keypoint is included."""
#         if not keypoints:
#             # If no keypoints, just do a random crop
#             max_crop_x = image_width - self.width
#             max_crop_y = image_height - self.height
#             return random.randint(0, max_crop_x), random.randint(0, max_crop_y)
#
#         # Pick a random keypoint to center the crop around
#         target_keypoint = random.choice(keypoints)
#         target_x, target_y = target_keypoint
#
#         # Center the crop on this keypoint
#         crop_x = int(target_x - self.width // 2)
#         crop_y = int(target_y - self.height // 2)
#
#         # Ensure crop stays within image bounds
#         crop_x = max(0, min(crop_x, image_width - self.width))
#         crop_y = max(0, min(crop_y, image_height - self.height))
#
#         return crop_x, crop_y
#
#     def apply(self, img: np.ndarray, crop_x: int = 0, crop_y: int = 0, **params) -> np.ndarray:
#         """Apply the crop to the image."""
#         return img[crop_y:crop_y + self.height, crop_x:crop_x + self.width]
#
#     def apply_to_keypoint(self, keypoint: Tuple[float, float, float, float], crop_x: int = 0, crop_y: int = 0,
#                           **params) -> Tuple[float, float, float, float]:
#         """Apply the crop to keypoints."""
#         x, y, angle, scale = keypoint
#
#         # Adjust keypoint coordinates relative to the crop
#         x_new = x - crop_x
#         y_new = y - crop_y
#
#         return x_new, y_new, angle, scale
#
#     def get_params_dependent_on_targets(self, params: Dict) -> Dict:
#         """Generate parameters for the transformation."""
#         img = params['image']
#         keypoints = params.get('keypoints', [])
#         image_height, image_width = img.shape[:2]
#         return_iguana = random.uniform(0, 1) > self.empty_probability
#         # Validate crop size
#         if self.height > image_height or self.width > image_width:
#             raise ValueError(
#                 f"Crop size ({self.width}x{self.height}) is larger than image size ({image_width}x{image_height})")
#
#         # Extract x,y coordinates from keypoints
#         keypoint_coords = [(kp[0], kp[1]) for kp in keypoints]
#         # if len(keypoint_coords) > 1:
#         #     pass
#         if not keypoint_coords:
#             # If no keypoints, fall back to regular random crop
#             max_crop_x = image_width - self.width
#             max_crop_y = image_height - self.height
#             crop_x = random.randint(0, max_crop_x)
#             crop_y = random.randint(0, max_crop_y)
#             self.last_crop = (crop_x, crop_y)
#
#             return {'crop_x': crop_x, 'crop_y': crop_y}
#
#         # Try random positions first
#         for _ in range(self.attempts):
#             max_crop_x = image_width - self.width
#             max_crop_y = image_height - self.height
#
#             crop_x = random.randint(0, max_crop_x)
#             crop_y = random.randint(0, max_crop_y)
#
#             if self._has_keypoint_in_crop(crop_x, crop_y, keypoint_coords) and return_iguana:
#                 self.last_crop = (crop_x, crop_y)
#                 return {'crop_x': crop_x, 'crop_y': crop_y}
#             elif not self._has_keypoint_in_crop(crop_x, crop_y, keypoint_coords) and not return_iguana:
#                 self.last_crop = (crop_x, crop_y)
#                 return {'crop_x': crop_x, 'crop_y': crop_y}
#             # the other cases are not of interest that much
#
#         # If random attempts failed, use guaranteed method
#         crop_x, crop_y = self._get_guaranteed_crop_position(image_height, image_width, keypoint_coords)
#         self.last_crop = (crop_x, crop_y)
#         return {'crop_x': crop_x, 'crop_y': crop_y}
#
#     @property
#     def targets_as_params(self) -> List[str]:
#         return ['image', 'keypoints']
#
#     def get_transform_init_args_names(self) -> Tuple[str, ...]:
#         return ('height', 'width', 'attempts')

"""
Fixed ObjectAwareRandomCrop transform with proper min_edge_distance enforcement.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
import warnings

"""
Fixed ObjectAwareRandomCrop transform - Albumentations compatible.
Only enforces min_edge_distance for the SELECTED keypoint.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
import warnings
from albumentations import DualTransform


class ObjectAwareRandomCrop(DualTransform):
    """
    Random crop that ensures at least one keypoint is included, with stitch-aware positioning.

    Designed for training models that will run inference with a sliding-window stitcher.
    The crop position is chosen so the selected keypoint can land anywhere in the crop —
    including near edges — matching the distribution the model will see during tiled inference.

    Why translation diversity matters: HerdNet runs inference in fixed 512x512 tiles at
    stride=392. If training crops over-represent any sub-tile position (especially the
    centre), the model learns a position prior tied to that location and at inference
    only fires when the stitcher's grid happens to align an iguana there. With 80% drone
    overlap, that grid alignment varies randomly across consecutive frames, producing
    erratic detection patterns. See investigation in
    `metashape_mosaicing/tests/center_bias_proof.png`.

    With the recommended defaults (`min_edge_distance=0`, `translation_jitter=0`) the
    selected keypoint is uniformly distributed in the crop — that's the right setting
    for stitcher compatibility. Use `translation_jitter` to add extra position noise
    on top, which is robust to subtle centre-bias paths in callers that pass
    `min_edge_distance > 0`.

    Args:
        height (int): Height of the crop.
        width (int): Width of the crop.
        min_edge_distance (int): Minimum distance in pixels between THE SELECTED keypoint and
            crop edge. Set to 0 for uniform placement (recommended for stitcher compatibility).
            Default: 0.
        empty_probability (float): Probability of creating a crop without any keypoints
            (pure background). Default: 0.0.
        edge_probability (float): Probability of deliberately placing the selected keypoint
            in the edge zone (within `edge_zone` pixels of the crop border). This trains the
            model to be confident at tile boundaries during stitched inference. Default: 0.0.
        edge_zone (int): Width of the edge zone in pixels. Should match the stitcher overlap
            (e.g. 120px for overlap=120). Only used when edge_probability > 0. Default: 120.
        max_attempts (int): Maximum attempts to find a valid crop. Default: 10.
        translation_jitter (int): After picking a crop position, perturb it by a uniform
            integer in [-translation_jitter, +translation_jitter] on each axis. Clamped to
            keep ≥1 keypoint inside the crop and the crop inside the image. Defaults to 0
            (no extra jitter). Setting this to ~stride/2 of your inference stitcher
            (e.g. 196 for the iguana config's stride=392) explicitly randomises the
            sub-tile position so the model can't latch onto any particular position prior.
            Recommended when min_edge_distance > 0 or when retraining a model that's
            shown centre-fixation pathology at inference.
        always_apply (bool): Whether to always apply this transform. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(
            self,
            height: int,
            width: int,
            min_edge_distance: int = 0,
            empty_probability: float = 0.0,
            edge_probability: float = 0.0,
            edge_zone: int = 120,
            max_attempts: int = 10,
            translation_jitter: int = 0,
            always_apply: bool = False,
            p: float = 1.0,
    ):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.min_edge_distance = min_edge_distance
        self.empty_probability = empty_probability
        self.edge_probability = edge_probability
        self.edge_zone = edge_zone
        self.max_attempts = max_attempts
        # translation_jitter: after picking a crop position, perturb it by a
        # uniform integer in [-translation_jitter, +translation_jitter] on
        # each axis (still clamped so the keypoint stays in the crop).
        # Set this to ~stride/2 of your inference stitcher (~196 for the
        # iguana config) to break any residual centre-bias the model might
        # otherwise learn from per-image position correlations.
        self.translation_jitter = translation_jitter

        if self.min_edge_distance < 0:
            raise ValueError("min_edge_distance must be non-negative")
        if not 0.0 <= self.empty_probability <= 1.0:
            raise ValueError("empty_probability must be between 0.0 and 1.0")
        if not 0.0 <= self.edge_probability <= 1.0:
            raise ValueError("edge_probability must be between 0.0 and 1.0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.translation_jitter < 0:
            raise ValueError("translation_jitter must be non-negative")

    def _is_keypoint_valid_for_crop(
            self,
            keypoint_x: float,
            keypoint_y: float,
            image_height: int,
            image_width: int
    ) -> bool:
        """
        Check if a keypoint can satisfy min_edge_distance constraint given image and crop size.

        Returns:
            True if it's possible to create a crop with this keypoint at min_edge_distance from edges.
        """
        # Check if there's enough space in the image for a crop with this keypoint
        # at min_edge_distance from all crop edges

        # For the keypoint to be at least min_edge_distance from left edge:
        # keypoint_x - crop_x >= min_edge_distance
        # => crop_x <= keypoint_x - min_edge_distance
        # Also: crop_x >= 0
        # So we need: keypoint_x >= min_edge_distance

        # For the keypoint to be at least min_edge_distance from right edge:
        # (crop_x + width) - keypoint_x >= min_edge_distance
        # => keypoint_x <= crop_x + width - min_edge_distance
        # Also: crop_x <= image_width - width
        # So we need: keypoint_x <= image_width - min_edge_distance

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
        """
        Calculate the valid range for crop position to keep keypoint at min_edge_distance from edges.

        Returns:
            Tuple of (x_range, y_range) where each range is (min, max) inclusive.
            Returns invalid range (min > max) if no valid crop exists.
        """
        # X coordinate constraints
        # Keypoint must be at least min_edge_distance from left edge of crop
        crop_x_max = int(keypoint_x - self.min_edge_distance)
        # Keypoint must be at least min_edge_distance from right edge of crop
        crop_x_min = int(keypoint_x - self.width + self.min_edge_distance)

        # Also constrain by image boundaries
        crop_x_min = max(0, crop_x_min)
        crop_x_max = min(image_width - self.width, crop_x_max)

        # Y coordinate constraints
        crop_y_max = int(keypoint_y - self.min_edge_distance)
        crop_y_min = int(keypoint_y - self.height + self.min_edge_distance)

        # Also constrain by image boundaries
        crop_y_min = max(0, crop_y_min)
        crop_y_max = min(image_height - self.height, crop_y_max)

        return (crop_x_min, crop_x_max), (crop_y_min, crop_y_max)

    def _get_random_crop_with_empty(
            self,
            image_height: int,
            image_width: int
    ) -> Tuple[int, int]:
        """Get a random crop position without considering keypoints."""
        max_crop_x = image_width - self.width
        max_crop_y = image_height - self.height

        crop_x = random.randint(0, max_crop_x)
        crop_y = random.randint(0, max_crop_y)

        return crop_x, crop_y

    def _verify_crop_constraint(
            self,
            keypoint_x: float,
            keypoint_y: float,
            crop_x: int,
            crop_y: int
    ) -> Tuple[bool, float]:
        """
        Verify if a crop satisfies the min_edge_distance constraint for a given keypoint.

        Returns:
            Tuple of (is_valid, min_distance)
        """
        # Calculate keypoint position within crop
        kp_x_in_crop = keypoint_x - crop_x
        kp_y_in_crop = keypoint_y - crop_y

        # Calculate distances to all edges
        dist_left = kp_x_in_crop
        dist_right = self.width - kp_x_in_crop
        dist_top = kp_y_in_crop
        dist_bottom = self.height - kp_y_in_crop

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        # Check if constraint is satisfied (use >= for inclusive comparison)
        is_valid = min_dist >= self.min_edge_distance

        return is_valid, min_dist

    def _get_edge_crop_position(
            self,
            keypoint_x: float,
            keypoint_y: float,
            image_height: int,
            image_width: int
    ) -> Tuple[int, int]:
        """
        Position the crop so the keypoint lands in the edge zone (within self.edge_zone
        pixels of the crop border). This simulates what the model sees at tile boundaries
        during stitched inference.
        """
        # Pick a random edge: 0=left, 1=right, 2=top, 3=bottom
        edge = random.randint(0, 3)
        ez = self.edge_zone

        if edge == 0:  # keypoint near LEFT edge of crop: kp_x_in_crop in [0, ez)
            # crop_x such that keypoint_x - crop_x is in [0, ez)
            crop_x_min = max(0, int(keypoint_x - ez + 1))
            crop_x_max = min(image_width - self.width, int(keypoint_x))
        elif edge == 1:  # keypoint near RIGHT edge
            crop_x_min = max(0, int(keypoint_x - self.width + 1))
            crop_x_max = min(image_width - self.width, int(keypoint_x - self.width + ez))
        else:
            # For top/bottom edges, x is unconstrained (just keep keypoint in crop)
            crop_x_min = max(0, int(keypoint_x - self.width + 1))
            crop_x_max = min(image_width - self.width, int(keypoint_x))

        if edge == 2:  # keypoint near TOP edge
            crop_y_min = max(0, int(keypoint_y - ez + 1))
            crop_y_max = min(image_height - self.height, int(keypoint_y))
        elif edge == 3:  # keypoint near BOTTOM edge
            crop_y_min = max(0, int(keypoint_y - self.height + 1))
            crop_y_max = min(image_height - self.height, int(keypoint_y - self.height + ez))
        else:
            crop_y_min = max(0, int(keypoint_y - self.height + 1))
            crop_y_max = min(image_height - self.height, int(keypoint_y))

        if crop_x_min > crop_x_max or crop_y_min > crop_y_max:
            # Fallback: just include the keypoint anywhere
            crop_x = max(0, min(int(keypoint_x - self.width // 2), image_width - self.width))
            crop_y = max(0, min(int(keypoint_y - self.height // 2), image_height - self.height))
            return crop_x, crop_y

        return random.randint(crop_x_min, crop_x_max), random.randint(crop_y_min, crop_y_max)

    def _get_crop_with_keypoint(
            self,
            keypoint_coords: List[Tuple[float, float]],
            image_height: int,
            image_width: int
    ) -> Tuple[int, int]:
        """
        Get a crop position that includes a random keypoint.

        With edge_probability > 0, sometimes places the keypoint in the edge zone
        to train the model for stitcher tile boundaries.
        """
        # Shuffle keypoints to try them in random order
        available_keypoints = keypoint_coords.copy()
        random.shuffle(available_keypoints)

        # Decide if this crop should have the keypoint in the edge zone
        force_edge = random.random() < self.edge_probability

        # Try to find a valid keypoint and crop position
        for attempt in range(min(int(self.max_attempts), len(available_keypoints) * 2)):
            target_x, target_y = available_keypoints[attempt % len(available_keypoints)]

            if force_edge:
                crop_x, crop_y = self._get_edge_crop_position(
                    target_x, target_y, image_height, image_width
                )
                return crop_x, crop_y

            # Check if this keypoint can possibly satisfy the constraint
            if not self._is_keypoint_valid_for_crop(target_x, target_y, image_height, image_width):
                continue

            # Get valid crop ranges
            (x_min, x_max), (y_min, y_max) = self._get_valid_crop_range(
                target_x, target_y, image_height, image_width
            )

            if x_min <= x_max and y_min <= y_max:
                crop_x = random.randint(x_min, x_max)
                crop_y = random.randint(y_min, y_max)

                is_valid, min_dist = self._verify_crop_constraint(target_x, target_y, crop_x, crop_y)
                if is_valid:
                    return crop_x, crop_y

        # Best-effort fallback: pick a UNIFORM crop position that includes a
        # random keypoint. Previously this centred the crop on the keypoint,
        # which silently injected a strong "iguana ⇒ centre" prior into the
        # model and broke translation equivariance at inference (the stitcher
        # then only fires on iguanas that happen to land near a tile centre,
        # producing erratic detection across consecutive overlapping drone
        # frames). See investigation in metashape_mosaicing/tests/.
        target_x, target_y = random.choice(keypoint_coords)
        x_min = max(0, int(target_x - self.width + 1))
        x_max = min(image_width - self.width, int(target_x))
        y_min = max(0, int(target_y - self.height + 1))
        y_max = min(image_height - self.height, int(target_y))
        crop_x = random.randint(x_min, x_max) if x_min <= x_max else max(0, min(int(target_x - self.width // 2), image_width - self.width))
        crop_y = random.randint(y_min, y_max) if y_min <= y_max else max(0, min(int(target_y - self.height // 2), image_height - self.height))

        return crop_x, crop_y

    def apply(self, img: np.ndarray, crop_x: int = 0, crop_y: int = 0, **params) -> np.ndarray:
        """Apply the crop to the image."""
        return img[crop_y:crop_y + self.height, crop_x:crop_x + self.width]

    def apply_to_keypoint(self, keypoint, crop_x: int = 0, crop_y: int = 0, **params):
        """Apply the crop to a single keypoint. Handles variable-length keypoint tuples."""
        x, y = keypoint[0], keypoint[1]
        rest = keypoint[2:] if len(keypoint) > 2 else ()
        return (x - crop_x, y - crop_y, *rest)

    def get_params_dependent_on_data(self, params: Dict, data: Dict) -> Dict:
        """Generate crop parameters based on image and keypoints (albumentations v2 API)."""
        img = data['image']
        keypoints = data.get('keypoints', [])
        image_height, image_width = img.shape[:2]

        # Validate crop size
        if self.height > image_height or self.width > image_width:
            raise ValueError(
                f"Crop size ({self.width}x{self.height}) is larger than "
                f"image size ({image_width}x{image_height})"
            )

        # Check if crop size allows for min_edge_distance
        if self.height < 2 * self.min_edge_distance or self.width < 2 * self.min_edge_distance:
            raise ValueError(
                f"Crop size ({self.width}x{self.height}) is too small for "
                f"min_edge_distance={self.min_edge_distance}. "
                f"Minimum crop size should be {2 * self.min_edge_distance}x{2 * self.min_edge_distance}"
            )

        # Extract x,y coordinates from keypoints
        keypoint_coords = [(kp[0], kp[1]) for kp in keypoints]

        # Decide whether to create empty crop or crop with keypoint
        create_empty_crop = random.random() < self.empty_probability

        if not keypoint_coords or create_empty_crop:
            # No keypoints or intentionally empty crop
            crop_x, crop_y = self._get_random_crop_with_empty(image_height, image_width)
        else:
            # Crop with keypoint at min distance from edges
            crop_x, crop_y = self._get_crop_with_keypoint(
                keypoint_coords, image_height, image_width
            )

        # Optional translation jitter: perturb the chosen crop position
        # uniformly in [-J, J] on each axis. Keeps any keypoint that was
        # inside the crop still inside (clamps if necessary), and never
        # leaves the image. This is the explicit "break the centre bias"
        # knob -- see CamouflageHerdNetConvNeXt centre-fixation analysis.
        if self.translation_jitter > 0 and keypoint_coords and not create_empty_crop:
            j = self.translation_jitter
            dx = random.randint(-j, j)
            dy = random.randint(-j, j)
            new_x = max(0, min(crop_x + dx, image_width - self.width))
            new_y = max(0, min(crop_y + dy, image_height - self.height))
            # Make sure at least one keypoint is still inside the new crop;
            # if jitter pushed all of them out, keep the original position.
            kept = any(
                new_x <= kx < new_x + self.width and new_y <= ky < new_y + self.height
                for kx, ky in keypoint_coords
            )
            if kept:
                crop_x, crop_y = new_x, new_y

        return {'crop_x': crop_x, 'crop_y': crop_y}

    def apply_to_keypoints(self, keypoints, crop_x=0, crop_y=0, **params):
        """Apply the crop to a list of keypoints (albumentations v2 API)."""
        result = [self.apply_to_keypoint(kp, crop_x=crop_x, crop_y=crop_y, **params) for kp in keypoints]
        return np.array(result) if result else np.array([])

    # ------------------------------------------------------------------
    # Albumentations v1 backward-compatibility shim.
    # The herdnet conda env ships albumentations 1.0.3, which calls
    # `get_params_dependent_on_targets(params)` — NOT the v2 hook
    # `get_params_dependent_on_data(params, data)`. Without these shims
    # the transform silently became a no-op fixed-position crop on v1,
    # meaning every batch saw the same crop. The shims forward to the
    # v2 implementation so behaviour is identical across versions.
    # ------------------------------------------------------------------
    @property
    def targets_as_params(self) -> List[str]:
        return ['image', 'keypoints']

    def get_params_dependent_on_targets(self, params: Dict) -> Dict:
        return self.get_params_dependent_on_data(params, params)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ('height', 'width', 'min_edge_distance', 'empty_probability',
                'edge_probability', 'edge_zone', 'max_attempts', 'translation_jitter')

# class ObjectAwareRandomCrop(DualTransform):
#     """
#     Random crop that ensures at least one keypoint is included with a minimum distance from edges.
#
#     This transformation selects a random keypoint and positions the crop such that the keypoint
#     is at least `min_edge_distance` pixels away from all crop edges.
#
#     Args:
#         height (int): Height of the crop.
#         width (int): Width of the crop.
#         min_edge_distance (int): Minimum distance in pixels between keypoint and crop edge. Default: 10.
#         empty_probability (float): Probability of creating a crop without any keypoints. Default: 0.0.
#         always_apply (bool): Whether to always apply this transform. Default: False.
#         p (float): Probability of applying the transform. Default: 1.0.
#     """
#
#     def __init__(
#             self,
#             height: int,
#             width: int,
#             min_edge_distance: int = 10,
#             empty_probability: float = 0.0,
#             always_apply: bool = False,
#             p: float = 1.0,
#     ):
#         super().__init__(always_apply, p)
#         self.height = height
#         self.width = width
#         self.min_edge_distance = min_edge_distance
#         self.empty_probability = empty_probability
#
#         if self.min_edge_distance < 0:
#             raise ValueError("min_edge_distance must be non-negative")
#         if not 0.0 <= self.empty_probability <= 1.0:
#             raise ValueError("empty_probability must be between 0.0 and 1.0")
#
#     def _get_valid_crop_range(
#             self,
#             keypoint_x: float,
#             keypoint_y: float,
#             image_height: int,
#             image_width: int
#     ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
#         """
#         Calculate the valid range for crop position to keep keypoint at min_edge_distance from edges.
#
#         Returns:
#             Tuple of (x_range, y_range) where each range is (min, max) inclusive.
#         """
#         # For the keypoint to be at least min_edge_distance from left edge:
#         # keypoint_x - crop_x >= min_edge_distance
#         # crop_x <= keypoint_x - min_edge_distance
#
#         # For the keypoint to be at least min_edge_distance from right edge:
#         # crop_x + width - keypoint_x >= min_edge_distance
#         # crop_x <= keypoint_x - min_edge_distance
#         # crop_x >= keypoint_x - width + min_edge_distance
#
#         crop_x_min = max(0, int(keypoint_x - self.width + self.min_edge_distance))
#         crop_x_max = min(image_width - self.width, int(keypoint_x - self.min_edge_distance))
#
#         crop_y_min = max(0, int(keypoint_y - self.height + self.min_edge_distance))
#         crop_y_max = min(image_height - self.height, int(keypoint_y - self.min_edge_distance))
#
#         return (crop_x_min, crop_x_max), (crop_y_min, crop_y_max)
#
#     def _get_random_crop_with_empty(
#             self,
#             image_height: int,
#             image_width: int
#     ) -> Tuple[int, int]:
#         """Get a random crop position without considering keypoints."""
#         max_crop_x = image_width - self.width
#         max_crop_y = image_height - self.height
#
#         crop_x = random.randint(0, max_crop_x)
#         crop_y = random.randint(0, max_crop_y)
#
#         return crop_x, crop_y
#
#     def _get_crop_with_keypoint(
#             self,
#             keypoint_coords: List[Tuple[float, float]],
#             image_height: int,
#             image_width: int
#     ) -> Tuple[int, int]:
#         """Get a crop position that includes a random keypoint with min edge distance."""
#         # Select a random keypoint
#         target_x, target_y = random.choice(keypoint_coords)
#
#         # Get valid crop ranges
#         (x_min, x_max), (y_min, y_max) = self._get_valid_crop_range(
#             target_x, target_y, image_height, image_width
#         )
#
#         # Check if valid crop is possible
#         if x_min > x_max or y_min > y_max:
#             # Keypoint is too close to image edge, fall back to centering on keypoint
#             crop_x = int(target_x - self.width // 2)
#             crop_y = int(target_y - self.height // 2)
#
#             # Clamp to image bounds
#             crop_x = max(0, min(crop_x, image_width - self.width))
#             crop_y = max(0, min(crop_y, image_height - self.height))
#         else:
#             # Random position within valid range
#             crop_x = random.randint(x_min, x_max)
#             crop_y = random.randint(y_min, y_max)
#
#         return crop_x, crop_y
#
#     def apply(self, img: np.ndarray, crop_x: int = 0, crop_y: int = 0, **params) -> np.ndarray:
#         """Apply the crop to the image."""
#         return img[crop_y:crop_y + self.height, crop_x:crop_x + self.width]
#
#     def apply_to_keypoint(
#             self,
#             keypoint: Tuple[float, float, float, float],
#             crop_x: int = 0,
#             crop_y: int = 0,
#             **params
#     ) -> Tuple[float, float, float, float]:
#         """Apply the crop to keypoints."""
#         x, y, angle, scale = keypoint
#
#         # Adjust keypoint coordinates relative to the crop
#         x_new = x - crop_x
#         y_new = y - crop_y
#
#         return x_new, y_new, angle, scale
#
#     def get_params_dependent_on_targets(self, params: Dict) -> Dict:
#         """Generate parameters for the transformation."""
#         img = params['image']
#         keypoints = params.get('keypoints', [])
#         image_height, image_width = img.shape[:2]
#
#         # Validate crop size
#         if self.height > image_height or self.width > image_width:
#             raise ValueError(
#                 f"Crop size ({self.width}x{self.height}) is larger than "
#                 f"image size ({image_width}x{image_height})"
#             )
#
#         # Check if crop size allows for min_edge_distance
#         if self.height <= 2 * self.min_edge_distance or self.width <= 2 * self.min_edge_distance:
#             raise ValueError(
#                 f"Crop size ({self.width}x{self.height}) is too small for "
#                 f"min_edge_distance={self.min_edge_distance}. "
#                 f"Minimum crop size should be {2 * self.min_edge_distance}x{2 * self.min_edge_distance}"
#             )
#
#         # Extract x,y coordinates from keypoints
#         keypoint_coords = [(kp[0], kp[1]) for kp in keypoints]
#
#         # Decide whether to create empty crop or crop with keypoint
#         create_empty_crop = random.random() < self.empty_probability
#
#         if not keypoint_coords or create_empty_crop:
#             # No keypoints or intentionally empty crop
#             crop_x, crop_y = self._get_random_crop_with_empty(image_height, image_width)
#         else:
#             # Crop with keypoint at min distance from edges
#             crop_x, crop_y = self._get_crop_with_keypoint(
#                 keypoint_coords, image_height, image_width
#             )
#
#         return {'crop_x': crop_x, 'crop_y': crop_y}
#
#     @property
#     def targets_as_params(self) -> List[str]:
#         return ['image', 'keypoints']
#
#     def get_transform_init_args_names(self) -> Tuple[str, ...]:
#         return ('height', 'width', 'min_edge_distance', 'empty_probability')



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
            # edge_black_blobs=False,
            empty_probability=0.0
    ):
        super().__init__(height, width, attempts, always_apply, p, empty_probability)






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

    def get_params_dependent_on_data(self, params, data):
        h, w = data["image"].shape[:2]
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


# import random
# import numpy as np
# from typing import Dict, List, Tuple, Union, Optional
# from albumentations.core.transforms_interface import DualTransform

#
# class ObjectAwareRandomCropV2(DualTransform):
#     """
#     Random crop that guarantees at least one keypoint remains within the cropped area.
#
#     This transformation attempts to find a valid crop position multiple times before falling back
#     to a crop that ensures at least one keypoint is included.
#
#     Args:
#         height (int): Height of the crop.
#         width (int): Width of the crop.
#         attempts (int): Number of random attempts to try before using guaranteed method. Default: 10.
#         always_apply (bool): Whether to always apply this transform. Default: False.
#         p (float): Probability of applying the transform. Default: 1.0.
#     """
#
#     def __init__(
#             self,
#             height: int,
#             width: int,
#             attempts: int = 10,
#             always_apply: bool = False,
#             p: float = 1.0,
#             edge_black_blobs=False,
#             empty_probability=0.0
#     ):
#         super().__init__(always_apply, p)
#         self.last_crop = None
#         self.height = height
#         self.width = width
#         self.attempts = attempts
#         self.empty_probability = empty_probability
#         self.edge_black_blobs = edge_black_blobs
#
#         if self.attempts < 1:
#             raise ValueError("attempts must be at least 1")
#
#     def _has_keypoint_in_crop(
#             self,
#             crop_x: int,
#             crop_y: int,
#             keypoints: List[Tuple[float, float]]
#     ) -> bool:
#         """Check if at least one keypoint is within the crop area."""
#         crop_x_max = crop_x + self.width
#         crop_y_max = crop_y + self.height
#
#         for x, y in keypoints:
#             if crop_x <= x < crop_x_max and crop_y <= y < crop_y_max:
#                 return True
#         return False
#
#     def _get_random_crop_position(
#             self,
#             image_height: int,
#             image_width: int,
#             keypoints: List[Tuple[float, float]],
#             want_keypoint: bool
#     ) -> Optional[Tuple[int, int]]:
#         """
#         Try to find a random crop position that satisfies the keypoint requirement.
#
#         Args:
#             image_height: Height of the source image
#             image_width: Width of the source image
#             keypoints: List of keypoint coordinates
#             want_keypoint: If True, crop should contain keypoints. If False, crop should be empty.
#
#         Returns:
#             (crop_x, crop_y) if valid position found, None otherwise
#         """
#         max_crop_x = image_width - self.width
#         max_crop_y = image_height - self.height
#
#         for _ in range(self.attempts):
#             crop_x = random.randint(0, max_crop_x)
#             crop_y = random.randint(0, max_crop_y)
#
#             has_keypoint = self._has_keypoint_in_crop(crop_x, crop_y, keypoints)
#
#             # Return if we found what we're looking for
#             if has_keypoint == want_keypoint:
#                 return crop_x, crop_y
#
#         return None
#
#     def _get_guaranteed_crop_position(
#             self,
#             image_height: int,
#             image_width: int,
#             keypoints: List[Tuple[float, float]]
#     ) -> Tuple[int, int]:
#         """Get a crop position that guarantees at least one keypoint is included."""
#         if not keypoints:
#             # If no keypoints, just do a random crop
#             max_crop_x = image_width - self.width
#             max_crop_y = image_height - self.height
#             return random.randint(0, max_crop_x), random.randint(0, max_crop_y)
#
#         # Pick a random keypoint to center the crop around
#         target_keypoint = random.choice(keypoints)
#         target_x, target_y = target_keypoint
#
#         # Center the crop on this keypoint
#         crop_x = int(target_x - self.width // 2)
#         crop_y = int(target_y - self.height // 2)
#
#         # Ensure crop stays within image bounds
#         crop_x = max(0, min(crop_x, image_width - self.width))
#         crop_y = max(0, min(crop_y, image_height - self.height))
#
#         return crop_x, crop_y
#
#     def _get_guaranteed_empty_crop_position(
#             self,
#             image_height: int,
#             image_width: int,
#             keypoints: List[Tuple[float, float]]
#     ) -> Tuple[int, int]:
#         """Get a crop position that guarantees NO keypoints are included."""
#         if not keypoints:
#             # If no keypoints, any random crop is empty
#             max_crop_x = image_width - self.width
#             max_crop_y = image_height - self.height
#             return random.randint(0, max_crop_x), random.randint(0, max_crop_y)
#
#         # Try to find areas far from all keypoints
#         max_crop_x = image_width - self.width
#         max_crop_y = image_height - self.height
#
#         best_crop = None
#         max_min_distance = 0
#
#         # Sample multiple random positions and pick the one farthest from keypoints
#         for _ in range(self.attempts * 2):  # More attempts for empty crops
#             crop_x = random.randint(0, max_crop_x)
#             crop_y = random.randint(0, max_crop_y)
#
#             # If this crop has no keypoints, return it immediately
#             if not self._has_keypoint_in_crop(crop_x, crop_y, keypoints):
#                 return crop_x, crop_y
#
#             # Otherwise, calculate minimum distance to any keypoint
#             crop_center_x = crop_x + self.width // 2
#             crop_center_y = crop_y + self.height // 2
#
#             min_distance = min(
#                 np.sqrt((crop_center_x - kx) ** 2 + (crop_center_y - ky) ** 2)
#                 for kx, ky in keypoints
#             )
#
#             if min_distance > max_min_distance:
#                 max_min_distance = min_distance
#                 best_crop = (crop_x, crop_y)
#
#         # Return the crop farthest from keypoints (might still contain some)
#         return best_crop if best_crop else (random.randint(0, max_crop_x), random.randint(0, max_crop_y))
#
#     def apply(self, img: np.ndarray, crop_x: int = 0, crop_y: int = 0, **params) -> np.ndarray:
#         """Apply the crop to the image."""
#         return img[crop_y:crop_y + self.height, crop_x:crop_x + self.width]
#
#     def apply_to_keypoint(self, keypoint: Tuple[float, float, float, float],
#                           crop_x: int = 0,
#                           crop_y: int = 0,
#                           **params) -> Tuple[float, float, float, float]:
#         """Apply the crop to keypoints."""
#         x, y, angle, scale = keypoint
#         x_new = x - crop_x
#         y_new = y - crop_y
#         # Adjust keypoint coordinates relative to the crop
#         eps = 1e-6
#         x_new = max(0.0, min(x_new, self.width - eps))
#         y_new = max(0.0, min(y_new, self.height - eps))
#
#         return x_new, y_new, angle, scale
#
#     def get_params_dependent_on_targets(self, params: Dict) -> Dict:
#         """Generate parameters for the transformation."""
#         img = params['image']
#         keypoints = params.get('keypoints', [])
#         image_height, image_width = img.shape[:2]
#
#         # Decide if we want a crop with keypoints or empty crop
#         want_keypoint = random.uniform(0, 1) > self.empty_probability
#
#         # Validate crop size
#         if self.height > image_height or self.width > image_width:
#             raise ValueError(
#                 f"Crop size ({self.width}x{self.height}) is larger than image size ({image_width}x{image_height})")
#
#         # Extract x,y coordinates from keypoints
#         keypoint_coords = [(kp[0], kp[1]) for kp in keypoints]
#
#         # If no keypoints, just do random crop
#         if not keypoint_coords:
#             max_crop_x = image_width - self.width
#             max_crop_y = image_height - self.height
#             crop_x = random.randint(0, max_crop_x)
#             crop_y = random.randint(0, max_crop_y)
#             self.last_crop = (crop_x, crop_y)
#             return {'crop_x': crop_x, 'crop_y': crop_y}
#
#         # Try to find a random crop position that meets our requirements
#         crop_position = self._get_random_crop_position(
#             image_height, image_width, keypoint_coords, want_keypoint
#         )
#
#         if crop_position is not None:
#             crop_x, crop_y = crop_position
#             self.last_crop = (crop_x, crop_y)
#             return {'crop_x': crop_x, 'crop_y': crop_y}
#
#         # If random attempts failed, use guaranteed method
#         if want_keypoint:
#             crop_x, crop_y = self._get_guaranteed_crop_position(
#                 image_height, image_width, keypoint_coords
#             )
#         else:
#             crop_x, crop_y = self._get_guaranteed_empty_crop_position(
#                 image_height, image_width, keypoint_coords
#             )
#
#         self.last_crop = (crop_x, crop_y)
#         return {'crop_x': crop_x, 'crop_y': crop_y}
#
#     @property
#     def targets_as_params(self) -> List[str]:
#         return ['image', 'keypoints']
#
#     def get_transform_init_args_names(self) -> Tuple[str, ...]:
#         return ('height', 'width', 'attempts')



if __name__ == "__main__":
    pass