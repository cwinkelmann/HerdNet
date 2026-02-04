__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"

import random

import numpy as np
import torch
import os
import PIL
import numpy
import albumentations
from loguru import logger

from torch.utils.data import Dataset

from typing import Any, Dict, List, Optional, Tuple, Union

from .register import DATASETS

from ..data.annotations import AnnotationsFromCSV
from ..data.transforms import SampleToTensor

from ..data import transforms


def dict_to_tensor(d: dict) -> Tuple[dict, dict]:
    tensor_params = {}
    types = {}

    for k, v in d.items():
        if isinstance(v, bool):
            tensor_params.update({k: v})
        elif isinstance(v, (int, float)):
            tensor_params.update({k: torch.tensor(v, dtype=torch.float64)})
        else:
            tensor_params.update({k: v})

        types.update({k: type(v)})

    return tensor_params, types


def retrieve_num_type(num: torch.Tensor, type: type) -> Union[int, float]:
    assert isinstance(num, torch.Tensor)
    if type == int:
        return int(torch.round(num))
    elif type == float:
        return float(num)


@DATASETS.register()
class CSVDataset(Dataset):
    ''' Class to create a Dataset from a CSV file

    This dataset is built on the basis of CSV files containing box coordinates, in
    [x_min, y_min, x_max, y_max] format, point coordinates in [x,y] format, or
    segmentation mask paths.

    The type of annotations is automatically detected internally. The conditions are:
    - Boxes: ['images', 'x_min', 'y_min', 'x_max', 'y_max', 'labels']
    - Points: ['images', 'x', 'y', 'labels']
    - Masks: ['images', 'mask_path', 'labels'] or ['images', 'masks', 'labels']

    Any additional information (i.e. additional columns) will be associated and returned
    by the dataset.

    If no data augmentation is specified, the dataset returns the image in PIL format
    and the targets as lists. If transforms are specified, the conversion to torch.Tensor
    is done internally, no need to specify this.
    '''

    def __init__(
            self,
            csv_file: str,
            root_dir: str,
            albu_transforms: Optional[list] = None,
            end_transforms: Optional[list] = None,
            augmentation_multiplier: int = 1,
            mosaic_prob: float = 0.0,
            input_size: Tuple[int, int] = (512, 512)

    ) -> None:
        '''
        Args:
            csv_file (str): absolute path to the csv file containing
                annotations
            root_dir (str) : path to the images folder

            albu_transforms (list, optional): an albumentations' transformations
                list that takes input sample as entry and returns a transformed
                version. Defaults to None.
            end_transforms (list, optional): list of transformations that takes
                tensor and expected target as input and returns a transformed
                version. These will be applied after albu_transforms. Defaults
                to None.
            augmentation_multiplier (int): How many times to multiply the dataset
                size for augmentation purposes. Defaults to 1.
            mosaic_prob
            input_size
        '''

        assert isinstance(albu_transforms, (list, type(None))), \
            f'albumentations-transformations must be a list, got {type(albu_transforms)}'

        assert isinstance(end_transforms, (list, type(None))), \
            f'end-transformations must be a list, got {type(end_transforms)}'

        self.csv_file = csv_file
        self.root_dir = root_dir
        self.albu_transforms = albu_transforms
        self.end_transforms = end_transforms
        self.augmentation_multiplier = augmentation_multiplier

        self.mosaic_prob = mosaic_prob
        self.input_size = input_size  # (H, W)

        # store end parameters for adaloss
        self._store_end_params()

        self.annotations = AnnotationsFromCSV(self.csv_file)
        self.data = self.annotations.dataframe

        self.anno_type = self.data.annos[0].atype

        used = set()
        self._img_names = [x for x in self.annotations.images
                           if x not in used and (used.add(x) or True)]

    def _detect_annotation_type(self) -> str:
        """Detect the type of annotations based on CSV columns"""
        columns = set(self.data.columns.tolist())

        if 'mask_path' in columns or 'masks' in columns:
            return 'Mask'
        elif all(col in columns for col in ['x_min', 'y_min', 'x_max', 'y_max']):
            return 'BoundingBox'
        elif all(col in columns for col in ['x', 'y']):
            return 'Point'
        else:
            raise ValueError(f"Could not detect annotation type from columns: {columns}")

    def _load_image(self, index: int) -> PIL.Image.Image:
        # Map the augmented index back to actual image index
        actual_index = index % len(self._img_names)
        img_name = self._img_names[actual_index]
        img_path = os.path.join(self.root_dir, img_name)

        return PIL.Image.open(img_path).convert('RGB')

    def _load_mask(self, index: int) -> PIL.Image.Image:
        """Load segmentation mask for the given index"""
        if self.anno_type != 'Mask':
            raise ValueError("Mask loading only supported for Mask annotation type")

        # Map the augmented index back to actual image index
        actual_index = index % len(self._img_names)
        img_name = self._img_names[actual_index]
        annotations = self.data[self.data['images'] == img_name]

        # Get mask path from CSV
        mask_column = 'mask_path' if 'mask_path' in annotations.columns else 'masks'
        mask_paths = annotations[mask_column].tolist()

        if len(mask_paths) == 0:
            raise ValueError(f"No mask found for image {img_name}")

        # For now, handle single mask per image (could be extended for multiple masks)
        mask_path = mask_paths[0]

        # Determine full mask path
        if self.mask_dir:
            full_mask_path = os.path.join(self.mask_dir, mask_path)
        elif os.path.isabs(mask_path):
            full_mask_path = mask_path
        else:
            full_mask_path = os.path.join(self.root_dir, mask_path)

        # Load mask as grayscale (class indices)
        mask = PIL.Image.open(full_mask_path).convert('L')
        return mask

    def _load_target(self, index: int) -> Dict[str, List[Any]]:
        # Map the augmented index back to actual image index
        actual_index = index % len(self._img_names)
        img_name = self._img_names[actual_index]
        annotations = self.data[self.data['images'] == img_name]
        annotations = annotations.drop(columns='images')

        # Use the augmented index for image_id to maintain uniqueness
        target = {
            'image_id': [index],  # Keep the augmented index for uniqueness
            'image_name': [f"{img_name}_aug_{index}"],  # Add augmentation suffix
            'original_image_name': [img_name],  # Keep original name for reference
            'augmentation_id': [index // len(self._img_names)]  # Which augmentation this is
        }

        for key in annotations.columns:
            target.update({key: list(annotations[key])})

            # convert annotations to tuple for points/boxes
            if key == 'annos' and hasattr(annotations[key].iloc[0], 'get_tuple'):
                target.update({key: [list(a.get_tuple) for a in annotations[key]]})

        return target

    def _load_raw_sample(self, index: int):
        img = self._load_image(index)
        target = self._load_target(index)
        mask = None
        if self.anno_type == 'Mask':
            mask = self._load_mask(index)
        return img, target, mask

    def load_mosaic(self, index: int):
        # 1. Select 3 other random indices
        indices = [index] + [random.randint(0, len(self) - 1) for _ in range(3)]

        H_target, W_target = self.input_size

        # Center point
        xc = int(random.uniform(H_target // 2, 2 * H_target - H_target // 2))
        yc = int(random.uniform(H_target // 2, 2 * H_target - H_target // 2))

        # Canvas
        full_img = np.full((H_target * 2, W_target * 2, 3), 114, dtype=np.uint8)
        full_mask = None
        if self.anno_type == 'Mask':
            full_mask = np.zeros((H_target * 2, W_target * 2), dtype=np.uint8)

        # --- FIX START: Robustly Identify Annotation Keys ---
        _, sample_target, _ = self._load_raw_sample(index)
        n_annos_sample = len(sample_target['annos'])

        # Define keys that are definitely NOT per-annotation data
        # These are per-image metadata keys created in _load_target
        ignored_keys = {
            'annos',
            'image_id',
            'image_name',
            'original_image_name',
            'augmentation_id'
        }

        annotation_keys = []
        for k, v in sample_target.items():
            if k in ignored_keys: continue
            # Only treat as annotation data if it matches length AND is a list
            if isinstance(v, list) and len(v) == n_annos_sample:
                annotation_keys.append(k)

        # Initialize storage
        mosaic_annos = []
        mosaic_extra_data = {k: [] for k in annotation_keys}
        # --------------------------------------------------

        # Base target (keeps image-level metadata from the first image)
        base_target = sample_target.copy()

        for i, idx in enumerate(indices):
            img_pil, target, mask_pil = self._load_raw_sample(idx)
            img = np.array(img_pil)
            h, w = img.shape[:2]

            # Resize if necessary
            if (h, w) != (H_target, W_target):
                img_pil = img_pil.resize((W_target, H_target))
                img = np.array(img_pil)
                if mask_pil is not None:
                    mask_pil = mask_pil.resize((W_target, H_target), resample=PIL.Image.NEAREST)

                scale_x = W_target / w
                scale_y = H_target / h

                # Scale annotations
                new_annos = []
                for anno in target['annos']:
                    if self.anno_type == 'Point':
                        new_annos.append([anno[0] * scale_x, anno[1] * scale_y])
                    elif self.anno_type == 'BoundingBox':
                        new_annos.append([
                            anno[0] * scale_x, anno[1] * scale_y,
                            anno[2] * scale_x, anno[3] * scale_y
                        ])
                target['annos'] = new_annos
                h, w = H_target, W_target

            # Place image in quadrant
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, W_target * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, H_target * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, W_target * 2), min(yc + h, H_target * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            full_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            if full_mask is not None and mask_pil is not None:
                mask_np = np.array(mask_pil)
                full_mask[y1a:y2a, x1a:x2a] = mask_np[y1b:y2b, x1b:x2b]

            padw = x1a - x1b
            padh = y1a - y1b

            # Loop through annos AND corresponding extra data
            for j, anno in enumerate(target['annos']):
                keep = False
                new_coords = None

                if self.anno_type == 'Point':
                    nx = anno[0] + padw
                    ny = anno[1] + padh
                    if 0 <= nx < W_target * 2 and 0 <= ny < H_target * 2:
                        new_coords = [nx, ny]
                        keep = True

                elif self.anno_type == 'BoundingBox':
                    nx1 = np.clip(anno[0] + padw, 0, W_target * 2)
                    ny1 = np.clip(anno[1] + padh, 0, H_target * 2)
                    nx2 = np.clip(anno[2] + padw, 0, W_target * 2)
                    ny2 = np.clip(anno[3] + padh, 0, H_target * 2)
                    if (nx2 - nx1) > 1 and (ny2 - ny1) > 1:
                        new_coords = [nx1, ny1, nx2, ny2]
                        keep = True

                if keep:
                    mosaic_annos.append(new_coords)
                    # Safe append: we now know 'key' is truly an annotation list
                    for key in annotation_keys:
                        mosaic_extra_data[key].append(target[key][j])

        # 4. Final Random Crop
        crop_x = max(0, xc - W_target // 2)
        crop_y = max(0, yc - H_target // 2)
        if crop_x + W_target > W_target * 2: crop_x = W_target * 2 - W_target
        if crop_y + H_target > H_target * 2: crop_y = H_target * 2 - H_target

        final_img = full_img[crop_y: crop_y + H_target, crop_x: crop_x + W_target]

        final_mask = None
        if full_mask is not None:
            final_mask = full_mask[crop_y: crop_y + H_target, crop_x: crop_x + W_target]
            final_mask = PIL.Image.fromarray(final_mask)

        # Adjust annotations to the final crop
        final_annos = []
        final_extra_data = {k: [] for k in annotation_keys}

        for i, anno in enumerate(mosaic_annos):
            keep = False
            new_coords = None

            if self.anno_type == 'Point':
                nx = anno[0] - crop_x
                ny = anno[1] - crop_y
                if 0 <= nx < W_target and 0 <= ny < H_target:
                    new_coords = [nx, ny]
                    keep = True

            elif self.anno_type == 'BoundingBox':
                nx1 = np.clip(anno[0] - crop_x, 0, W_target)
                ny1 = np.clip(anno[1] - crop_y, 0, H_target)
                nx2 = np.clip(anno[2] - crop_x, 0, W_target)
                ny2 = np.clip(anno[3] - crop_y, 0, H_target)
                if (nx2 - nx1) > 1 and (ny2 - ny1) > 1:
                    new_coords = [nx1, ny1, nx2, ny2]
                    keep = True

            if keep:
                final_annos.append(new_coords)
                for key in annotation_keys:
                    final_extra_data[key].append(mosaic_extra_data[key][i])

        final_img = PIL.Image.fromarray(final_img)

        base_target['annos'] = final_annos
        for key in annotation_keys:
            base_target[key] = final_extra_data[key]

        return final_img, base_target, final_mask

    def _transforms(
            self,
            image: PIL.Image.Image,
            target: dict,
            mask: Optional[PIL.Image.Image] = None
    ) -> Tuple[torch.Tensor, dict]:

        label_fields = target.copy()
        for key in ['annos', 'image_id', 'image_name', 'original_image_name',
                    'augmentation_id']:
            label_fields.pop(key, None)  # Use pop with default to avoid KeyError

        if self.albu_transforms:

            # Segmentation Masks
            if self.anno_type == 'Mask':
                if mask is None:
                    raise ValueError("Mask is required for Mask annotation type")

                transform_pipeline = albumentations.Compose(
                    self.albu_transforms,
                    additional_targets={'mask': 'mask'}
                )

                transformed = transform_pipeline(
                    image=numpy.array(image),
                    mask=numpy.array(mask),
                    **label_fields
                )

                tr_image = numpy.asarray(transformed['image'])
                tr_mask = numpy.asarray(transformed['mask'])

                # Remove image and mask from transformed dict
                transformed.pop('image')
                transformed.pop('mask')

                # Add mask to target
                transformed['masks'] = tr_mask

                # Preserve metadata
                for key in ['image_id', 'image_name', 'original_image_name', 'augmentation_id']:
                    if key in target:
                        transformed[key] = target[key]

                tr_image, tr_target = SampleToTensor()(tr_image, transformed, 'mask')

                if self.end_transforms is not None:
                    for trans in self.end_transforms:
                        tr_image, tr_target = trans(tr_image, tr_target)

                return tr_image, tr_target

            # Bounding boxes
            elif self.anno_type == 'BoundingBox':
                transform_pipeline = albumentations.Compose(
                    self.albu_transforms,
                    bbox_params=albumentations.BboxParams(
                        format='pascal_voc',
                        label_fields=list(label_fields.keys())
                    )
                )

                transformed = transform_pipeline(
                    image=numpy.array(image),
                    bboxes=target['annos'],
                    **label_fields
                )

                tr_image = numpy.asarray(transformed['image'])
                transformed.pop('image')

                transformed['boxes'] = transformed['bboxes']
                transformed.pop('bboxes')

                for key in ['image_id', 'image_name', 'original_image_name', 'augmentation_id']:
                    if key in target:
                        transformed[key] = target[key]

                tr_image, tr_target = SampleToTensor()(tr_image, transformed)

                if self.end_transforms is not None:
                    for trans in self.end_transforms:
                        tr_image, tr_target = trans(tr_image, tr_target)

                return tr_image, tr_target

            # Points
            elif self.anno_type == 'Point':
                transform_pipeline = albumentations.Compose(
                    self.albu_transforms,
                    keypoint_params=albumentations.KeypointParams(
                        format='xy',
                        label_fields=list(label_fields.keys())
                    )
                )



                transformed = transform_pipeline(
                    image=numpy.array(image),
                    keypoints=target['annos'],
                    **label_fields
                )

                tr_image = numpy.asarray(transformed['image'])
                transformed.pop('image')

                transformed['points'] = transformed['keypoints']
                transformed.pop('keypoints')

                for key in ['image_id', 'image_name', 'original_image_name', 'augmentation_id']:
                    if key in target:
                        transformed[key] = target[key]

                tr_image, tr_target = SampleToTensor()(tr_image, transformed, 'point')

                if self.end_transforms is not None:
                    for trans in self.end_transforms:
                        tr_image, tr_target = trans(tr_image, tr_target)

                return tr_image, tr_target

        else:
            # No transforms case
            if self.anno_type == 'Mask' and mask is not None:
                target['masks'] = numpy.array(mask)
            return image, target

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        # Logic: If training (implied by mosaic_prob > 0), try mosaic
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            img, target, mask = self.load_mosaic(index)
        else:
            img, target, mask = self._load_raw_sample(index)

        tr_img, tr_target = self._transforms(img, target, mask)

        return tr_img, tr_target

    def load_end_param(self, end_param: str, value: float) -> None:
        self.end_params[end_param] = value

    def update_end_transforms(self) -> None:
        new_transforms = []
        up_params = self._update_end_params()
        for trans, params in zip(self.end_transforms, up_params):
            name = type(trans).__name__
            new_transforms.append(transforms.__dict__[name](**params))

        self.end_transforms = new_transforms
        self._store_end_params()

    def __len__(self) -> int:
        return len(self._img_names) * self.augmentation_multiplier

    def get_actual_length(self) -> int:
        """Returns the actual number of unique images"""
        return len(self._img_names)

    def get_augmentation_info(self, index: int) -> Dict[str, Any]:
        """Get information about which augmentation this index represents"""
        actual_index = index % len(self._img_names)
        augmentation_id = index // len(self._img_names)
        return {
            'actual_image_index': actual_index,
            'augmentation_id': augmentation_id,
            'original_image_name': self._img_names[actual_index]
        }

    def _store_end_params(self) -> None:
        self.end_params = {}
        self._end_params_types = []

        if self.end_transforms is not None:
            for trans in self.end_transforms:
                tensor_params, types = dict_to_tensor(trans.__dict__)
                self._end_params_types.append(types)
                self.end_params.update(tensor_params)

    def _update_end_params(self) -> list:
        up_params = []
        for trans in self._end_params_types:
            up_dict = {}
            for k, v in trans.items():
                up_num = self.end_params[k]
                if isinstance(self.end_params[k], torch.Tensor):
                    up_num = retrieve_num_type(self.end_params[k], v)

                up_dict.update({k: up_num})

            up_params.append(up_dict)

        return up_params