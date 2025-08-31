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

import torch
import os
import PIL
import numpy
import albumentations

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
            augmentation_multiplier: int = 1
    ) -> None:
        '''
        Args:
            csv_file (str): absolute path to the csv file containing
                annotations
            root_dir (str) : path to the images folder
            mask_dir (str, optional): path to the masks folder. If None,
                mask paths in CSV are assumed to be absolute or relative to root_dir
            albu_transforms (list, optional): an albumentations' transformations
                list that takes input sample as entry and returns a transformed
                version. Defaults to None.
            end_transforms (list, optional): list of transformations that takes
                tensor and expected target as input and returns a transformed
                version. These will be applied after albu_transforms. Defaults
                to None.
            augmentation_multiplier (int): How many times to multiply the dataset
                size for augmentation purposes. Defaults to 1.
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

    def _transforms(
            self,
            image: PIL.Image.Image,
            target: dict,
            mask: Optional[PIL.Image.Image] = None
    ) -> Tuple[torch.Tensor, dict]:

        label_fields = target.copy()
        for key in ['annos', 'image_id', 'image_name', 'original_image_name',
                    'augmentation_id', 'mask_path', 'masks']:
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
        img = self._load_image(index)
        target = self._load_target(index)

        mask = None
        if self.anno_type == 'Mask':
            mask = self._load_mask(index)

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