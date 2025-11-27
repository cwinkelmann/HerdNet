import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import random
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class SmartCropDataset(Dataset):
    def __init__(
            self,
            image_dir,
            csv_path,
            crop_size=512,
            object_prob=0.7,
            augmentations=None,
            image_column='image_name',
            x_column='x',
            y_column='y',
            class_column='class',
            image_extension='.jpg'
    ):
        """
        Dataset that performs smart cropping with guaranteed object inclusion probability.

        Args:
            image_dir: Path to directory containing images
            csv_path: Path to CSV file with object annotations
            crop_size: Size of output crops (square)
            object_prob: Probability of including an object in the crop (0.0 to 1.0)
            augmentations: Albumentations transform pipeline
            image_column: Name of image filename column in CSV
            x_column: Name of x coordinate column in CSV
            y_column: Name of y coordinate column in CSV
            class_column: Name of class label column in CSV
            image_extension: Image file extension
        """
        self.image_dir = Path(image_dir)
        self.crop_size = crop_size
        self.object_prob = object_prob
        self.image_extension = image_extension

        # Load annotations
        self.annotations = pd.read_csv(csv_path)
        self.annotations[image_column] = self.annotations[image_column].astype(str)

        # Group annotations by image
        self.image_objects = self.annotations.groupby(image_column).apply(
            lambda x: list(zip(x[x_column], x[y_column], x[class_column]))
        ).to_dict()

        # Get list of all unique images
        self.image_names = list(self.image_objects.keys())

        # Set up augmentations
        if augmentations is None:
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.augmentations = augmentations

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = self.image_dir / f"{image_name}{self.image_extension}"

        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Get objects for this image
        objects = self.image_objects[image_name]

        # Decide whether to do object-centered or random crop
        if random.random() < self.object_prob and objects:
            # Object-centered crop
            crop_info = self._get_object_centered_crop(image, objects, h, w)
        else:
            # Random crop
            crop_info = self._get_random_crop(image, h, w)

        cropped_image, crop_x, crop_y, labels = crop_info

        # Apply augmentations
        augmented = self.augmentations(image=cropped_image)
        final_image = augmented['image']

        return {
            'image': final_image,
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_name': image_name,
            'crop_coords': (crop_x, crop_y)  # Top-left coordinates of crop
        }

    def _get_object_centered_crop(self, image, objects, h, w):
        """Get a crop centered around a randomly selected object."""
        # Randomly select an object
        selected_obj = random.choice(objects)
        obj_x, obj_y, obj_class = selected_obj

        # Calculate crop boundaries centered on object
        half_crop = self.crop_size // 2

        # Add some randomness to the centering (±25% of crop size)
        offset_range = self.crop_size // 4
        x_offset = random.randint(-offset_range, offset_range)
        y_offset = random.randint(-offset_range, offset_range)

        center_x = obj_x + x_offset
        center_y = obj_y + y_offset

        # Calculate crop boundaries
        crop_x1 = max(0, center_x - half_crop)
        crop_y1 = max(0, center_y - half_crop)
        crop_x2 = min(w, crop_x1 + self.crop_size)
        crop_y2 = min(h, crop_y1 + self.crop_size)

        # Adjust if crop is too small due to image boundaries
        if crop_x2 - crop_x1 < self.crop_size:
            crop_x1 = max(0, crop_x2 - self.crop_size)
        if crop_y2 - crop_y1 < self.crop_size:
            crop_y1 = max(0, crop_y2 - self.crop_size)

        # Extract crop
        cropped_image = image[crop_y1:crop_y1 + self.crop_size, crop_x1:crop_x1 + self.crop_size]

        # Find objects within the crop
        labels = self._get_labels_in_crop(objects, crop_x1, crop_y1)

        return cropped_image, crop_x1, crop_y1, labels

    def _get_random_crop(self, image, h, w):
        """Get a random crop from the image."""
        # Random crop coordinates
        max_x = max(0, w - self.crop_size)
        max_y = max(0, h - self.crop_size)
        crop_x1 = random.randint(0, max_x)
        crop_y1 = random.randint(0, max_y)

        # Extract crop
        cropped_image = image[crop_y1:crop_y1 + self.crop_size, crop_x1:crop_x1 + self.crop_size]

        # Find objects within the crop (empty list if none)
        objects = self.image_objects[self.image_names[0]]  # Placeholder, will be overridden
        labels = []  # For random crops, we typically don't guarantee objects

        return cropped_image, crop_x1, crop_y1, labels

    def _get_labels_in_crop(self, objects, crop_x, crop_y):
        """Find all object labels within the crop boundaries."""
        labels = []
        crop_x2 = crop_x + self.crop_size
        crop_y2 = crop_y + self.crop_size

        for obj_x, obj_y, obj_class in objects:
            if crop_x <= obj_x < crop_x2 and crop_y <= obj_y < crop_y2:
                labels.append(obj_class)

        return labels if labels else [0]  # Return background class if no objects




# Example usage and utility functions
def create_augmentation_pipeline():
    """Create a typical augmentation pipeline for training."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.CLAHE(clip_limit=2.0, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """Create a DataLoader with appropriate settings."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = SmartCropDataset(
        image_dir="path/to/images",
        csv_path="path/to/annotations.csv",
        crop_size=512,
        object_prob=0.8,  # 80% chance of including an object
        augmentations=create_augmentation_pipeline()
    )

    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=16)

    # Test the dataset
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Labels: {sample['labels']}")
    print(f"Image name: {sample['image_name']}")
    print(f"Crop coordinates: {sample['crop_coords']}")