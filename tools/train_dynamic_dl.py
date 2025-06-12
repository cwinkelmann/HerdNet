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

import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict

import albumentations as A
import hydra
import numpy as np
import pandas
import pandas as pd
import torch
import torchvision
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import Dataset

import animaloc
from animaloc.eval import Evaluator, PointsMetrics, Stitcher, BoxesMetrics, ImageLevelMetrics
from animaloc.utils.seed import set_seed


def visualize_dataset_examples(dataset: Dataset,
                               num_examples: int = 1,
                               show_context: bool = True,
                               force_positive: Optional[bool] = None,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None) -> None:
    """
    Visualize examples from the DynamicCropDataset.

    Args:
        dataset: The DynamicCropDataset instance
        num_examples: Number of examples to display
        show_context: Whether to show the original image context
        force_positive: If True, force positive samples; if False, force negative; if None, use dataset ratio
        figsize: Figure size for matplotlib
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    import numpy as np

    # Calculate grid layout
    if show_context:
        cols = 4  # Original, crop, annotations on original, annotations on crop
        rows = (num_examples + 1) // 2
    else:
        cols = min(4, num_examples)
        rows = (num_examples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Color map for different classes
    colors = plt.cm.Set3(np.linspace(0, 1, 10))

    for i in range(num_examples):
        if i >= len(dataset):
            break

        # Get random index
        idx = random.randint(0, len(dataset) - 1)

        # Get crop based on force_positive parameter
        if force_positive is not None:
            if force_positive:
                crop_img, target = dataset.get_positive_sample(idx)
                sample_type = "Positive"
            else:
                crop_img, target = dataset.get_negative_sample(idx)
                sample_type = "Negative"
        else:
            crop_img, target = dataset[idx]
            # Determine if it's positive or negative
            has_annotations = any(len(target.get(key, [])) > 0 for key in ['x_min', 'x', 'labels'] if key in target)
            sample_type = "Positive" if has_annotations else "Negative"

        if show_context and i < rows:
            # Show original image context and crop location
            row = i

            # Load original image
            img_name = dataset._ordered_img_names[idx]
            full_img = dataset._load_image(idx)

            # Original image
            ax_orig = axes[row, 0]
            ax_orig.imshow(full_img)
            ax_orig.set_title(f"Original: {img_name}")
            ax_orig.axis('off')

            # Crop
            ax_crop = axes[row, 1]
            ax_crop.imshow(crop_img)
            ax_crop.set_title(f"Crop: {sample_type}")
            ax_crop.axis('off')

            # Original with annotations
            ax_orig_ann = axes[row, 2]
            ax_orig_ann.imshow(full_img)
            _draw_annotations_on_axis(ax_orig_ann, dataset.data[dataset.data['images'] == img_name], colors,
                                      'Original')
            ax_orig_ann.set_title("Original + Annotations")
            ax_orig_ann.axis('off')

            # Crop with projected annotations
            ax_crop_ann = axes[row, 3]
            ax_crop_ann.imshow(crop_img)
            _draw_annotations_on_axis(ax_crop_ann, target, colors, 'Crop')

            # Count annotations
            num_anns = sum(len(target.get(key, [])) for key in ['x_min', 'x'] if key in target)
            ax_crop_ann.set_title(f"Crop + Annotations ({num_anns} objects)")
            ax_crop_ann.axis('off')

        else:
            # Simple grid layout
            row = i // cols
            col = i % cols

            ax = axes[row, col] if rows > 1 else axes[col]
            ax.imshow(crop_img)

            # Draw annotations
            _draw_annotations_on_axis(ax, target, colors, 'Crop')

            # Count annotations
            num_anns = sum(len(target.get(key, [])) for key in ['x_min', 'x'] if key in target)
            ax.set_title(f"{sample_type} Sample ({num_anns} objects)")
            ax.axis('off')



    # Remove empty subplots
    total_subplots = rows * cols
    if show_context:
        used_subplots = min(num_examples, rows) * cols
    else:
        used_subplots = min(num_examples, total_subplots)

    for i in range(used_subplots, total_subplots):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def _draw_annotations_on_axis(ax, annotations, colors, coord_type='Crop'):
    """
    Draw annotations on a matplotlib axis.

    Args:
        ax: Matplotlib axis
        annotations: Annotations data (DataFrame or dict)
        colors: Color palette
        coord_type: 'Original' or 'Crop' coordinates
    """
    import matplotlib.patches as patches

    # Handle different annotation formats
    if hasattr(annotations, 'to_dict'):  # DataFrame
        ann_dict = annotations.to_dict('records')
    elif isinstance(annotations, dict):
        # Convert dict format to list of records
        if any(isinstance(v, list) for v in annotations.values()):
            # Dict with lists (target format)
            max_len = max(len(v) if isinstance(v, list) else 1 for v in annotations.values())
            ann_dict = []
            for i in range(max_len):
                record = {}
                for key, value in annotations.items():
                    if isinstance(value, list) and len(value) > i:
                        record[key] = value[i]
                    elif not isinstance(value, list):
                        record[key] = value
                ann_dict.append(record)
        else:
            ann_dict = [annotations]
    else:
        ann_dict = annotations

    color_idx = 0
    for ann in ann_dict:
        color = colors[color_idx % len(colors)]

        # Draw bounding boxes
        if all(key in ann for key in ['x_min', 'y_min', 'x_max', 'y_max']):
            if not (pd.isna(ann['x_min']) or pd.isna(ann['y_min']) or
                    pd.isna(ann['x_max']) or pd.isna(ann['y_max'])):
                width = ann['x_max'] - ann['x_min']
                height = ann['y_max'] - ann['y_min']
                rect = patches.Rectangle(
                    (ann['x_min'], ann['y_min']), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

                # Add label if available
                label = ann.get('labels', f'Object {color_idx}')
                ax.text(ann['x_min'], ann['y_min'] - 5, str(label),
                        color=color, fontsize=8, fontweight='bold')

        # Draw points
        elif all(key in ann for key in ['x', 'y']):
            if not (pd.isna(ann['x']) or pd.isna(ann['y'])):
                ax.plot(ann['x'], ann['y'], 'o', color=color, markersize=8, markeredgewidth=2)

                # Add label if available
                label = ann.get('labels', f'Point {color_idx}')
                ax.text(ann['x'] + 5, ann['y'] - 5, str(label),
                        color=color, fontsize=8, fontweight='bold')

        color_idx += 1


def analyze_dataset_balance(dataset: Dataset,
                            num_samples: int = 1000,
                            show_plot: bool = True) -> Dict:
    """
    Analyze the balance of positive vs negative samples in the dataset.

    Args:
        dataset: The DynamicCropDataset instance
        num_samples: Number of samples to analyze
        show_plot: Whether to show analysis plots

    Returns:
        Dictionary with analysis results
    """
    import matplotlib.pyplot as plt

    positive_count = 0
    negative_count = 0
    annotation_counts = []

    print(f"Analyzing {num_samples} samples...")

    for i in range(min(num_samples, len(dataset) * 10)):  # Allow multiple crops per image
        idx = random.randint(0, len(dataset) - 1)
        try:
            _, target = dataset[idx]

            # Count annotations
            num_anns = sum(len(target.get(key, [])) for key in ['x_min', 'x'] if key in target)
            annotation_counts.append(num_anns)

            if num_anns > 0:
                positive_count += 1
            else:
                negative_count += 1

        except Exception as e:
            print(f"Error analyzing sample {i}: {e}")
            continue

    total_analyzed = positive_count + negative_count
    positive_ratio = positive_count / total_analyzed if total_analyzed > 0 else 0

    results = {
        'total_samples_analyzed': total_analyzed,
        'positive_samples': positive_count,
        'negative_samples': negative_count,
        'positive_ratio': positive_ratio,
        'avg_annotations_per_positive': np.mean([c for c in annotation_counts if c > 0]) if positive_count > 0 else 0,
        'max_annotations': max(annotation_counts) if annotation_counts else 0,
        'min_annotations': min(annotation_counts) if annotation_counts else 0
    }

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart of positive vs negative
        ax1.pie([positive_count, negative_count],
                labels=['Positive', 'Negative'],
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        ax1.set_title('Sample Distribution')

        # Histogram of annotation counts
        ax2.hist(annotation_counts, bins=20, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Number of Annotations per Sample')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Annotation Counts')
        ax2.axvline(np.mean(annotation_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(annotation_counts):.2f}')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    # Print summary
    print(f"\nDataset Balance Analysis:")
    print(f"Total samples analyzed: {total_analyzed}")
    print(f"Positive samples: {positive_count} ({positive_ratio:.1%})")
    print(f"Negative samples: {negative_count} ({1 - positive_ratio:.1%})")
    print(f"Target positive ratio: {dataset.positive_ratio:.1%}")
    print(f"Average annotations per positive sample: {results['avg_annotations_per_positive']:.2f}")

    return results


def compare_sample_types(dataset: Dataset,
                         num_examples: int = 6,
                         figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Compare positive and negative samples side by side.

    Args:
        dataset: The DynamicCropDataset instance
        num_examples: Number of example pairs to show
        figsize: Figure size for matplotlib
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, num_examples, figsize=figsize)
    if num_examples == 1:
        axes = axes.reshape(2, 1)

    colors = plt.cm.Set3(np.linspace(0, 1, 10))

    for i in range(num_examples):
        if i >= len(dataset):
            break

        idx = random.randint(0, len(dataset) - 1)

        try:
            # Get positive sample
            pos_img, pos_target = dataset.get_positive_sample(idx)
            pos_anns = sum(len(pos_target.get(key, [])) for key in ['x_min', 'x'] if key in pos_target)

            # Get negative sample
            neg_img, neg_target = dataset.get_negative_sample(idx)
            neg_anns = sum(len(neg_target.get(key, [])) for key in ['x_min', 'x'] if key in neg_target)

            # Plot positive sample
            axes[0, i].imshow(pos_img)
            _draw_annotations_on_axis(axes[0, i], pos_target, colors, 'Crop')
            axes[0, i].set_title(f"Positive ({pos_anns} objects)")
            axes[0, i].axis('off')

            # Plot negative sample
            axes[1, i].imshow(neg_img)
            _draw_annotations_on_axis(axes[1, i], neg_target, colors, 'Crop')
            axes[1, i].set_title(f"Negative ({neg_anns} objects)")
            axes[1, i].axis('off')

        except Exception as e:
            print(f"Error comparing samples {i}: {e}")
            continue

    plt.suptitle("Positive vs Negative Sample Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()



def _set_species_labels(cls_dict: dict, df: pandas.DataFrame) -> None:
    # FIXME 'species' is not in train_patches.csv
    assert 'species' in df.columns
    cls_dict = dict(map(reversed, cls_dict.items()))
    df['labels'] = df['species'].map(cls_dict)

    # assert none of the labels are None
    assert df['labels'].isnull().any() == False


def _load_albu_transforms(tr_cfg: dict) -> list:
    transforms = []
    for name, kwargs in tr_cfg.items():
        transforms.append(A.__dict__[name](**kwargs))

    return transforms


def _load_end_transforms(tr_cfg: DictConfig) -> Optional[list]:
    if tr_cfg is not None:
        transforms = []
        for name, kwargs in tr_cfg.items():

            if name == 'MultiTransformsWrapper':
                tr_list = []
                for n, k in kwargs.items():
                    tr_list.append(animaloc.data.transforms.__dict__[n](**k))

                transforms.append(animaloc.data.transforms.__dict__[name](tr_list))

            else:
                transforms.append(animaloc.data.transforms.__dict__[name](**kwargs))

        return transforms

    else:
        return None


def _build_sampler(sampler_cfg: DictConfig, dl_kwargs: dict, dataset: Dataset) -> dict:
    dl_kwargs = dl_kwargs.copy()

    sampler = animaloc.data.samplers.__dict__[sampler_cfg.name]
    if sampler_cfg.data_source == 'dataset':
        sampler = sampler(dataset, **dict(sampler_cfg.kwargs))
    else:
        raise NotImplementedError

    if sampler_cfg.batch:
        dl_kwargs.update(dict(batch_size=1, shuffle=False, batch_sampler=sampler))
    else:
        dl_kwargs.update(dict(shuffle=False, sampler=sampler))

    return dl_kwargs


def _get_collate_fn(cfg: DictConfig) -> Callable:
    fn = cfg.datasets.collate_fn
    if fn is not None:
        fn = animaloc.data.batch_utils.__dict__[fn]
    return fn


def _build_model(cfg: DictConfig) -> torch.nn.Module:
    name = cfg.model.name
    from_torchvision = cfg.model.from_torchvision

    if from_torchvision:
        assert name in torchvision.models.__dict__.keys(), \
            f'\'{name}\' unfound in torchvision\'s models'

        model = torchvision.models.__dict__[name]

    else:
        assert name in animaloc.models.__dict__.keys(), \
            f'\'{name}\' class unfound, make sure you have included the class in the models list'

        model = animaloc.models.__dict__[name]

    kwargs = dict(cfg.model.kwargs)
    for k in ['num_classes']:
        kwargs.pop(k, None)

    model = model(**kwargs, num_classes=cfg.datasets.num_classes)

    return model


def _load_losses(cfg: DictConfig) -> tuple:
    criterions = []
    if cfg.losses is not None:
        for loss, args in cfg.losses.items():

            kwargs = {}
            if 'kwargs' in args.keys():
                kwargs = dict(args.kwargs)

                if 'weights' in kwargs.keys():
                    kwargs['weights'] = torch.Tensor(kwargs['weights'])
                elif 'weight' in kwargs.keys():
                    kwargs['weight'] = torch.Tensor(kwargs['weight']).to(torch.device(cfg.device_name))

            crit_dict = {}
            if args.from_torch:
                crit_dict.update({'loss': torch.nn.__dict__[loss](**kwargs)})
            else:
                crit_dict.update({'loss': animaloc.train.losses.__dict__[loss](**kwargs)})

            crit_dict.update({
                'idx': args.output_idx,
                'idy': args.target_idx,
                'lambda': args.lambda_const,
                'name': args.print_name
            })

            criterions.append(crit_dict)

    return criterions


def _define_stitcher(
        model: torch.nn.Module,
        cfg: DictConfig
) -> Stitcher:
    kwargs = dict(cfg.training_settings.stitcher.kwargs)
    for k in ['model', 'size', 'device_name']:
        kwargs.pop(k, None)

    stitcher = animaloc.eval.stitchers.__dict__[cfg.training_settings.stitcher.name](
        model=model,
        size=cfg.datasets.img_size,
        **kwargs,
        device_name=cfg.device_name
    )

    return stitcher


def _define_evaluator(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        cfg: DictConfig
) -> Evaluator:
    name = cfg.training_settings.evaluator.name
    anno_type = cfg.datasets.anno_type

    assert name in animaloc.eval.evaluators.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the evaluators list'

    if anno_type == 'point':
        metrics = PointsMetrics(
            radius=cfg.training_settings.evaluator.threshold,
            num_classes=cfg.datasets.num_classes
        )
    elif anno_type == 'bbox':
        metrics = BoxesMetrics(
            iou=cfg.training_settings.evaluator.threshold,
            num_classes=cfg.datasets.num_classes
        )
    elif anno_type == 'image':
        metrics = ImageLevelMetrics(
            num_classes=cfg.datasets.num_classes
        )
    else:
        raise NotImplementedError

    stitcher = None
    if cfg.training_settings.stitcher is not None:
        stitcher = _define_stitcher(model, cfg)

    kwargs = dict(cfg.training_settings.evaluator.kwargs)
    for k in ['model', 'dataloader', 'metrics', 'device_name', 'stitcher', 'header', 'vizual_fn']:
        kwargs.pop(k, None)

    vizual_fn = None
    if cfg.training_settings.vizual_fn is not None:
        vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]

    evaluator = animaloc.eval.evaluators.__dict__[name](
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        device_name=cfg.device_name,
        stitcher=stitcher,
        header='[TEST]',
        vizual_fn=vizual_fn,
        **kwargs
    )

    return evaluator






@hydra.main(config_path='../configs', config_name="config_2025_06_08_hasty_dynamic")
def main(cfg: DictConfig) -> None:
    work_dir = None
    logger.info(f"Using config: {cfg}")
    # if cfg.work_dir is not None:
    #     work_dir = Path(cfg.work_dir).resolve()
    #     if not work_dir.exists():
    #         work_dir.mkdir(parents=True)

    cfg = cfg.train
    # Set the seed
    logger.info(f'Setting the seed to {cfg.seed}')
    set_seed(cfg.seed)
    current_directory = Path(os.curdir).resolve()
    logger.info(f"current_directory: {current_directory}")
    # Prepare datasets and dataloaders
    logger.info('Building datasets ...')
    device = torch.device(cfg.device_name)

    train_args = cfg.datasets.train
    val_args = cfg.datasets.validate

    train_df = pandas.read_csv(train_args.csv_file)
    # TODO I would argue, modifying the training data while training is not a good idea
    # _set_species_labels(dict(cfg.datasets.class_def), train_df)

    train_dataset = animaloc.datasets.__dict__[train_args.name](
        csv_file=train_df,
        root_dir=train_args.root_dir,
        albu_transforms=_load_albu_transforms(train_args.albu_transforms),
        end_transforms=_load_end_transforms(train_args.end_transforms)
    )

    visualize_dataset_examples(train_dataset, num_examples=1)

    logger.info(f'Number of training samples: {len(train_dataset)}')




if __name__ == '__main__':
    # hydra.initialize(config_path='../configs', job_name="dynamic_hydra")
    # cfg = hydra.compose(config_name="config_2025_02_22_segments")
    # cfg = hydra.compose(config_name="config_FMO03_02_05")
    # main(cfg)

    main()