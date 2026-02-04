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
import subprocess
from typing import Callable, Optional

import albumentations as A
import pandas
import torch
import torchvision
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import Dataset

import animaloc
from animaloc.eval import Evaluator, PointsMetrics, Stitcher, BoxesMetrics, ImageLevelMetrics
from animaloc.vizual.plots import Visualiser
from animaloc.eval import DensityAwarePointsMetrics


# from datasets import visualize_dataset_examples



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
        try:
            transforms.append(A.__dict__[name](**kwargs))
        except KeyError as e:

            from animaloc.utils import augmentations as ca
            transforms.append(ca.__dict__.get(name, None)(**kwargs))


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

def _get_show_batch_fn() -> Callable:

    fn = animaloc.data.batch_utils.show_batch
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
        metrics = DensityAwarePointsMetrics(
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

    # vizual_fn = None
    # if cfg.training_settings.vizual_fn is not None:
    #     vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]
    if cfg.training_settings.visualiser is not None:
        visualiser = _define_visualiser(cfg)
    else:
        visualiser = None

    if cfg.training_settings.debug_visualiser is not None:
        debug_visualiser = _define_debug_visualiser(cfg)
    else:
        debug_visualiser = None

    evaluator = animaloc.eval.evaluators.__dict__[name](
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        device_name=cfg.device_name,
        stitcher=stitcher,
        header='[TEST]',
        vizual_fn=visualiser,
        vizual_debug_fn=debug_visualiser,
        **kwargs
    )

    return evaluator


def _define_visualiser(
        cfg: DictConfig
) -> Visualiser:

    name = cfg.training_settings.visualiser.name

    assert name in animaloc.vizual.plots.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the evaluators list'


    visualisor = animaloc.vizual.plots.__dict__[name](
        output_path=cfg.training_settings.visualiser.output_dir,
        down_ratio=cfg.training_settings.visualiser.down_ratio
    )

    return visualisor


def _define_debug_visualiser(
        cfg: DictConfig
) -> Visualiser:

    name = cfg.training_settings.debug_visualiser.name
    assert name in animaloc.vizual.plots.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the evaluators list'


    visualisor = animaloc.vizual.plots.__dict__[name](
        output_path=cfg.training_settings.visualiser.output_dir,
    )

    return visualisor


def get_least_occupied_gpu_nvidia_smi() -> str:
    """
    Get the GPU with the least memory usage using nvidia-smi.
    More accurate as it shows total system memory usage, not just PyTorch.

    Returns:
        int: GPU device ID with least memory usage
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        # Run nvidia-smi to get GPU memory info
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_id = int(parts[0])
                memory_used = int(parts[1])  # MB
                memory_total = int(parts[2])  # MB
                usage_percent = memory_used / memory_total
                gpu_info.append((gpu_id, memory_used, usage_percent))

        # Sort by memory usage and return GPU with least usage
        gpu_info.sort(key=lambda x: x[1])  # Sort by absolute memory used
        import pandas as pd
        df_gpu_usage = pd.DataFrame(gpu_info, columns=['gpu_id', 'memory_used', 'usage'])

        logger.info(f"GPU memory usage: {df_gpu_usage}")

        min_memory = df_gpu_usage['memory_used'].min()
        candidates = df_gpu_usage[df_gpu_usage['memory_used'] <= min_memory + 500]['gpu_id']
        return f"cuda:{random.choice(candidates)}"

        return gpu_info[0][0]

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nvidia-smi: {e}")
        return None