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

from pathlib import Path

import torch
import hydra
import animaloc
import wandb
import pandas
import os
import torchvision
from loguru import logger
import albumentations as A

from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from typing import Callable, Optional

from animaloc.models.utils import LossWrapper, load_model
from animaloc.eval import Evaluator, PointsMetrics, Stitcher, BoxesMetrics, ImageLevelMetrics

from animaloc.utils.seed import set_seed
from animaloc.utils.useful_funcs import current_date
from animaloc.vizual.plots import Visualiser


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

            from utils import augmentations as ca
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

    # vizual_fn = None
    # if cfg.training_settings.vizual_fn is not None:
    #     vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]
    visualiser = _define_visualiser(cfg)

    evaluator = animaloc.eval.evaluators.__dict__[name](
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        device_name=cfg.device_name,
        stitcher=stitcher,
        header='[TEST]',
        vizual_fn=visualiser,
        **kwargs
    )

    return evaluator


def _define_visualiser(
        cfg: DictConfig
) -> Visualiser:

    name = cfg.training_settings.visualiser.name
    anno_type = cfg.datasets.anno_type

    assert name in animaloc.vizual.plots.__dict__.keys(), \
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

    visualisor = animaloc.vizual.plots.__dict__[name](
        output_path=cfg.training_settings.visualiser.output_dir
    )

    return visualisor