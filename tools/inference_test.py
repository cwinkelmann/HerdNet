"""
config based inference script which takes the test/herdnets.yaml configuration file, predicts instances, evalustes performances etc

"""


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

import PIL
import numpy
import torch
import hydra
import wandb
import animaloc
import os
import torchvision
import pandas

import albumentations as A

from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Callable

from animaloc.data.transforms import DownSample
from animaloc.models.utils import load_model, LossWrapper
from animaloc.eval import Evaluator, Metrics, PointsMetrics, BoxesMetrics
from animaloc.eval.stitchers import Stitcher
from loguru import logger
from animaloc.utils.useful_funcs import current_date, mkdir
from animaloc.vizual import PlotPrecisionRecall, draw_points, draw_text


def _set_species_labels(cls_dict: dict, df: pandas.DataFrame) -> None:
    assert 'species' in df.columns
    cls_dict = dict(map(reversed, cls_dict.items()))
    df['labels'] = df['species'].map(cls_dict)

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
    
    model = model(**kwargs, num_classes=cfg.dataset.num_classes)
    model = LossWrapper(model, [])
    model = load_model(model, cfg.model.pth_file)
    return model

def _get_collate_fn(cfg: DictConfig) -> Callable:
    fn = cfg.dataset.collate_fn
    if fn is not None:
        fn = animaloc.data.batch_utils.__dict__[fn]
    return fn

def _define_stitcher(model: torch.nn.Module, cfg: DictConfig) -> Stitcher:

    name = cfg.stitcher.name

    assert name in animaloc.eval.stitchers.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the stitchers list'

    kwargs = dict(cfg.stitcher.kwargs)
    for k in ['model','size','device_name']:
        kwargs.pop(k, None)

    stitcher = animaloc.eval.stitchers.__dict__[name](
        model = model,
        size = cfg.dataset.img_size,
        **kwargs,
        device_name = cfg.device_name
        ) 

    return stitcher

def _define_evaluator(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    metrics: Metrics, 
    cfg: DictConfig
    ) -> Evaluator:

    name = cfg.evaluator.name

    assert name in animaloc.eval.evaluators.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the evaluators list'

    stitcher = None
    if cfg.stitcher is not None:
        stitcher = _define_stitcher(model, cfg)
    
    kwargs = dict(cfg.evaluator.kwargs)
    for k in ['model','dataloader','metrics','device_name','stitcher','header']:
        kwargs.pop(k, None)

    evaluator = animaloc.eval.evaluators.__dict__[name](
        model = model,
        dataloader = dataloader,
        metrics = metrics,
        device_name = cfg.device_name,
        stitcher = stitcher,
        header = '[TEST]',
        **kwargs
    )

    return evaluator


@hydra.main(config_path='../configs', config_name="config")
def main(cfg: DictConfig) -> None:

    cfg = cfg.test

    down_ratio = 1
    if 'down_ratio' in cfg.model.kwargs.keys():
        down_ratio = cfg.model.kwargs.down_ratio

    if cfg.wandb_flag:
        # Set up wandb
        wandb.init(
            project = cfg.wandb_project,
            entity = cfg.wandb_entity,
            config = dict(
                model = cfg.model,
                down_ratio = down_ratio,
                num_classes = cfg.dataset.num_classes,
                threshold = cfg.evaluator.threshold
                )
            )

        date = current_date()
        wandb.run.name = f'{date}_' + cfg.wandb_run + f'_RUN_{wandb.run.id}'

    device = torch.device(cfg.device_name)

    # Prepare dataset and dataloader
    print('Building the test dataset ...')

    cls_dict = dict(cfg.dataset.class_def)
    cls_names = list(cls_dict.values())

    test_df = pandas.read_csv(cfg.dataset.csv_file)
    _set_species_labels(cls_dict, df = test_df)

    test_dataset = animaloc.datasets.__dict__[cfg.dataset.name](
        csv_file = test_df,
        root_dir = cfg.dataset.root_dir,
        albu_transforms = [A.Normalize(cfg.dataset.mean, cfg.dataset.std)],
        end_transforms = [DownSample(down_ratio=down_ratio, anno_type=cfg.dataset.anno_type)]
        )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
        sampler=torch.utils.data.SequentialSampler(test_dataset), collate_fn=_get_collate_fn(cfg))
    
    # Build the trained model
    print('Building the trained model ...')
    model = _build_model(cfg).to(device)

    # Build the evaluator
    print('Preparing for testing ...')
    anno_type = cfg.dataset.anno_type

    if anno_type == 'point':
        metrics = PointsMetrics(radius = cfg.evaluator.threshold, num_classes = cfg.dataset.num_classes)
    elif anno_type == 'bbox':
        metrics = BoxesMetrics(iou = cfg.evaluator.threshold, num_classes = cfg.dataset.num_classes)
    else:
        raise NotImplementedError

    evaluator = _define_evaluator(model, test_dataloader, metrics, cfg)

    # Start testing
    logger.info(f'Starting testing ...')
    out = evaluator.evaluate(wandb_flag=cfg.wandb_flag, viz=False)
    logger.info(f'Done with predictions testing ...')

    # Save results
    print('Saving the results ...')
    
     # 1) PR curves
    plots_path = os.path.join(os.getcwd(), 'plots')
    mkdir(plots_path)
    pr_curve = PlotPrecisionRecall(legend=True)

    metrics = evaluator._stored_metrics
    for c in range(1, metrics.num_classes):
        rec, pre = metrics.rec_pre_lists(c)
        pr_curve.feed(rec, pre, label=cls_dict[c])
    
    pr_curve.save(os.path.join(plots_path, 'precision_recall_curve.png'))
    
    # 2) metrics per class
    res = evaluator.results
    cols = res.columns.tolist()
    str_cls_dict = {str(k): v for k,v in cls_dict.items()}
    str_cls_dict.update({'binary': 'binary'})
    res['species'] = res['class'].map(str_cls_dict)
    res = res[['class', 'species'] + cols[1:]]
    print(res[["species", "precision", "recall", "f1_score", "mae"]])

    res.to_csv(os.path.join(os.getcwd(), 'metrics_results.csv'), index=False)

    # 3) confusion matrix
    cm = pandas.DataFrame(metrics.confusion_matrix, columns=cls_names, index=cls_names)
    cm.to_csv(os.path.join(os.getcwd(), 'confusion_matrix.csv'))
    print(cm)

    # 4) detections
    detections =  evaluator.detections
    logger.info(f"Num detections: {len(detections)}")
    detections['species'] = detections['labels'].map(cls_dict)
    logger.warning(f"Manually scale up the coordinates by a factor of down_ratio: {down_ratio}")
    detections['x'] = detections['x'] * down_ratio
    detections['y'] = detections['y'] * down_ratio
    detections.to_csv(os.path.join(os.getcwd(), 'detections.csv'), index=False)

    # plot only false positves
    # fp = detections[detections['FP'] == 1]

    # 5) plot the detections
    dest = os.getcwd()
    print('Exporting plots and thumbnails ...')
    dest_plots = os.path.join(dest, 'plots')
    mkdir(dest_plots)
    dest_thumb = os.path.join(dest, 'thumbnails')
    mkdir(dest_thumb)
    img_names = numpy.unique(detections['images'].values).tolist()

    for img_name in img_names:
        img = PIL.Image.open(os.path.join(cfg.dataset.root_dir, img_name))

        img_cpy = img.copy()
        pts = list(detections[detections['images'] == img_name][['y', 'x']].to_records(index=False))

        logger.warning(f"The coordinates are manually upscaled by a factor of down_ratio: {down_ratio}")
        pts = [(y, x) for y, x in pts]
        output = draw_points(img, pts, color='red', size=30)
        output.save(os.path.join(dest_plots, img_name), quality=95)

        ts = 256 # Thumbnail size
        # Create and export thumbnails
        sp_score = list(detections[detections['images'] == img_name][['species', 'scores']].to_records(index=False))
        for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score)):
            off = ts // 2
            coords = (x - off, y - off, x + off, y + off)
            thumbnail = img_cpy.crop(coords)
            score = round(score * 100, 0)
            thumbnail = draw_text(thumbnail, f"{sp} | {score}%", position=(10, 5), font_size=int(0.08 * ts))
            thumbnail.save(os.path.join(dest_thumb, img_name[:-4] + f'_{i}.JPG'))

    logger.info(f'Testing done, wrote results to: {os.getcwd()}')

if __name__ == '__main__':
    main()