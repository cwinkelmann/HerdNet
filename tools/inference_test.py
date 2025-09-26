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

from pathlib import Path

import PIL
import numpy
import numpy as np
import pandas as pd
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

from PIL import Image

from tools.train_helper import get_least_occupied_gpu_nvidia_smi, _load_albu_transforms, _load_end_transforms, \
    _define_visualiser, _load_losses

Image.MAX_IMAGE_PIXELS = None  # Disable the limit


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
    
    model = model(**kwargs, num_classes=cfg.datasets.num_classes)
    # criterions = _load_losses(cfg)
    model = LossWrapper(model, [])
    model = load_model(model, cfg.model.load_from)
    return model

def _get_collate_fn(cfg: DictConfig) -> Callable:
    fn = cfg.datasets.collate_fn
    if fn is not None:
        fn = animaloc.data.batch_utils.__dict__[fn]
    return fn

def _define_stitcher(model: torch.nn.Module, cfg: DictConfig) -> Stitcher:

    name = cfg.training_settings.stitcher.name

    assert name in animaloc.eval.stitchers.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the stitchers list'

    kwargs = dict(cfg.training_settings.stitcher.kwargs)
    for k in ['model','size','device_name']:
        kwargs.pop(k, None)

    stitcher = animaloc.eval.stitchers.__dict__[name](
        model = model,
        size = cfg.datasets.img_size,
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

    name = cfg.training_settings.evaluator.name

    assert name in animaloc.eval.evaluators.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the evaluators list'

    stitcher = None
    if cfg.training_settings.stitcher is not None:
        stitcher = _define_stitcher(model, cfg)
    
    kwargs = dict(cfg.training_settings.evaluator.kwargs)
    for k in ['model','dataloader','metrics','device_name','stitcher','header']:
        kwargs.pop(k, None)
        
    visualiser = None
    if cfg.wandb_flag:

        if cfg.training_settings.visualiser is not None:
            visualiser = _define_visualiser(cfg)

    evaluator = animaloc.eval.evaluators.__dict__[name](
        model = model,
        dataloader = dataloader,
        metrics = metrics,
        device_name = cfg.device_name,
        stitcher = stitcher,
        header = '[TEST]',
        vizual_fn = visualiser,
        **kwargs
    )

    return evaluator

# config_name="config_2025_04_14_dla"
# config_name = "config_2025_07_27_iguana_timm_DinoV2"

# config_name = "config_2025_08_08_dinov2_train_val_inverted_val_corrected"
# config_name = "x5_aed_winning_corr_dinoS_pub_train_full_eval_full_aug_all"
# config_name = "x5_aed_winning_corr_dinoL_pub_train_full_eval_full_aug_all"
# config_name = "x5_aed_winning_corr_dla34_pub_train_full_eval_full_aug_all"
 # config_name = "x5_aed_winning_corr_dla102_pub_train_full_eval_full_aug_all"

# config_name = "config_2025_08_08_dinov2_train_val_inverted_val_corrected"
# config_name = "x6_aed_winning_corr_dinoS_pub_train_full_eval_full_aug_all"
# config_name = "x6_aed_winning_corr_dinoL_pub_train_full_eval_full_aug_all"
# config_name = "x6_aed_winning_corr_dla34_pub_train_full_eval_full_aug_all"
# config_name = "x6_aed_winning_corr_dla102_pub_train_full_eval_full_aug_all"

# config_name = "x6_iguana_winning_corr_dinoL_pub_train_full_eval_full_aug_all"
# config_name = "x6_iguana_winning_corr_dinoS_pub_train_full_eval_full_aug_all"
config_name = "x6_iguana_winning_corr_dla34_pub_train_full_eval_full_aug_all"
# config_name = "x6_iguana_winning_corr_dla102_pub_train_full_eval_full_aug_all"

# config_name = "x6_iguana_winning_corr_dinoL_pub_train_full_eval_full_aug_all_ds_fcdm"
# config_name = "x6_iguana_winning_corr_dinoS_pub_train_full_eval_full_aug_all_ds_fcdm"
# config_name = "x6_iguana_winning_corr_dla34_pub_train_full_eval_full_aug_all_ds_fcdm"
# config_name = "x6_iguana_winning_corr_dla102_pub_train_full_eval_full_aug_all_ds_fcdm"


config_path = "../configs/experiment_publication_reproduction/"

@hydra.main(config_path=config_path, config_name=config_name)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the inference test with the given configuration.
    It initializes the model, prepares the dataset, and evaluates the model on the test set.
    """
    logger.info(f"Running inference test with config: {cfg}")
    inference(cfg)


def inference(cfg: DictConfig, plain_inference = False) -> pd.DataFrame:

    # retrieving the test part of the config
     # TODO move this to the other config

    current_directory = Path(os.getcwd())
    logger.info(f"Current directory: {current_directory}")

    # down_ratio = 1

    down_ratio = 4
    if 'down_ratio' in cfg.model.kwargs.keys():
        down_ratio = cfg.model.kwargs.down_ratio



    if cfg.wandb_flag:
        # Set up wandb
        wandb.init(
            project = f"{cfg.wandb_project}_Test",
            entity = cfg.wandb_entity,
            config = dict(
                model = cfg.model,
                down_ratio = down_ratio,
                num_classes = cfg.datasets.num_classes,
                threshold = cfg.training_settings.evaluator.threshold,
                )
            )

        date = current_date()
        wandb.run.name = f'{date}_' + cfg.wandb_run + f'_RUN_{wandb.run.id}'

    if cfg.device_name is None:
        # Automatically select the least occupied GPU
        cfg.device_name = get_least_occupied_gpu_nvidia_smi()
        logger.info(f"Using device: {cfg.device_name}")
    device = torch.device(cfg.device_name)

    # Prepare dataset and dataloader
    logger.info('Building the test dataset ...')

    cls_dict = dict(cfg.datasets.class_def)
    cls_names = list(cls_dict.values())


    # Code for the case of doing just inference
    if plain_inference:
        img_names = [i.name for i in Path(cfg.datasets.test.root_dir).glob("*")
                     if i.name.endswith(('.JPG', '.jpg', '.JPEG', '.jpeg', ".tiff", ".tif")) and not i.name.startswith('.')]


        n = len(img_names)
        if n == 0:
            raise FileNotFoundError(f"No images found in {cfg.datasets.test.root_dir}.")
        test_df = pandas.DataFrame(data={'images': img_names, 'x': [0] * n, 'y': [0] * n, 'labels': [1] * n})
        test_df["species"] = "iguana"
    # load ground truth annotations
    else:
        test_df = pandas.read_csv(cfg.datasets.test.csv_file)
        #test_df = test_df[test_df['species'] == 'iguana_point'].reset_index(drop=True) # FIXME TODO this is a hack becauce too many labels are in the data
        #test_df["species"] = "iguana"
        #_set_species_labels(cls_dict, df=test_df)


    test_dataset = animaloc.datasets.__dict__[cfg.datasets.test.name](
        csv_file=test_df,
        root_dir=cfg.datasets.test.root_dir,
        albu_transforms=_load_albu_transforms(cfg.datasets.test.albu_transforms),
        end_transforms=_load_end_transforms(cfg.datasets.test.end_transforms)
    )
        # TODO figure out how a bigger batch size is possible
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
                                 collate_fn=_get_collate_fn(cfg))
    
    # Build the trained model
    logger.info('Building the trained model ...')
    model = _build_model(cfg).to(device)

    # Build the evaluator
    logger.info('Preparing for testing ...')
    anno_type = cfg.datasets.anno_type

    if anno_type == 'point':
        metrics = PointsMetrics(radius = cfg.training_settings.evaluator.threshold, num_classes = cfg.datasets.num_classes)
    elif anno_type == 'bbox':
        metrics = BoxesMetrics(iou = cfg.training_settings.evaluator.threshold, num_classes = cfg.datasets.num_classes)
    else:
        raise NotImplementedError

    logger.info(f"Define Evaluator: {anno_type}")
    evaluator = _define_evaluator(model, test_dataloader, metrics, cfg)

    # Start testing
    logger.info(f'Starting testing ...')
    out = evaluator.evaluate(wandb_flag=cfg.wandb_flag, viz=True, dont_finish=True)
    logger.info(f'Done with predictions ...')

    # Save results
    plots_path = current_directory / 'plots'
    plots_path.mkdir(exist_ok=True, parents=True)

    if not plain_inference:
        # 1) PR curves

        pr_curve = PlotPrecisionRecall(legend=True)
        logger.info(f"Saving the results ..., plots: {plots_path}")

        metrics = evaluator._stored_metrics
        for c in range(1, metrics.num_classes):
            rec, pre = metrics.rec_pre_lists(c)
            pr_curve.feed(rec, pre, label=cls_dict[c])
        try:
            pr_curve.save(plots_path / 'precision_recall_curve.png')
            wandb.log({"precision_recall_curve": wandb.Image(pr_curve.fig)})
        except IndexError:
            logger.error('Weird index error, skipping PR curve plot')

        logger.info(" 2) metrics per class")
        df_res = evaluator.results
        cols = df_res.columns.tolist()
        str_cls_dict = {str(k): v for k,v in cls_dict.items()}
        str_cls_dict.update({'binary': 'binary'})
        df_res['species'] = df_res['class'].map(str_cls_dict)
        df_res = df_res[['class', 'species'] + cols[1:]]
        logger.info(df_res[["species", "precision", "recall", "f1_score", "mae"]])

        df_res.to_csv(current_directory / 'metrics_results.csv', index=False)

        logger.info(" 3) confusion matrix")
        cm = pandas.DataFrame(metrics.confusion_matrix, columns=cls_names, index=cls_names)
        cm.to_csv(current_directory / 'confusion_matrix.csv')
        logger.info(cm)

    logger.info("4) detections")
    detections =  evaluator.detections
    logger.info(f"Num detections: {len(detections)}")
    detections['species'] = detections['labels'].map(cls_dict)

    logger.warning(f"Manually scale up the coordinates by a factor of down_ratio: {down_ratio}")
    detections['x'] = detections['x'] * down_ratio
    detections['y'] = detections['y'] * down_ratio
    detections.to_csv(current_directory / 'detections.csv', index=False)

    # Method 2: With metadata and description
    artifact = wandb.Artifact(
        name='detections',
        type='predictions',
        description='Model predictions with confidence scores and point coordinates'
    )
    artifact.add_file(current_directory / 'detections.csv')
    artifact.metadata = {
        'model': cfg.model.name,
        'loaded_from': cfg.model.load_from,
        'dataset': cfg.datasets.test.csv_file,
    }
    wandb.log_artifact(artifact)

    # plot only false positves
    # fp = detections[detections['FP'] == 1]

    logger.info("5) plot the detections")
    logger.info('Exporting plots and thumbnails ...')
    dest_plots = plots_path
    mkdir(dest_plots)
    dest_thumb = current_directory / 'thumbnails'
    dest_thumb.mkdir(exist_ok=True, parents=True)
    img_names = numpy.unique(detections['images'].values).tolist()

    for img_name in img_names:
        img = PIL.Image.open(os.path.join(cfg.datasets.test.root_dir, img_name))
        if img.format != 'JPEG':
            img = img.convert("RGB")


        img_cpy = img.copy()
        pts = list(detections[detections['images'] == img_name][['y', 'x']].to_records(index=False))

        # logger.warning(f"The coordinates are manually upscaled by a factor of down_ratio: {down_ratio}")
        pts = [(y, x) for y, x in pts]
        output = draw_points(img, pts, color='red', size=30)
        output.save(os.path.join(dest_plots, img_name), format="JPEG", quality=95)

        ts = 256 # Thumbnail size
        # Create and export thumbnails
        sp_score = list(detections[detections['images'] == img_name][['species', 'scores']].to_records(index=False))
        for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score)):
            if score < 0.3:
                continue
            off = ts // 2
            # TODO the fact this fails if an image is empty shows the code was never evaluated with empty images/or never predicted nothing even if the image was empty
            coords = (x - off, y - off, x + off, y + off)
            if all(np.isnan(coords)):
                logger.warning(f"Coords are all NaN: {coords}, skipping")
                continue
            thumbnail = img_cpy.crop(coords)
            score = round(score * 100, 0)
            thumbnail = draw_text(thumbnail, f"{sp} | {score}%", position=(10, 5), font_size=int(0.08 * ts))
            thumbnail.save(os.path.join(dest_thumb, img_name[:-4] + f'_{i}.JPG'))

            wandb.log({"thumbnails": wandb.Image(thumbnail)})


    logger.info(f'Testing done, wrote results to: {os.getcwd()}')


    wandb.finish()

    return detections

if __name__ == '__main__':
    main()