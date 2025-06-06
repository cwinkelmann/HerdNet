"""
config based inference script which takes the test/herdnets.yaml configuration file, predicts instances, evalustes performances etc

"""


import PIL
import albumentations as A
import hydra
import numpy
import numpy as np
import os
import pandas
import torch
import wandb
from PIL import Image
from loguru import logger
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader

import animaloc
from animaloc.data.transforms import DownSample
from animaloc.eval import PointsMetrics, BoxesMetrics
from animaloc.utils.useful_funcs import current_date, mkdir
from animaloc.vizual import PlotPrecisionRecall, draw_points, draw_text
from tools.inference_test import _set_species_labels, _get_collate_fn, _build_model, _define_evaluator

Image.MAX_IMAGE_PIXELS = None  # Disable the limit




@hydra.main(config_path='../configs', config_name="config_2025_04_14_dla")
def main(cfg: DictConfig, plain_inference = True) -> None:

    # retrieving the test part of the config
    cfg = cfg.test # TODO move this to the other config
    ts = 256  # Thumbnail size

    current_directory = Path(os.getcwd())
    logger.info(f"Current directory: {current_directory}")


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


    # Code for the case of doing just inference
    if plain_inference:
        img_names = [i for i in os.listdir(cfg.dataset.root_dir)
                     if i.endswith(('.JPG', '.jpg', '.JPEG', '.jpeg', ".tiff", ".tif"))]
        n = len(img_names)
        if n == 0:
            raise FileNotFoundError(f"No images found in {cfg.dataset.root_dir}.")
        test_df = pandas.DataFrame(data={'images': img_names, 'x': [0] * n, 'y': [0] * n, 'labels': [1] * n})
        test_df["species"] = "iguana"

    else:
        test_df = pandas.read_csv(cfg.dataset.csv_file)

        _set_species_labels(cls_dict, df=test_df)

    # TODO why is this definehere and the config is not used?
    test_dataset = animaloc.datasets.__dict__[cfg.dataset.name](
        csv_file = test_df,
        root_dir = cfg.dataset.root_dir,
        albu_transforms = [A.Normalize(cfg.dataset.mean, cfg.dataset.std)],
        end_transforms = [DownSample(down_ratio=down_ratio, anno_type=cfg.dataset.anno_type)]
        )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
                                 collate_fn=_get_collate_fn(cfg))
    
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
    logger.info(f'Done with predictions ...')

    # Save results
    plots_path = current_directory / 'plots'
    plots_path.mkdir(exist_ok=True, parents=True)

    logger.info("4) detections")
    detections =  evaluator.detections
    logger.info(f"Num detections: {len(detections)}")
    detections['species'] = detections['labels'].map(cls_dict)

    logger.warning(f"Manually scale up the coordinates by a factor of down_ratio: {down_ratio}")
    detections['x'] = detections['x'] * down_ratio
    detections['y'] = detections['y'] * down_ratio
    detections.to_csv(current_directory / 'detections.csv', index=False)

    # plot only false positves
    # fp = detections[detections['FP'] == 1]

    logger.info("5) plot the detections")
    print('Exporting plots and thumbnails ...')
    dest_plots = plots_path
    mkdir(dest_plots)
    dest_thumb = current_directory / 'thumbnails'
    dest_thumb.mkdir(exist_ok=True, parents=True)
    img_names = numpy.unique(detections['images'].values).tolist()

    for img_name in img_names:
        img = PIL.Image.open(os.path.join(cfg.dataset.root_dir, img_name))
        if img.format != 'JPEG':
            img = img.convert("RGB")


        img_cpy = img.copy()
        pts = list(detections[detections['images'] == img_name][['y', 'x']].to_records(index=False))

        logger.warning(f"The coordinates are manually upscaled by a factor of down_ratio: {down_ratio}")
        pts = [(y, x) for y, x in pts]
        output = draw_points(img, pts, color='red', size=30)
        output.save(os.path.join(dest_plots, img_name), format="JPEG", quality=95)

        # Create and export thumbnails
        sp_score = list(detections[detections['images'] == img_name][['species', 'scores']].to_records(index=False))
        for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score)):
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

    logger.info(f'Testing done, wrote results to: {os.getcwd()}')

if __name__ == '__main__':
    main()