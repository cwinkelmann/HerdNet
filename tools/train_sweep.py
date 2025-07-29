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
from jedi.settings import cache_directory

import animaloc
import wandb
import pandas
import os
import torchvision
from loguru import logger
import albumentations as A
import omegaconf

from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from typing import Callable, Optional

from animaloc.models.utils import LossWrapper, load_model
from animaloc.eval import Evaluator, PointsMetrics, Stitcher, BoxesMetrics, ImageLevelMetrics

from animaloc.utils.seed import set_seed
from animaloc.utils.useful_funcs import current_date
from tools.train_helper import _get_collate_fn, _build_sampler, _load_albu_transforms, _load_end_transforms, \
    _build_model, _load_losses, _define_evaluator



def main(cfg: DictConfig) -> None:
    work_dir = None
    logger.info(f"Using config: {cfg}")
    # if cfg.work_dir is not None:
    #     work_dir = Path(cfg.work_dir).resolve()
    #     if not work_dir.exists():
    #         work_dir.mkdir(parents=True)

    cfg = cfg
    train_args = cfg.datasets.train
    val_args = cfg.datasets.validate
    # test_args = cfg.datasets.test
    # Set the seed
    logger.info(f'Setting the seed to {cfg.seed}')
    set_seed(cfg.seed)
    current_directory = Path(os.curdir).resolve()
    logger.info(f"current_directory: {current_directory}")
    # Prepare datasets and dataloaders
    logger.info('Building datasets ...')
    device = torch.device(cfg.device_name)

    train_df = pandas.read_csv(train_args.csv_file)
    # TODO I would argue, modifying the training data while training is not a good idea
    # _set_species_labels(dict(cfg.datasets.class_def), train_df)

    train_dataset = animaloc.datasets.__dict__[train_args.name](
        csv_file=train_df,
        root_dir=train_args.root_dir,
        albu_transforms=_load_albu_transforms(train_args.albu_transforms),
        end_transforms=_load_end_transforms(train_args.end_transforms)
    )

    # visualize_dataset_examples(train_dataset, num_examples=8)

    train_dl_kwargs = dict(
        batch_size=cfg.training_settings.batch_size,
        shuffle=True,
        collate_fn=_get_collate_fn(cfg),
        num_workers=cfg.training_settings.num_workers
    )

    if train_args.sampler is not None:
        train_dl_kwargs = _build_sampler(train_args.sampler, dl_kwargs=train_dl_kwargs,
                                         dataset=train_dataset)

    train_dataloader = DataLoader(train_dataset, **train_dl_kwargs)

    if val_args is not None:

        val_df = pandas.read_csv(val_args.csv_file)
        # TODO I would argue modifying data in here is not a good idea. It should immutable
        # _set_species_labels(dict(cfg.datasets.class_def), val_df)

        val_dataset = animaloc.datasets.__dict__[val_args.name](
            csv_file=val_df,
            root_dir=val_args.root_dir,
            albu_transforms=_load_albu_transforms(val_args.albu_transforms),
            end_transforms=_load_end_transforms(val_args.end_transforms)
        )

        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=_get_collate_fn(cfg))
    else:
        val_dataloader = None

    # Set up wandb
    logger.info('Connecting to Weights & Biases ...')
    settings = cfg.training_settings
    losses = cfg.losses
    if losses is not None:
        losses = list(cfg.losses.keys())

    # Disable cache for this run
    os.environ["WANDB_ARTIFACT_CACHE_SIZE"] = "10GB"

    # Or set custom cache location
    os.environ["WANDB_CACHE_DIR"] = "/raid/cwinkelmann/.cache/wandb-cache"

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=dict(
            batch_size=settings.batch_size,
            optimizer=settings.optimizer,
            lr=settings.lr,
            weight_decay=settings.weight_decay,
            warmup_iters=settings.warmup_iters,
            epochs=settings.epochs,
            losses=losses,
            seed=cfg.seed,
            data_augmentation=list(cfg.datasets.train.albu_transforms.keys()),
            n_data_augmentation=len(list(cfg.datasets.train.albu_transforms.keys())),
            input_size=cfg.datasets.img_size,
            class_def=cfg.datasets.class_def,
            **cfg.model.kwargs,
            loss_dict=cfg.losses,
            dataloader={"train": cfg.datasets.train.name, "val": cfg.datasets.validate.name},
            data={"train_csv": cfg.datasets.train.csv_file, "val_csv": cfg.datasets.validate.csv_file},
            training_settings=cfg.training_settings,
            num_training_annotations=len(train_df),
            num_val_annotations=len(val_df),
            num_training_images=train_df.images.nunique(),
            num_val_images=val_df.images.nunique(),
            num_main_images=len(set(train_df['images'].str.replace(r'_x\d+_y\d+\.', '.', regex=True)))

        )
    )

    date = current_date()
    wandb.run.name = f'{date}_' + cfg.wandb_run + f'_{wandb.run.id}'
    wandb.run.tags = [f"train_ds: {cfg.datasets.train.name}", f"val_ds: {cfg.datasets.validate.name}"] + cfg.wandb_tags
    # TODO this is the time to upload metrics about the data

    # Build the model
    logger.info('Building the model ...')
    model = _build_model(cfg)

    # Prepare for training
    logger.info('Preparing for training ...')
    criterions = _load_losses(cfg)
    model = LossWrapper(model, criterions).to(device)

    if cfg.model.load_from is not None:
        model = load_model(model, cfg.model.load_from)

        if 'HerdNet' in cfg.model.name:
            if cfg.model.freeze is not None:
                model.model.freeze(layers=list(cfg.model.freeze))
                logger.info(f"Layers {list(cfg.model.freeze)} freezed")

    if cfg.training_settings.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training_settings.lr,
            weight_decay=cfg.training_settings.weight_decay
        )
    elif cfg.training_settings.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training_settings.lr,
            weight_decay=cfg.training_settings.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training_settings.lr,
            weight_decay=cfg.training_settings.weight_decay
        )

    # Watch the model's gradients during training
    wandb.watch(model)

    if cfg.training_settings.evaluator is not None:

        assert val_dataloader is not None, \
            'A validation dataset must be defined to build an evaluator'

        evaluator = _define_evaluator(model, val_dataloader, cfg)
        select = cfg.training_settings.evaluator.select_mode
        validate_on = cfg.training_settings.evaluator.validate_on

    else:
        # Evaluator ?
        evaluator = None
        validate_on = 'recall'
        select = 'min'

    # Start training & validation
    auto_lr = cfg.training_settings.auto_lr
    if auto_lr:
        auto_lr = dict(cfg.training_settings.auto_lr)

    vizual_fn = None
    if cfg.training_settings.vizual_fn is not None:
        vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]

    trainer = animaloc.train.trainers.__dict__[cfg.training_settings.trainer](
        model,
        train_dataloader,
        optimizer=optimizer,
        num_epochs=cfg.training_settings.epochs,
        auto_lr=auto_lr,
        # adaloss = cfg.training_settings.adaloss,
        val_dataloader=val_dataloader,
        evaluator=evaluator,
        device_name=cfg.device_name,
        vizual_fn=vizual_fn,
        work_dir=work_dir,
        print_freq=cfg.training_settings.print_freq,
        valid_freq=cfg.training_settings.valid_freq,
    )

    if cfg.model.resume_from is not None:
        logger.info(f'Resuming training from \'{cfg.model.resume_from}\' ...')
        trainer.resume(
            pth_path=cfg.model.resume_from,
            select=select,
            validate_on=validate_on,
            load_optim=True,
            wandb_flag=True
        )
    else:
        logger.info('Starting training ...')
        trainer.start(
            cfg.training_settings.warmup_iters,
            select=select,
            validate_on=validate_on,
            wandb_flag=True
        )

    # Add information in .pth files
    for pth_name in ['best_model.pth', 'latest_model.pth']:
        path = current_directory / pth_name
        if not path.exists():
            raise FileNotFoundError(f'\'{pth_name}\' not found in {current_directory}')

        # TODO add this to the training loop somehow
        pth_file = torch.load(path)
        norm_trans = _load_albu_transforms(train_args.albu_transforms)[-1]
        pth_file['classes'] = dict(cfg.datasets.class_def)
        pth_file['mean'] = list(norm_trans.mean)
        pth_file['std'] = list(norm_trans.std)

        torch.save(pth_file, path)
        logger.info(f"Saved Model {pth_name} with added information in {path}")

    wandb.finish()



config_name = "config_2025_07_08_all_detection"


@hydra.main(config_path='../configs', config_name=config_name)
def main_wrapper(cfg: DictConfig):
    """
    Main function to run the training process with hydra configuration.
    """
    run_name_template = cfg.wandb_run


    omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    main(cfg)


if __name__ == '__main__':

    main_wrapper()
