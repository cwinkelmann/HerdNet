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
import shutil

from pathlib import Path
from typing import List, Tuple, Any

import hydra
import pandas
import torch
from loguru import logger

try:
    import wandb
except ImportError:
    wandb = None
from matplotlib import pyplot as plt
from omegaconf import DictConfig, omegaconf
from torch.utils.data import DataLoader

import animaloc
from animaloc.models.utils import LossWrapper, load_model
from animaloc.utils.seed import set_seed
from animaloc.utils.useful_funcs import current_date
from animaloc.utils.train_helper import _get_collate_fn, _build_sampler, _load_albu_transforms, _load_end_transforms, \
    _build_model, _load_losses, _define_evaluator, _define_visualiser, get_least_occupied_gpu_nvidia_smi, \
    _get_show_batch_fn, _define_debug_visualiser
from animaloc.vizual.custom_vis import plot_heatmaps, plot_heatmaps_combined

config_path = '../configs/reference_data/delplanque2022'
config_name = 'dla34_custom_publication'

def _setup_file_logging(log_dir: Path) -> int:
    """Add a loguru file sink. Returns the sink ID for cleanup."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date = current_date()
    log_path = log_dir / f"{date}_training.log"
    sink_id = logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="100 MB",
    )
    logger.info(f"Logging to {log_path}")
    return sink_id


def main(cfg: DictConfig) -> Path:
    work_dir = None
    current_directory = Path(os.curdir).resolve()

    # Set up file logging in the Hydra output dir (or cwd)
    log_sink_id = _setup_file_logging(current_directory)

    logger.info(f"Using config: {cfg}")

    train_args = cfg.datasets.train
    val_args = cfg.datasets.validate

    logger.info(f'Setting the seed to {cfg.seed}')
    set_seed(cfg.seed)
    logger.info(f"current_directory: {current_directory}")
    # Prepare datasets and dataloaders
    logger.info('Building datasets ...')

    if cfg.device_name is None:
        # Automatically select the least occupied GPU
        cfg.device_name = get_least_occupied_gpu_nvidia_smi()
        logger.info(f"Using device: {cfg.device_name}")
    device = torch.device(cfg.device_name)

    train_df = pandas.read_csv(train_args.csv_file)
    # TODO I would argue, modifying the training data while training is not a good idea
    # _set_species_labels(dict(cfg.datasets.class_def), train_df)

    train_dataset = animaloc.datasets.__dict__[train_args.name](
        csv_file=train_df,
        root_dir=train_args.root_dir,
        albu_transforms=_load_albu_transforms(train_args.albu_transforms),
        end_transforms=_load_end_transforms(train_args.end_transforms),
        augmentation_multiplier=train_args.augmentation_multiplier,
        # mosaic_prob = train_args.mosaic_prob , # check if it is in there
    )

    val_dataloader = None
    val_loss_dataloader = None
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
        val_loss_dataset = animaloc.datasets.__dict__[val_args.name](
            csv_file=val_df,
            root_dir=val_args.root_dir,
            albu_transforms=_load_albu_transforms(val_args.albu_transforms),
            end_transforms=_load_end_transforms(train_args.end_transforms)
        )
        # TODO why is it batch size 1 here: because the the datasets keeps track of metadata image_name etc. we get a wrong RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 27
        if hasattr(cfg.training_settings, 'val_batch_size') and cfg.training_settings.val_batch_size is not None:
            val_batch_size = cfg.training_settings.val_batch_size
        else:
            val_batch_size = 1

        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size,
                                    shuffle=False,
                                    collate_fn=_get_collate_fn(cfg))

        if hasattr(cfg.training_settings, 'loss_evaluation') and cfg.training_settings.loss_evaluation is not None:
            val_loss_dataloader = DataLoader(val_loss_dataset,
                                             batch_size=cfg.training_settings.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.training_settings.num_workers // 2
                                             )

    # Set up wandb
    logger.info('Connecting to Weights & Biases ...')
    settings = cfg.training_settings
    losses = cfg.losses
    if losses is not None:
        losses = list(cfg.losses.keys())

    # Disable cache for this run
    os.environ.setdefault("WANDB_ARTIFACT_CACHE_SIZE", "10GB")
    date = current_date()

    if cfg.wandb_flag and wandb is None:
        logger.warning("wandb_flag is True but wandb is not installed. Install with: pip install -e '.[tracking]'")
        cfg.wandb_flag = False

    if cfg.wandb_flag:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=dict(
                model_name=cfg.model.name,
                model_load=cfg.model.load_from,
                batch_size=settings.batch_size,
                optimizer=settings.optimizer,
                lr=settings.lr,
                weight_decay=settings.weight_decay,
                warmup_iters=settings.warmup_iters,
                epochs=settings.epochs,
                losses=losses,
                seed=cfg.seed,
                train_data_augmentation=list(cfg.datasets.train.albu_transforms.keys()),
                train_data_augmentation_v=dict(cfg.datasets.train.albu_transforms),
                validate_data_augmentation_v=dict(cfg.datasets.validate.albu_transforms),

                n_data_augmentation=len(list(cfg.datasets.train.albu_transforms.keys())),
                # end_transforms=list(cfg.datasets.train.end_transforms.keys()),
                # FIDT=cfg.datasets.train.end_transforms.MultiTransformsWrapper.FIDT, # TODO get this right for the comparison
                # PointsToMask=dict(cfg.datasets.train.end_transforms.MultiTransformsWrapper.PointsToMask),
                input_size=cfg.datasets.img_size,
                class_def=cfg.datasets.class_def,
                augmentation_multiplier=cfg.datasets.train.augmentation_multiplier,
                model=dict(cfg.model.kwargs),
                model_backbone=cfg.model.kwargs.backbone if "backbone" in cfg.model.kwargs else None,
                # TODO fix it when it works
                loss_dict=dict(cfg.losses),
                base_model=cfg.model.load_from,
                dataloader={"train": cfg.datasets.train.name, "val": cfg.datasets.validate.name},
                data={"train_csv": cfg.datasets.train.csv_file, 'train_dir': cfg.datasets.train.root_dir,
                      "val_csv": cfg.datasets.validate.csv_file, 'val_dir': cfg.datasets.validate.root_dir, },
                training_settings=dict(cfg.training_settings),
                training_auto_lr=dict(cfg.training_settings.auto_lr),

                num_training_annotations=len(train_df),
                num_val_annotations=len(val_df),
                num_training_images=train_df.images.nunique(),
                num_val_images=val_df.images.nunique(),
                num_main_images=len(set(train_df['images'].str.replace(r'_x\d+_y\d+\.', '.', regex=True))),

                evaluator_name=cfg.training_settings.evaluator.name if cfg.training_settings.evaluator is not None else None,
                evaluator_kwargs=cfg.training_settings.evaluator.kwargs if cfg.training_settings.evaluator is not None else None,
                evaluator=cfg.training_settings.evaluator,
                stitcher=cfg.training_settings.stitcher if cfg.training_settings.stitcher is not None else None,
                date=date,

            )
        )

        wandb.run.name = f'{date}_' + cfg.wandb_run + f'_{wandb.run.id}'
        # wandb.run.name = f'{cfg.wandb_run}'
        wandb.run.tags = [f"train_ds: {cfg.datasets.train.name}",
                          f"val_ds: {cfg.datasets.validate.name}"] + cfg.wandb_tags
        # TODO this is the time to upload metrics about the data
        wandb.run.notes = cfg.wandb_notes if cfg.wandb_notes is not None else ""

    train_dl_kwargs = dict(
        batch_size=cfg.training_settings.batch_size,
        shuffle=True,
        collate_fn=_get_collate_fn(cfg),
        num_workers=cfg.training_settings.num_workers
    )

    if train_args.sampler is not None:
        train_dl_kwargs = _build_sampler(train_args.sampler, dl_kwargs=train_dl_kwargs,
                                         dataset=train_dataset)

    # little hack to visuliase training data examples
    train_dataloader = DataLoader(train_dataset, **train_dl_kwargs)

    if cfg.wandb_flag and cfg.model.name != "HerdNetP2P":  # TODO visualise the training data
        # iterate through the dataloader to check if it works
        max_plot = 10
        for i, (img_tensor, target) in enumerate(train_dataset):
            if i >= max_plot:
                break

            heatmap = target[0].squeeze(0)
            cls_map = target[1]
            cfg.datasets.num_classes

            if hasattr(cfg.training_settings, 'visualiser'):
                if cfg.training_settings.visualiser is not None and cfg.training_settings.visualiser.output_dir is not None:
                    output_dir = Path(cfg.training_settings.visualiser.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # fig, axes = plot_heatmaps(img_tensor.squeeze(0),
                    #                           heatmap,
                    #                           show_argmax_overlay=False,
                    #                           max_channels=cfg.datasets.num_classes-2)
                    #
                    # wandb.log({f'augmented_dataset_examples': wandb.Image(fig)})
                    # fig.savefig((output_dir /  f'augmented_dataset_example_{i}.png'))

                    fig, axes = plot_heatmaps_combined(img_tensor.squeeze(0),
                                                       heatmap,
                                                       )

                    wandb.log({f'augmented_dataset_examples': wandb.Image(fig)})
                    fig.savefig((output_dir / f'augmented_dataset_example_{i}.png'))

                    plt.close(fig)

    # Build the model
    logger.info('Building the model ...')
    model = _build_model(cfg)
    try:
        model.check_trainable_parameters()  # TODO implement this in all models
    except AttributeError as e:
        logger.error(f"The model has not check for trainable_parameters: {e}")

    model.reshape_classes(num_classes=cfg.datasets.num_classes)
    logger.info('Preparing for training ...')

    # Standard behavior for Density Models: Load external losses and wrap.
    criterions = _load_losses(cfg)
    final_model = LossWrapper(model, criterions).to(device)

    if cfg.model.load_from is not None:
        final_model = load_model(final_model, cfg.model.load_from, device=device)
        # After loading, reshape again to reinitialize any classification
        # head layers that were skipped due to shape mismatch.
        final_model.model.reshape_classes(num_classes=cfg.datasets.num_classes)

        # if 'HerdNet' in cfg.model.name:
        #     if cfg.model.freeze is not None and cfg.model.freeze > 0:
        #         model.model.freeze(layers=list(cfg.model.freeze))
        #         logger.info(f"Layers {list(cfg.model.freeze)} freezed")

    # if hasattr(cfg.model, 'freeze_backbone') and cfg.model.freeze_backbone:
    #     model.model.freeze_backbone_completely()
    #     logger.info("Backbone frozen")

    try:
        final_model.model.check_trainable_parameters()  # TODO implement this in all models
    except AttributeError as e:
        logger.error(f"The model has not check for trainable_parameters: {e}")

    if cfg.training_settings.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            final_model.parameters(),
            lr=cfg.training_settings.lr,
            weight_decay=cfg.training_settings.weight_decay
        )
    # # 2. Define the Optimizer with Groups
    # elif cfg.training_settings.optimizer == 'adamW' and cfg.training_settings.backbone_lr is not None:
    #
    #     optimizer = torch.optim.AdamW([
    #     {
    #         'params': model.model.loc_head.parameters(),
    #         'lr': cfg.training_settings.lr,  # HIGH (Learn fast)
    #         'weight_decay': cfg.training_settings.weight_decay  # Standard weight decay
    #     },
    #     # Check if classification head exists and add it too
    #     {
    #         'params': model.model.cls_head.parameters(),
    #         'lr': cfg.training_settings.lr,
    #         'weight_decay': cfg.training_settings.weight_decay
    #     },
    #     {
    #         'params': model.model.backbone.parameters(),
    #         'lr': cfg.training_settings.backbone_lr,  # VERY LOW (Keep pre-trained knowledge)
    #         'weight_decay': cfg.training_settings.weight_decay_backbone  # ViTs like high weight decay
    #     },
    #
    #     ])
    elif cfg.training_settings.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(
            final_model.parameters(),
            lr=cfg.training_settings.lr,
            weight_decay=cfg.training_settings.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            final_model.parameters(),
            lr=cfg.training_settings.lr,
            weight_decay=cfg.training_settings.weight_decay
        )

    # Watch the model's gradients during training
    visualiser = None
    debug_visualiser = None
    if cfg.wandb_flag:

        # wandb.watch(model) # TODO make this configurable

        if cfg.training_settings.visualiser is not None:
            visualiser = _define_visualiser(cfg)
        if cfg.training_settings.debug_visualiser is not None:
            debug_visualiser = _define_debug_visualiser(cfg)
            # TODO it is actually used inthe evaluators
    if cfg.training_settings.evaluator is not None:

        assert val_dataloader is not None, \
            'A validation dataset must be defined to build an evaluator'

        evaluator = _define_evaluator(final_model, val_dataloader, cfg)
        select = cfg.training_settings.evaluator.select_mode
        validate_on = cfg.training_settings.evaluator.validate_on
    else:
        evaluator = None
        select = cfg.training_settings.loss_evaluation.select_mode
        validate_on = cfg.training_settings.loss_evaluation.validate_on
        val_dataloader = None

    # Start training & validation
    auto_lr = cfg.training_settings.auto_lr
    if auto_lr:
        auto_lr = dict(cfg.training_settings.auto_lr)

    vizual_fn = None
    if cfg.training_settings.vizual_fn is not None:
        vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]

    trainer = animaloc.train.trainers.__dict__[cfg.training_settings.trainer](
        final_model,
        train_dataloader,
        optimizer=optimizer,
        num_epochs=cfg.training_settings.epochs,
        auto_lr=auto_lr,
        # adaloss = cfg.training_settings.adaloss,
        val_dataloader=val_dataloader,
        val_loss_dataloader=val_loss_dataloader,
        evaluator=evaluator,
        device_name=cfg.device_name,
        vizual_fn=visualiser,
        debug_vizual_fn=debug_visualiser,
        work_dir=work_dir,
        print_freq=cfg.training_settings.print_freq,
        valid_freq=cfg.training_settings.valid_freq,
        early_stopping=cfg.training_settings.early_stopping if hasattr(cfg.training_settings,
                                                                       "early_stopping") else False,
        patience=cfg.training_settings.early_stopping_patience if hasattr(cfg.training_settings,
                                                                          "early_stopping_patience") else False,
    )

    if cfg.model.resume_from is not None:
        logger.info(f'Resuming training from \'{cfg.model.resume_from}\' ...')
        trainer.resume(
            pth_path=cfg.model.resume_from,
            select=select,
            validate_on=validate_on,
            load_optim=True,
            wandb_flag=cfg.wandb_flag
        )
    else:
        logger.info('Starting training ...')
        trainer.start(
            warmup_iters=cfg.training_settings.warmup_iters,
            select=select,
            validate_on=validate_on,
            wandb_flag=cfg.wandb_flag
        )

    try:
        # Add information in .pth files
        for pth_name in ['best_model.pth', 'latest_model.pth']:
            path = current_directory / pth_name
            if not path.exists():
                raise FileNotFoundError(f'\'{pth_name}\' not found in {current_directory}')

            pth_file = torch.load(path, weights_only=False)
            norm_trans = _load_albu_transforms(train_args.albu_transforms)[-1]
            pth_file['classes'] = dict(cfg.datasets.class_def)
            pth_file['mean'] = list(norm_trans.mean)
            pth_file['std'] = list(norm_trans.std)
            pth_file['config'] = cfg # TODO add the config into the pth

            torch.save(pth_file, path)
            logger.info(f"Saved Model {pth_name} with added information in {path}")
    except Exception as e:
        logger.error(f"Could not add information to pth files: {e}")

    if cfg.wandb_flag:
        wandb.finish()

    logger.info(f"Training complete. Output in {current_directory}")
    logger.remove(log_sink_id)

    return current_directory




@hydra.main(config_path=config_path, config_name=config_name, version_base="1.1")
def main_wrapper(cfg: DictConfig = None):
    """
    Main function to run the training process with hydra configuration.
    """

    # Convert the config to a dictionary and check if everything is there
    cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Call the main function with the config
    main(cfg)

if __name__ == '__main__':

    main_wrapper()
