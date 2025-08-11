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

from pathlib import Path
from typing import List, Tuple, Any

import hydra
import pandas
import torch
import wandb
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, omegaconf
from torch.utils.data import DataLoader

import animaloc
from animaloc.models.utils import LossWrapper, load_model
from animaloc.utils.seed import set_seed
from animaloc.utils.useful_funcs import current_date
from tools.train_helper import _get_collate_fn, _build_sampler, _load_albu_transforms, _load_end_transforms, \
    _build_model, _load_losses, _define_evaluator, _define_visualiser, get_least_occupied_gpu_nvidia_smi
from animaloc.vizual.custom_vis import plot_heatmaps








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

    if cfg.device_name is None:
        # Automatically select the least occupied GPU
        cfg.device_name = get_least_occupied_gpu_nvidia_smi()
        logger.info(f"Using device: {cfg.device_name}")
    device = torch.device(cfg.device_name)



    train_df = pandas.read_csv(train_args.csv_file)
    # TODO I would argue, modifying the training data while training is not a good idea
    # _set_species_labels(dict(cfg.datasets.class_def), train_df)

    train_dataset = animaloc.datasets.__dict__[train_args.name](
        csv_file = train_df,
        root_dir = train_args.root_dir,
        albu_transforms = _load_albu_transforms(train_args.albu_transforms),
        end_transforms = _load_end_transforms(train_args.end_transforms),
        augmentation_multiplier = train_args.augmentation_multiplier,
        )

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
        # TODO why is it batch size 1 here: because the the datasets keeps track of metadata image_name etc. we get a wrong RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 27
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
                end_transforms=list(cfg.datasets.train.end_transforms.keys()),
                FIDT=cfg.datasets.train.end_transforms.MultiTransformsWrapper.FIDT,
                PointsToMask=dict(cfg.datasets.train.end_transforms.MultiTransformsWrapper.PointsToMask),
                input_size=cfg.datasets.img_size,
                class_def=cfg.datasets.class_def,
                augmentation_multiplier=cfg.datasets.train.augmentation_multiplier,
                model = dict(cfg.model.kwargs),
                model_backbone=cfg.model.kwargs.backbone if "backbone" in cfg.model.kwargs else None, # TODO fix it when it works
                loss_dict=dict(cfg.losses),
                base_model=cfg.model.load_from,
                dataloader={"train": cfg.datasets.train.name, "val": cfg.datasets.validate.name},
                data={"train_csv": cfg.datasets.train.csv_file, 'train_dir': cfg.datasets.train.root_dir,
                      "val_csv": cfg.datasets.validate.csv_file, 'val_dir': cfg.datasets.validate.root_dir, },
                training_settings=dict(cfg.training_settings),
                training_auto_ld=dict(cfg.training_settings.auto_lr),

                num_training_annotations=len(train_df),
                num_val_annotations=len(val_df),
                num_training_images=train_df.images.nunique(),
                num_val_images=val_df.images.nunique(),
                num_main_images=len(set(train_df['images'].str.replace(r'_x\d+_y\d+\.', '.', regex=True))),


                evaluator_name=cfg.training_settings.evaluator.name if cfg.training_settings.evaluator is not None else None,
                evaluator_kwargs=cfg.training_settings.evaluator.kwargs if cfg.training_settings.evaluator is not None else None,
                evaluator=cfg.training_settings.evaluator,
                stitcher=cfg.training_settings.stitcher if cfg.training_settings.stitcher is not None else None
            )
        )

        date = current_date()
        wandb.run.name = f'{date}_' + cfg.wandb_run + f'_{wandb.run.id}'
        wandb.run.tags = [f"train_ds: {cfg.datasets.train.name}", f"val_ds: {cfg.datasets.validate.name}"] + cfg.wandb_tags
        # TODO this is the time to upload metrics about the data


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

    if cfg.wandb_flag:
        # iterate through the dataloader to check if it works
        max_plot = 30
        for i, (img_tensor, target) in enumerate(train_dataset):
            if i >= max_plot:
                break

            heatmap = target[0].squeeze(0)
            cls_map = target[1]

            fig, axes = plot_heatmaps(img_tensor.squeeze(0), heatmap,
                                      show_argmax_overlay=False, max_channels=1)

            wandb.log({f'augmented_dataset_examples': wandb.Image(fig)})
            fig.savefig(os.path.join("/home/cwinkelmann/work/Herdnet/playground/transformed_training_data", # TODO get the right path
                                     f'augmented_dataset_examples.png'))

            plt.close(fig)



    # Build the model
    logger.info('Building the model ...')
    model = _build_model(cfg)
    # model.check_trainable_parameters() # TODO implement this in all models
    # Prepare for training
    logger.info('Preparing for training ...')
    criterions = _load_losses(cfg)
    model = LossWrapper(model, criterions).to(device)

    if cfg.model.load_from is not None:
        model = load_model(model, cfg.model.load_from, device=device)

        if 'HerdNet' in cfg.model.name:
            if cfg.model.freeze is not None:
                model.model.freeze(layers=list(cfg.model.freeze))
                logger.info(f"Layers {list(cfg.model.freeze)} freezed")
    
    if cfg.training_settings.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = cfg.training_settings.lr,
            weight_decay = cfg.training_settings.weight_decay
            )
    elif cfg.training_settings.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = cfg.training_settings.lr,
            weight_decay = cfg.training_settings.weight_decay
            )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr = cfg.training_settings.lr, 
            weight_decay = cfg.training_settings.weight_decay
            )
    
    # Watch the model's gradients during training
    visualiser = None
    if cfg.wandb_flag:
        wandb.watch(model)

        if cfg.training_settings.visualiser is not None:
            visualiser = _define_visualiser(cfg)

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
        optimizer = optimizer, 
        num_epochs = cfg.training_settings.epochs, 
        auto_lr = auto_lr,
        # adaloss = cfg.training_settings.adaloss,
        val_dataloader = val_dataloader, 
        evaluator = evaluator,
        device_name = cfg.device_name,
        vizual_fn = visualiser,
        work_dir = work_dir,
        print_freq = cfg.training_settings.print_freq,
        valid_freq = cfg.training_settings.valid_freq,
        )

    if cfg.model.resume_from is not None:
        logger.info(f'Resuming training from \'{cfg.model.resume_from}\' ...')
        trainer.resume(
            pth_path = cfg.model.resume_from, 
            select = select,
            validate_on = validate_on, 
            load_optim = True,
            wandb_flag = cfg.wandb_flag
            )
    else:
        logger.info('Starting training ...')
        trainer.start(
            cfg.training_settings.warmup_iters, 
            select = select,
            validate_on = validate_on, 
            wandb_flag = cfg.wandb_flag
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
        pth_file['mean'] =  list(norm_trans.mean)
        pth_file['std'] = list(norm_trans.std)

        torch.save(pth_file, path)
        logger.info(f"Saved Model {pth_name} with added information in {path}")

    if cfg.wandb_flag:
        wandb.finish()


# config_name = "config_2025_07_10_hasty_floreana"
# config_name = "config_2025_07_10_hasty_floreana_sweep"
# config_name = "config_2025_07_10_hasty_fernandina_m"
# config_name = "config_2025_07_10_hasty_fernandina_s"
# config_name = "config_2025_07_10_hasty_genovesa"
# config_name = "config_2025_07_10_hasty_all_single"

# config_name = "config_2025_07_11_eikelboom"

# config_name = "config_2025_07_04_hasty_floreana_n"

#config_name = "config_2025_07_04_hasty_Rest"
#config_name = "config_2025_07_04_hasty_all"
#config_name="config_2025_07_07_geo_head_LQ"
# config_name="config_2025_07_07_geo_body_LQ"
# config_name="config_2025_07_07_geo_body_HQ"
#config_name="config_2025_07_08_all_detection"


# config_name="config_2025_07_13_hasty_edge_blackout_1024"
# config_name="config_2025_07_22_weinstein_640"
# config_name="config_2025_07_27_weinstein_full"

## Refactoring of herdnet
# config_name="config_2025_07_27_iguana_legacy_dla34"
# config_name="config_2025_07_27_iguana_timm_DLA"
#config_name="config_2025_07_27_iguana_timm_convnext"
# config_name="config_2025_07_27_iguana_timm_DinoV2"

# config_name="config_2025_07_10_hasty_floreana"
# config_name="config_2025_07_13_hasty_fernandina_s_edge_blackout_512"

## label correction experiment
#config_name = "config_2025_08_08_dinov2_train_val_classic"
#config_name = "config_2025_08_08_dinov2_train_val_corrected"
#config_name = "config_2025_08_08_dinov2_train_val_inverted_val_classic"
#config_name = "config_2025_08_08_dinov2_train_val_inverted_val_corrected"
#config_name = "config_2025_08_08_dla34_train_val_inverted_val_corrected"

config_name = "config_2025_08_10_dinov2_floreana_all_val_fernandina"
# config_name = "config_2025_08_10_dinov2_floreana_fernandia_all_val_fernandina"

@hydra.main(config_path='../configs/iguana/label_correction', config_name=config_name)
def main_wrapper(cfg: DictConfig):
    """
    Main function to run the training process with hydra configuration.
    """
    
    # learning curve setup


    # for i in range(1, 25, 1):
    #     # run_name_template = 'ig_Fernandina_s_learning_curve_dino'
    #     augmentation_inflation = 1
    #     # wandb_tags = ['train_hasty', 'single', 'pretrained=true', 'Fernandina_s', 'dinoV2', 'learning_curve', f'augmentation_inflation={augmentation_inflation}', f'num={i}']
    #
    #     # cfg.wandb_run = f'{run_name_template}_image{i}'
    #     #
    #     # cfg.datasets.train.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_s_detection_il_{i}/train/herdnet_format_512_0_crops.csv'
    #     # cfg.datasets.train.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_s_detection_il_{i}/train/crops_512_num{i}_overlap0'
    #
    #     # cfg.datasets.train.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/floreana_sample/train/herdnet_format_512_0_crops.csv'
    #     # cfg.datasets.train.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/floreana_sample/train/crops_512_numNone_overlap0'
    #
    #
    #     #cfg.datasets.validate.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_s_detection/val/herdnet_format_512_0_crops.csv'
    #     #cfg.datasets.validate.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_s_detection/val/crops_512_numNone_overlap0'
    #
    #     cfg.datasets.validate.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Floreana_detection/val/herdnet_format.csv'
    #     cfg.datasets.validate.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Floreana_detection/val/Default'
    #     # cfg.wandb_tags = wandb_tags
    #     # cfg.device_name = null
    #     cfg.datasets.train.augmentation_multiplier = augmentation_inflation
    #
    #     # omegaconf.OmegaConf.to_container(
    #     #     cfg, resolve=True, throw_on_missing=True
    #     # )
    #
    #     main(cfg)

    # omegaconf.OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # )
    #
    main(cfg)

if __name__ == '__main__':
    #hydra.initialize(config_path='../configs', job_name="dynamic_hydra")

    #cfg = hydra.compose(config_name=config_name)
    # cfg = hydra.compose(config_name="config_FMO03_02_05")
    main_wrapper()
    #main(config_name)