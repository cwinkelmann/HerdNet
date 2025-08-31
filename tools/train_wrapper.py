import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig
import yaml
from loguru import logger
from tools.train import main

# config_name = "config_2025_07_10_hasty_floreana"
# config_name = "config_2025_07_10_hasty_floreana_sweep"
# config_name = "config_2025_07_10_hasty_fernandina_m"
# config_name = "config_2025_07_10_hasty_fernandina_s"
# config_name = "config_2025_07_10_hasty_genovesa"
# config_name = "config_2025_07_10_hasty_all_single"

# config_name = "config_2025_07_11_eikelboom"

# config_name = "config_2025_07_04_hasty_floreana_n"

# config_name = "config_2025_07_04_hasty_Rest"
# config_name = "config_2025_07_04_hasty_all"
# config_name="config_2025_07_07_geo_head_LQ"
# config_name="config_2025_07_07_geo_body_LQ"
# config_name="config_2025_07_07_geo_body_HQ"
# config_name="config_2025_07_08_all_detection"


# config_name="config_2025_07_13_hasty_edge_blackout_1024"
# config_name="config_2025_07_22_weinstein_640"
# config_name="config_2025_07_27_weinstein_full"

## Refactoring of herdnet
# config_name="config_2025_07_27_iguana_legacy_dla34"
config_name = "config_2025_07_27_iguana_timm_DLA"
# config_name="config_2025_07_27_iguana_timm_convnext"
# config_name="config_2025_07_27_iguana_timm_DinoV2"

# config_name="config_2025_07_10_hasty_floreana"
# config_name="config_2025_07_13_hasty_fernandina_s_edge_blackout_512"

## label correction experiment
# config_name = "config_2025_08_08_dinov2_train_val_classic"
# config_name = "config_2025_08_08_dinov2_train_val_corrected"
# config_name = "config_2025_08_08_dinov2_train_val_inverted_val_classic"
# config_name = "config_2025_08_08_dinov2_train_val_inverted_val_corrected"
# config_name = "config_2025_08_08_dla34_train_val_inverted_val_corrected"

# config_name = "dinov2_masks"
# config_name = "config_2025_08_10_dinov2_floreana_fernandia_all_val_fernandina"

config_path = '../configs'


# config_name = 'dla34_custom_publication'
# config_name = 'dla34_timm_2'
# config_name = 'convnext_timm_random_crop'
# config_name = 'convnext_dla_random_crop'

config_path = '../configs/reference_data/delplanque2022'
config_list = [
    'dla34_custom_publication',
    'dla34_timm',
    # "dla102x",
    # "convnext_tiny",
    # "convnext_small",
    # "efficientnet_b7_offline_crop",
    # "DinoV2_small",
    # "DinoV2_base"
]

# config_path = '../configs/reference_data/delplanque2022_opt'
# config_list = [
#     # 'dla34_custom_publication',
#     # 'dla34_timm',
#     # "dla_102x",
#     # "convnext_timm",
#      # "convnext_small",
#     # "efficientnet_b7_offline_crop",
#     # "DinoV2"
#     "dla102x_ft"
# ]

# config_path = '../configs/reference_data/delplanque2022_opt_full_size_val'
# config_list = [
#     # 'dla34_custom_publication',
#     # 'dla34_timm',
#     # "dla_102x",
#     # "convnext_timm",
#      # "convnext_small",
#     # "efficientnet_b7_offline_crop",
#     # "DinoV2_rcrop_alpha_beta"
#     "DinoV2_rcrop"
#     # "dla102x_ft"
# ]

# config_path = '../configs/reference_data/delplanque2022_opt'
# # config_name = 'dla34ft_random_crop'
# # config_name = 'dla102x_ft'
# config_name = 'dla60_res2next_random_crop'
# config_name = 'efficientnet_b7'
# config_name = 'efficientnet_b7_offline_crop'
# config_name = 'DinoV2'


config_path = '../configs/experiment_publication_reproduction'
config_list = [
    # 'customdla34_publication_setting',
    #  'customdla34_publication_setting_quick_val_los',
     'customdla34_publication_setting',
    # 'timmdla34_publication_setting',
    # 'timmdla102x_publication_setting',
    # "dla102x",
    # "convnext_tiny",
    # "convnext_small",
    # "efficientnet_b7_offline_crop",
    # "DinoV2_small",
    # "DinoV2_small_publication_setting",
    # "DinoV2_base"
]


@hydra.main(config_path=config_path, config_name=config_name)
def main_wrapper_learning_curve(cfg: DictConfig):
    """
    Main function to run the training process with hydra configuration.
    """

    # learning curve setup

    for i in range(1, 25, 1):
        # run_name_template = 'ig_Fernandina_s_learning_curve_dino'
        augmentation_inflation = 1
        # wandb_tags = ['train_hasty', 'single', 'pretrained=true', 'Fernandina_s', 'dinoV2', 'learning_curve', f'augmentation_inflation={augmentation_inflation}', f'num={i}']

        # cfg.wandb_run = f'{run_name_template}_image{i}'
        #
        # cfg.datasets.train.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_s_detection_il_{i}/train/herdnet_format_512_0_crops.csv'
        # cfg.datasets.train.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_s_detection_il_{i}/train/crops_512_num{i}_overlap0'

        # cfg.datasets.train.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/floreana_sample/train/herdnet_format_512_0_crops.csv'
        # cfg.datasets.train.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/floreana_sample/train/crops_512_numNone_overlap0'


        #cfg.datasets.validate.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_s_detection/val/herdnet_format_512_0_crops.csv'
        #cfg.datasets.validate.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_s_detection/val/crops_512_numNone_overlap0'

        cfg.datasets.validate.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Floreana_detection/val/herdnet_format.csv'
        cfg.datasets.validate.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Floreana_detection/val/Default'
        # cfg.wandb_tags = wandb_tags
        # cfg.device_name = null
        cfg.datasets.train.augmentation_multiplier = augmentation_inflation

        # omegaconf.OmegaConf.to_container(
        #     cfg, resolve=True, throw_on_missing=True
        # )

        output_path = main(cfg)

        shutil.copy(Path(config_path) / f"{config_name}.yaml", output_path / f"{config_name}.yaml")




if __name__ == '__main__':
    for config_name in config_list:
        @hydra.main(config_path=config_path, config_name=config_name, version_base="1.1")
        def main_wrapper(cfg: DictConfig):
            """
            Main function to run the training process with hydra configuration.
            """
            # cfg.training_settings.epochs = 1
            # cfg.training_settings.valid_freq = 1
            main(cfg)

        main_wrapper()
