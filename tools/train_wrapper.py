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

config_name = "x14_learning_curve_DR2_fernandina"
config_name = "x14_learning_curve_DR2_floreana"

config_path = '../configs/experiment_publication_reproduction'





@hydra.main(config_path=config_path, config_name=config_name)
def main_wrapper_learning_curve(cfg: DictConfig):
    """
    Main function to run the training process with hydra configuration.
    """


    # learning curve setup Floreana
    # for i in range(1, 130, 5):
    #     augmentation_inflation = 1
    #     run_name_template = 'ig_Floreana_learning_curve'
    #     wandb_tags = ['train_hasty', 'single', 'pretrained=true', 'Floreana', 'dla34', 'learning_curve',
    #                   f'augmentation_inflation={augmentation_inflation}', f'num={i}']
    #     cfg.wandb_project = 'ig_Floreana_learning_curve'
    #
    #     cfg.wandb_tags = wandb_tags
    #     cfg.wandb_run = f'{run_name_template}_image{i}'
    #     cfg.datasets.train.csv_file = f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Floreana_detection_il_{i}/train/herdnet_format_512_0_crops.csv'
    #     cfg.datasets.train.root_dir = f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Floreana_detection_il_{i}/train/crops_512_num{i}_overlap0'
    #
    #     # cfg.datasets.train.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/floreana_sample/train/herdnet_format_512_0_crops.csv'
    #     # cfg.datasets.train.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/floreana_sample/train/crops_512_numNone_overlap0'
    #
    #     cfg.datasets.validate.csv_file = f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Floreana_detection/val/herdnet_format_512_0_crops.csv'
    #     cfg.datasets.validate.root_dir = f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Floreana_detection/val/crops_512_numNone_overlap0'
    #
    #     # cfg.datasets.validate.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_detection/val/herdnet_format.csv'
    #     # cfg.datasets.validate.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Floreana_detection/val/Default'
    #     # cfg.wandb_tags = wandb_tags
    #     # cfg.device_name = null
    #     # cfg.datasets.train.augmentation_multiplier = augmentation_inflation
    #
    #     # omegaconf.OmegaConf.to_container(
    #     #     cfg, resolve=True, throw_on_missing=True
    #     # )
    #
    #     output_path = main(cfg)

    for i in range(185, 160, -5):
        augmentation_inflation = 1
        run_name_template = 'ig_Fernandina_s_learning_curve'
        wandb_tags = ['train_hasty', 'single', 'pretrained=true', 'Fernandina_s', 'dla34', 'learning_curve', f'augmentation_inflation={augmentation_inflation}', f'num={i}']
        cfg.wandb_project = 'ig_Fernandina_s_learning_curve'


        cfg.wandb_tags = wandb_tags
        cfg.wandb_run = f'{run_name_template}_image{i}'
        cfg.datasets.train.csv_file = f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Fernandina_s_detection_il_{i}/train/herdnet_format_512_0_crops.csv'
        cfg.datasets.train.root_dir = f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Fernandina_s_detection_il_{i}/train/crops_512_num{i}_overlap0'

        # cfg.datasets.train.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/floreana_sample/train/herdnet_format_512_0_crops.csv'
        # cfg.datasets.train.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/floreana_sample/train/crops_512_numNone_overlap0'


        cfg.datasets.validate.csv_file = f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Fernandina_s_detection/val/herdnet_format_512_0_crops.csv'
        cfg.datasets.validate.root_dir = f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Fernandina_s_detection/val/crops_512_numNone_overlap0'

        # cfg.datasets.validate.csv_file = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Fernandina_detection/val/herdnet_format.csv'
        # cfg.datasets.validate.root_dir = f'/home/cwinkelmann/work/Herdnet/data/2025_07_10_final_point_detection_edge_black_512/Floreana_detection/val/Default'
        # cfg.wandb_tags = wandb_tags
        # cfg.device_name = null
        # cfg.datasets.train.augmentation_multiplier = augmentation_inflation

        # omegaconf.OmegaConf.to_container(
        #     cfg, resolve=True, throw_on_missing=True
        # )

        output_path = main(cfg)

        # shutil.copy(Path(config_path) / f"{config_name}.yaml", output_path / f"{config_name}.yaml")




if __name__ == '__main__':
    # for config_name in config_list:
    #     @hydra.main(config_path=config_path, config_name=config_name, version_base="1.1")
    #     def main_wrapper(cfg: DictConfig):
    #         """
    #         Main function to run the training process with hydra configuration.
    #         """
    #         # cfg.training_settings.epochs = 1
    #         # cfg.training_settings.valid_freq = 1
    #         main(cfg)
    #
    #     main_wrapper()

    main_wrapper_learning_curve()