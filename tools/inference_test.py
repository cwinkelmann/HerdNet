"""
config based inference script which takes the test/herdnets.yaml configuration file, predicts instances, evaluates performances etc

"""

import hydra
from PIL import Image
from loguru import logger
from omegaconf import DictConfig

from animaloc.utils.inference import inference

Image.MAX_IMAGE_PIXELS = None  # Disable the limit

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
# config_name = "x6_iguana_winning_corr_dla34_pub_train_full_eval_full_aug_all"
# config_name = "x6_iguana_winning_corr_dla102_pub_train_full_eval_full_aug_all"

# config_name = "x6_iguana_winning_corr_dinoL_pub_train_full_eval_full_aug_all_ds_fcdm"
# config_name = "x6_iguana_winning_corr_dinoS_pub_train_full_eval_full_aug_all_ds_fcdm"
# config_name = "x6_iguana_winning_corr_dla34_pub_train_full_eval_full_aug_all_ds_fcdm"
# config_name = "x6_iguana_winning_corr_dla102_pub_train_full_eval_full_aug_all_ds_fcdm"

# config_name = "x1_eikelboom_dla34_train_crop_eval_crop_publication"
# config_name = "x1_eikelboom_dla60_train_crop_eval_crop_publication"
# config_name = "x1_eikelboom_dla102_train_crop_eval_crop_publication"
# config_name = "x02_reference_DLA34_eikelboom_ObjectAwareCrop_full_validation"

# config_path = "../configs/experiment_publication_reproduction/"
#
# config_name = "genovesa_dla34"
# config_path = "../configs/submission/"

# config_name = "x02_INFERENCE_DLA34_delplanque" # for inferecning we need to use a config with a stitcher.
# config_path = "../configs/experiment_publication_reproduction/"

config_name = "f1_alldata_all_best_dla34"
config_path = "../configs/submission/"


config_name = "f1_last_run_convnext_camouflaged"
config_path = "../configs/submission/"


@hydra.main(config_path=config_path, config_name=config_name)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the inference test with the given configuration.
    It initializes the model, prepares the dataset, and evaluates the model on the test set.
    """
    logger.info(f"Running inference test with config: {cfg}")
    inference(cfg, plain_inference=True)





if __name__ == '__main__':
    main()