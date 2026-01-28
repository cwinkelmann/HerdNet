"""
config based inference script which takes the test/herdnets.yaml configuration file, predicts instances, evaluates performances etc

"""

import hydra
from PIL import Image
from loguru import logger
from omegaconf import DictConfig

from animaloc.utils.inference import inference

Image.MAX_IMAGE_PIXELS = None  # Disable the limit


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