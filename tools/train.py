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

import hydra
from omegaconf import DictConfig, omegaconf

from animaloc.utils.train import main

config_path = '../configs/reference_data/delplanque2022'
config_name = 'dla34_custom_publication'

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
