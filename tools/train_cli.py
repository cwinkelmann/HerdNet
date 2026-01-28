import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from animaloc.utils.train import main

# Defaults
config_name = "TODO"
config_path="TODO"



if __name__ == '__main__':
    @hydra.main(config_path=config_path, config_name=config_name, version_base="1.1")
    def main_wrapper(cfg: DictConfig):
        Path(os.curdir).resolve()

        return main(cfg)

    main_wrapper()