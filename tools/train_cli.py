import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tools.train import main
# Defaults
config_name = "dinoV2_large_publication_setting"
config_path="../configs/experiment_publication_reproduction"



if __name__ == '__main__':
    @hydra.main(config_path=config_path, config_name=config_name, version_base="1.1")
    def main_wrapper(cfg: DictConfig):
        Path(os.curdir).resolve()

        return main(cfg)


    main_wrapper()