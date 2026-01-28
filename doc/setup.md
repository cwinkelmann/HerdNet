## Setup Extended

### Docker Training container
```shell
docker build -t herdnet -f Dockerfile .

```

### Run a wandb sweep
Since the training generates many models set .wandb/settings
[default]
artifact_cache_size = 10GB

```shell
conda activate HerdNetCarrotConda
wandb sweep sweep_hyp.yaml 

Run sweep agent with: 

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../ wandb agent karisu/herdnet/bottq19g
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../ wandb agent karisu/herdnet/bottq19g
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../ wandb agent karisu/herdnet/bottq19g
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../ wandb agent karisu/herdnet/bottq19g
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=../ wandb agent karisu/herdnet/bottq19g

CUDA_VISIBLE_DEVICES=5 PYTHONPATH=../ wandb agent karisu/herdnet/bottq19g


CUDA_VISIBLE_DEVICES=6 PYTHONPATH=../ wandb agent karisu/herdnet/bottq19g

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=../ wandb agent karisu/herdnet/bottq19g



```


```shell


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="aed_dinov2_base_publication_setting_train_full_eval_crop_aug_all" --config-path="../configs/experiment_publication_reproduction"


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="aed_dinov2_base_publication_setting_train_full_eval_crop_aug_all" --config-path="../configs/experiment_publication_reproduction" --config-name="aed_t_dinov2_base_publication_setting_train_full_eval_crop_aug_all"
--config-path="../configs/experiment_publication_reproduction" &