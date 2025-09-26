# Model Optimization Guidelide

## Experiment 1: Reproduce Publication Results
First the Timm model is compared to the custom DLA34 implementation from the publication on 

Use these configs:
- exp1_customdla34_publication_setting
- exp1_timmdla34_publication_setting
- exp1_timmdla102_publication_setting
- exp1_timmdla102x_publication_setting

```shell
HerdNet/tools/train_cli.py --config-name="dinoV2_large_publication_setting" --config-path="../configs/iguana"
```


## Experiment 2: Hyperparameter Optimization with WandDB sweeps
This will optimise the dla34 model and look for better batch_sizes, learning rates, weight decays. To get this running, do the following:

```shell
# Optional, start a screen session
screen -S herdnet_sweep
# in your HerdNet BaseFolder
conda activate <your conda environemt> # activate the conda environment you use for this repo
wandb sweep configs/experiment_publication_reproduction/exp2_sweep_hyp.yaml  # create a sweep with the sweep config. This creates an ID for the sweep
# It will output sth. like: wandb: Run sweep agent with: wandb agent username/herdnet_delplanque2022_exp2_hyp_sweep/uuxyz7ch

# Run sweep agent with on multiple GPUS

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ wandb agent <The sweep ID> --count 100

# if you trust the config run and detach
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ wandb agent <The sweep ID> --count 100 

# Start multiple agents in the background (on different GPUs if possible)
mkdir -p logs

# Start multiple agents (adjust count as needed)
for i in {1..4}; do
    PYTHONPATH=./ wandb agent karisu/herdnet_delplanque2022_exp2_hyp_sweep/ursam84c > logs/agent_$i.log 2>&1 &
    echo "Started agent $i with PID: $!"
    sleep 2  # Small delay between starts
done

```


## Experiment 3: Wandb sweeps for better augmentation strategies



```shell 
# pip install hydra-joblib-launcher
PYTHONPATH=$PYTHONPATH:../ python train_cli.py \
  --config-name=experiment_publication_reproduction \
  --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=3
```


### Experiment Setup for Summer School
```shell

# Optional, start a screen session
screen -S herdnet

screen -ls

screen -r Herdnet

# list all windows
ctrl-A w
# switch between windows
ctrl-A n  (next)

cd tools


### best setting
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_aed_winning_corr_dinoS_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_aed_winning_corr_dinoL_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/"   > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_aed_winning_corr_dla34_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_aed_winning_corr_dla102_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

# offline crops
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

#### aed with default loss

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x4_aed_winning_corr_dla34_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x4_aed_winning_corr_dla102_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x4_aed_winning_corr_dinoL_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x4_aed_winning_corr_dinoS_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

## aed with default loss, offline crops and default augmentations
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x3_aed_corr_dla34_pub_train_crop_eval_full_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

## aed with with winning loss, offline crops and default augmentations
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x3_aed_corr_dla34_pub_train_crop_eval_full_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_iguana_winning_corr_dinoS_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_iguana_winning_corr_dinoL_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_iguana_winning_corr_dla34_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 



#### Train the delplanque general dataset
```shell

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x3_GD_dla34_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x3_GD_dla102_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x3_GD_dinoS_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x3_GD_dinoL_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

```


### Train Eikelboom Experiment 1
```shell
PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x1_eikelboom_dinoL_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x1_eikelboom_dinoB_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x1_eikelboom_dinoS_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x1_eikelboom_dla34_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x1_eikelboom_dla60_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x1_eikelboom_dla102_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

```


PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x2_eikelboom_dinoL_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x2_eikelboom_dinoB_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x2_eikelboom_dinoS_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x2_eikelboom_dla34_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x2_eikelboom_dla60_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 train_cli.py --config-name="x2_eikelboom_dla102_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

full size image inference and evaluation, use inference_test
```shell 

```


