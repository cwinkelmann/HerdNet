# Model Optimization Guidelide

### Experiment Setup using screen
```shell

# Optional, start a screen session
screen -S herdnet

screen -ls

screen -r herdnet

# list all windows
ctrl-A w
# switch between windows
ctrl-A n  (next)
ctrl-A c  (new window)

cd HerdNet
```


### Training Data Curve

```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_wrapper.py > /dev/null 2>&1 & 
```


## Experiment 1: Reproduce Publication Results
First the Timm model is compared to the custom DLA34 implementation from the publication on eikelboom and delplanque dataset



```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla34_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_Herdnetdla34_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla60_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla102_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoL_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoS_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10

```

Eikelboom dataset with default loss function and more augmentations
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla34_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla60_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla102_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoL_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoS_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10




```


```shell
## general dataset
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_Herdnetdla34_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dla34_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dla60_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dla102_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dinoS_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dinoL_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
```


## Experiment 2: Hyperparameter Optimization with WandDB sweeps
This will optimise the dla34 model and look for better batch_sizes, learning rates, weight decays. To get this running, do the following:

```shell
# Optional, start a screen session
screen -S herdnet_sweep
# in your HerdNet BaseFolder
conda activate <your conda environemt> # activate the conda environment you use for this repo

# training hyperparameter sweep
wandb sweep configs/experiment_publication_reproduction/exp2_sweep_hyp.yaml  # create a sweep with

# augmentation hyperparameter sweep optimising for a low val_focal_loss
wandb sweep configs/experiment_publication_reproduction/x10_sweep_hyp_aug.yaml  # create a sweep with
# the sweep config. This creates an ID for the sweep
# It will output sth. like: wandb: Run sweep agent with: wandb agent username/herdnet_delplanque2022_exp2_hyp_sweep/uuxyz7ch

# Run sweep agent with on multiple GPUS

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ wandb agent <The sweep ID> --count 100



```

### 3 Sweeps
#### 1. Hyperparameter Sweep for Learning Rate, Weight Decay, Batch Size, etc
```shell

# Test if the config works 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x15_2_hyper_parameter_optimisation_floreana" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 

wandb sweep configs/experiment_publication_reproduction/x15_sweep_hyp_train.yaml
PYTHONPATH=./ wandb agent  karisu/iguana_train_sweep/x3eo08es --count 100  > /dev/null 2>&1 & 

```




```shell

### best setting
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_aed_winning_corr_dinoS_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_aed_winning_corr_dinoL_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/"   > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_aed_winning_corr_dla34_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_aed_winning_corr_dla102_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

# offline crops
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

#### aed with default loss

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x4_aed_winning_corr_dla34_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x4_aed_winning_corr_dla102_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x4_aed_winning_corr_dinoL_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x4_aed_winning_corr_dinoS_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

## aed with default loss, offline crops and default augmentations
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_aed_corr_dla34_pub_train_crop_eval_full_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

## aed with with winning loss, offline crops and default augmentations
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_aed_corr_dla34_pub_train_crop_eval_full_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dinoS_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dinoL_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla34_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 



#### Train the delplanque general dataset
```shell

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_GD_dla34_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_GD_dla102_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_GD_dinoS_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_GD_dinoL_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

```


### Train Eikelboom Experiment 1
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoL_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoB_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoS_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla34_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla60_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla102_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

```

```
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoL_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoB_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoS_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla34_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla60_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla102_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
```

```shell 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x7_iguana_cvat_corr_dinoL_pub_train_crop_eval_crop_aug_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x7_iguana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x7_iguana_cvat_class_dinoL_pub_train_crop_eval_crop_aug_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

```


##### Add more augmentations
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x8_iguana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x8_iguana_cvat_class_dla34_pub_train_tile_eval_tile_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x8_iguana_cvat_corr_dinoL_pub_train_tile_eval_tile_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &




Train on random Crops of the full images

```
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x9_iguana_cvat_corr_dla34_pub_train_full_eval_tile_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


```

Evaluate if the hyperparameters found on crops are actually good on the real life metrics
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_default__floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_opt__floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

# fernandina

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_opt__fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_default__fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

```


Can FWK be solved with more data?
```shell
# fernandina

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x11_hyp_opt__fwk_hypopt_dinoL" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

# fancy augmentations
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x11_hyp_opt__fwk_hypopt_dla34" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

# default augmentaions
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x11_hyp_opt__fwk_default_aug_dla34" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x11_hyp_opt__fwk_hypopt_dla102" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
```


full size image inference and evaluation, use inference_test
```shell 

```


### Test or the effect of the down ratio

```
# Model Optimization Guidelide

### Experiment Setup using screen
```shell

# Optional, start a screen session
screen -S herdnet

screen -ls

screen -r herdnet

# list all windows
ctrl-A w
# switch between windows
ctrl-A n  (next)
ctrl-A c  (new window)

cd HerdNet
```


### Training Data Curve

```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_wrapper.py > /dev/null 2>&1 & 
```


## Experiment 1: Reproduce Publication Results
First the Timm model is compared to the custom DLA34 implementation from the publication on eikelboom and delplanque dataset



```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla34_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_Herdnetdla34_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla60_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla102_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoL_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoS_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10

```

### Eikelboom dataset with default loss function and more augmentations
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla34_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla60_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla102_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoL_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoS_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10




```

### General Dataset with different backbones
```shell
## general dataset
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_Herdnetdla34_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dla34_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dla60_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dla102_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dinoS_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_GD_dinoL_train_crop_eval_crop_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
sleep 10
```


## Experiment 2: Hyperparameter Optimization with WandDB sweeps
This will optimise the dla34 model and look for better batch_sizes, learning rates, weight decays. To get this running, do the following:

```shell
# Optional, start a screen session
screen -S herdnet_sweep
# in your HerdNet BaseFolder
conda activate <your conda environemt> # activate the conda environment you use for this repo

# training hyperparameter sweep
wandb sweep configs/experiment_publication_reproduction/exp2_sweep_hyp.yaml  # create a sweep with

# augmentation hyperparameter sweep optimising for a low val_focal_loss
wandb sweep configs/experiment_publication_reproduction/x10_sweep_hyp_aug.yaml  # create a sweep with
# the sweep config. This creates an ID for the sweep
# It will output sth. like: wandb: Run sweep agent with: wandb agent username/herdnet_delplanque2022_exp2_hyp_sweep/uuxyz7ch

# Run sweep agent with on multiple GPUS

PYTHONPATH=./ wandb agent <The sweep ID> --count 100

# if you trust the config run and detach
PYTHONPATH=./ wandb agent <The sweep ID> --count 100 

# Start multiple agents in the background (on different GPUs if possible)
mkdir -p logs



```


## Experiment 3: Wandb sweeps for better augmentation strategies




```shell

### best setting
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_aed_winning_corr_dinoS_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_aed_winning_corr_dinoL_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/"   > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_aed_winning_corr_dla34_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_aed_winning_corr_dla102_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

# offline crops
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

#### aed with default loss

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x4_aed_winning_corr_dla34_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x4_aed_winning_corr_dla102_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x4_aed_winning_corr_dinoL_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x4_aed_winning_corr_dinoS_pub_train_full_eval_full_aug_all_loss_defaul" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

## aed with default loss, offline crops and default augmentations
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_aed_corr_dla34_pub_train_crop_eval_full_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

## aed with with winning loss, offline crops and default augmentations
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_aed_corr_dla34_pub_train_crop_eval_full_aug_def_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dinoS_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dinoL_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla34_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_full_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x5_iguana_winning_corr_dla102_pub_train_crop_eval_full_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 



#### Train the delplanque general dataset
```shell

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_GD_dla34_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_GD_dla102_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_GD_dinoS_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x3_GD_dinoL_train_crop_eval_crop_aug_all_loss_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

```


### Train Eikelboom Experiment 1
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoL_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoB_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dinoS_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla34_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla60_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x1_eikelboom_dla102_train_crop_eval_crop_publication" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

```

```
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoL_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoB_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dinoS_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla34_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla60_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x2_eikelboom_dla102_train_crop_eval_crop_publication_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 
```

```shell 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x7_iguana_cvat_corr_dinoL_pub_train_crop_eval_crop_aug_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x7_iguana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x7_iguana_cvat_class_dinoL_pub_train_crop_eval_crop_aug_def" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

```


##### Add more augmentations
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x8_iguana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x8_iguana_cvat_class_dla34_pub_train_tile_eval_tile_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x8_iguana_cvat_corr_dinoL_pub_train_tile_eval_tile_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &




Train on random Crops of the full images

```
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x9_iguana_cvat_corr_dla34_pub_train_full_eval_tile_aug_all" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


```

Evaluate if the hyperparameters found on crops are actually good on the real life metrics
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_default__floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_opt__floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

# fernandina

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_opt__fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_default__fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

```


Can FWK be solved with more data?
```shell
# fernandina

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x11_hyp_opt__fwk_hypopt_dinoL" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

# fancy augmentations
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x11_hyp_opt__fwk_hypopt_dla34" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

# default augmentaions
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x11_hyp_opt__fwk_default_aug_dla34" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x11_hyp_opt__fwk_hypopt_dla102" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
```


## Effect of Down Ratio
```shell 
# Fernandina

# DR 16
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x12_hyp_DR2_to_8_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" model.kwargs.down_ratio=32 wandb_run='x12_hyp_DR32_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def' > /dev/null 2>&1 &

## DR 8
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x12_hyp_DR2_to_8_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" model.kwargs.down_ratio=8 wandb_run='x12_hyp_DR8_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def' > /dev/null 2>&1 &

## DR 4
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x12_hyp_DR2_to_8_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" model.kwargs.down_ratio=4 wandb_run='x12_hyp_DR4_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def' > /dev/null 2>&1 &

## DR 2
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x12_hyp_DR2_to_8_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" model.kwargs.down_ratio=2 wandb_run='x12_hyp_DR2_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def' > /dev/null 2>&1 &
```

```shell 
# Floreana

## DR 16
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x12_hyp_DR2_to_8_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" model.kwargs.down_ratio=16 wandb_run='x12_hyp_DR16_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def' > /dev/null 2>&1 &

## DR 8
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x12_hyp_DR2_to_8_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" model.kwargs.down_ratio=8 wandb_run='x12_hyp_DR8_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def' > /dev/null 2>&1 &

## DR 4
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x12_hyp_DR2_to_8_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" model.kwargs.down_ratio=4 wandb_run='x12_hyp_DR4_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def' > /dev/null 2>&1 &

## DR 2
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x12_hyp_DR2_to_8_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/" model.kwargs.down_ratio=2 wandb_run='x12_hyp_DR2_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_def' > /dev/null 2>&1 &
```






## Effect of Model Size

### Fernandina
```shell
# Fernandina
## DLA34 Reference
 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_RefDLA34_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25

## dla34
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA34_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA60_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA102_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA169_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DinoV2L_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
```


### Floreana
```shell

## DLA34 Reference
 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_RefDLA34_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25

## dla34
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA34_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA60_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA102_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA169_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DinoV2L_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
```

#### Model Sie with Patch Overlap
```shell
# Fernandina
## DLA34 with overlap
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA34_fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default_patchOverlap160" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25

# Floreana
## DLA34 with overlap

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x13_model_size_DLA34_floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_default_patchOverlap160" --config-path="../configs/experiment_publication_reproduction/" > /dev/null 2>&1 &
sleep 25
```


### Learning Curve once again
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_wrapper.py


```