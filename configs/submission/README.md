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

conda activate HerdNetCarrotConda


```


### Default performance of on islands

```shell
# Floreana
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_inference__floreana_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 &


# Fernandina 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="x10_hyp_inference__fernandina_cvat_corr_dla34_pub_train_tile_eval_tile_aug_hypopt" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 &


# Genovesa
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_genovesa_dla34_all_default" --config-path="../configs/submission/"  > /dev/null 2>&1 &


```
### optimized parameters performance of on islands

```shell
#  Fernandina
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_fernandina_all_best_dla34" --config-path="../configs/submission/"  > /dev/null 2>&1 &


# Floreana 
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_floreana_all_best_dla34" --config-path="../configs/submission/"  > /dev/null 2>&1 &


## Genovesa
#PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_genovesa_dla34_all_default" --config-path="../configs/submission/"  > /dev/null 2>&1 &

```


```

### Learning Data Curve

```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_wrapper.py > /dev/null 2>&1 & 
```


## Experiment 1: Fine Tune an Iguana Model on Genovesa
First the Timm model is compared to the custom DLA34 implementation from the publication on eikelboom and delplanque dataset



```shell

### best setting
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="genovesa_dla34" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 
PYTHONPATH=$PYTHONPATH:./ python3 tools/inference_test.py --config-name="genovesa_dla34" --config-path="../configs/experiment_publication_reproduction/"  > /dev/null 2>&1 & 

# use the best model for inference on genovesa test set
PYTHONPATH=$PYTHONPATH:./ python3 tools/inference_test.py model.load_from="/home/cwinkelmann/work/Herdnet/best_models/genovesa/13-05-47/best_loss_model.pth"

# evaluate the detection using prediction from the above inference test
051_evaluate_point_detector.py 

# convert a orthomosaic and its shapefile to tiles for inference

inferencing/014_simple_geospatial_inference # produces a geojson

training_data_preparation/orthomosaic/0432_convert_shapefile_ortho_herdnet.py

071_correction_factor_geospatial_vs_dino
```


Train a big dino model
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_fernandina_all_best_dla34" --config-path="../configs/submission/"  > /dev/null 2>&1 &

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_fernandina_all_dinov2" --config-path="../configs/submission/"  > /dev/null 2>&1 &


PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_fernandina_all_dinov3" --config-path="../configs/submission/"  > /dev/null 2>&1 &

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_fernandina_all_dinov3_fpn" --config-path="../configs/submission/"  > /dev/null 2>&1 &
```


Train a big dino model
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_alldata_all_best_DINO" --config-path="../configs/submission/"  > /dev/null 2>&1 &
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_alldata_all_best_dla34" --config-path="../configs/submission/"  > /dev/null 2>&1 &
```


### Final training will all combined insights, corrected datasets
```shell
PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_last_run_DINO" --config-path="../configs/submission/"  > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_last_run_dla34" --config-path="../configs/submission/"  > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_last_run_convnext" --config-path="../configs/submission/"  > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_last_run_swin" --config-path="../configs/submission/"  > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_last_run_convnext_camouflaged" --config-path="../configs/submission/"  > /dev/null 2>&1 & 

PYTHONPATH=$PYTHONPATH:./ python3 tools/train_cli.py --config-name="f1_last_run_dual" --config-path="../configs/submission/"  > /dev/null 2>&1 & 


```