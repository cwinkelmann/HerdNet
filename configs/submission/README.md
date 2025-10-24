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


