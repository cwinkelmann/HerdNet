# HerdNet Training Skill

## Trigger Keywords
HerdNet, animaloc, FIDT, point detection, wildlife counting, DLA-34, DLA-60, DLA-102, DLA-169, DINOv2, HerdNetStitcher, LMDS, Hann window, patcher, CSVDataset, aerial wildlife, density map, focal inverse distance transform, drone counting, iguana detection, point annotation, HerdNetEvaluator, HerdNetLMDS, tile inference, overlapping patches, down_ratio, head_conv, matching_radius, ObjectAwareRandomCrop, FocalLoss, LossWrapper, warmup_iters, backbone_lr

## Does NOT Trigger

| If the user wants... | Use this skill instead |
|---|---|
| Species classification (post-detection) | wildlife-classification |
| Tiled bounding-box inference via SAHI | sahi-inference |
| MegaDetector on camera trap images | megadetector |
| Course material from thesis/defence slides | iguana-case-study |
| PCV curriculum adaptation | practical-cv-wildlife |
| Active learning loop orchestration | active-learning-wildlife |

## Description
This skill covers the full lifecycle of training HerdNet point-detection models using the `animaloc` package. HerdNet is a two-headed CNN architecture for simultaneously localizing and classifying animals in aerial imagery. It produces a localization heatmap via Focal Inverse Distance Transform (FIDT) maps and a classification map, enabling accurate counting and species identification from drone orthomosaics.

The canonical application is counting marine iguanas (*Amblyrhynchus cristatus*) on the Galapagos Islands, as validated in the Miesner 2025 thesis.

## Source Repository
- **HerdNet**: `animaloc` package, installed via `pip install -e HerdNet/`
- **Key source files**:
  - `animaloc/models/herdnet_timm_dla.py` -- HerdNetTimmDLA (DLA backbone via timm)
  - `animaloc/models/herdnet_dino_v2.py` -- HerdNetDINOv21 (DINOv2 backbone)
  - `animaloc/data/transforms.py` -- FIDT class (lines 290-425)
  - `animaloc/train/trainers.py` -- Trainer class (warmup, LR scheduling, early stopping)
  - `animaloc/eval/evaluators.py` -- HerdNetEvaluator (stitching, metrics, visualization)
  - `animaloc/eval/stitchers.py` -- HerdNetStitcher (Hann windowing, overlapping tile inference)
  - `animaloc/eval/lmds.py` -- HerdNetLMDS (Local Maxima Detection and Suppression)
  - `animaloc/datasets/csv.py` -- CSVDataset (auto-detects points/boxes from CSV headers)
  - `tools/patcher.py` -- Tiling large orthomosaics into training patches
  - `tools/train.py` -- Main Hydra-based training entry point
  - `configs/demo/` -- Config examples (4-level Hydra tree)

## Modes

### 1. train-from-scratch
Full pipeline: data preparation, config creation, training, evaluation. Walks through CSV annotation format, patcher usage, Hydra config construction, Trainer parameters, and HerdNetEvaluator output interpretation.

### 2. fine-tune
Start from a pretrained checkpoint. Covers `load_from` vs `resume_from`, differential learning rates (`backbone_lr` vs `head_lr`), `freeze_backbone_completely()`, and `reshape_classes()` for new species.

### 3. inference-only
Run a trained model on new orthomosaics. Covers HerdNetStitcher (tile size, overlap, Hann windowing, reduction='mean'), HerdNetLMDS (kernel_size, adapt_ts, neg_ts), and CSV output of detections.

### 4. diagnose-training
Debug training failures. Covers: loss not decreasing (check down_ratio mismatch, FIDT params, class weights), F1 plateau (check matching_radius, LMDS kernel_size, adapt_ts), overfitting (DLA-34 vs DLA-60+), NaN losses (check learning rate, AMP compatibility).

### 5. explain-concept
Explain any HerdNet/animaloc concept in depth. Includes FIDT map generation, Hann window stitching, LMDS algorithm, the two-head architecture, CSVDataset auto-detection, Hydra interpolation.

### 6. create-exercise
Design a hands-on exercise for a specific concept. Includes starter code, expected output, common mistakes, and grading rubric.

### 7. full-course-module
Build a complete teaching module combining explanation, worked examples, exercises, and assessment for a topic area.

## Agent Team

| Agent | File | Role |
|-------|------|------|
| Intake | `agents/intake_agent.md` | Determines what user has, routes to appropriate mode and agents |
| Data Prep | `agents/data_prep_agent.md` | CSV format, patcher.py usage, tile sizing, annotation validation |
| Backbone Selection | `agents/backbone_selection_agent.md` | DLA-34/60/102/169 vs DINOv2 vs ConvNext comparison with thesis benchmarks |
| Hydra Config | `agents/hydra_config_agent.md` | Constructs valid 4-level Hydra YAML tree with interpolation dependencies |
| Training | `agents/training_agent.md` | Trainer params, W&B integration, log interpretation, failure diagnosis |
| Inference | `agents/inference_agent.md` | HerdNetStitcher, LMDS, full inference loop on orthomosaics |
| Evaluation | `agents/evaluation_agent.md` | HerdNetEvaluator output, matching_radius, cross-run comparison |
| Exercise Designer | `agents/exercise_designer_agent.md` | Course exercises with starter code and rubrics |

## Routing Logic

```
User query
  |
  v
[Intake Agent] -- analyzes query
  |
  +-- "I have images and want to train" --> Data Prep -> Backbone Selection -> Hydra Config -> Training -> Evaluation
  +-- "I have a trained model"          --> Inference (+ Evaluation if test set available)
  +-- "My training is failing"          --> Training (diagnose mode) -> possibly Hydra Config (fix)
  +-- "Explain how X works"             --> relevant reference doc + explain-concept mode
  +-- "Create an exercise about X"      --> Exercise Designer
  +-- "Build a course module on X"      --> Exercise Designer (full-course-module mode)
  +-- "Compare backbones"               --> Backbone Selection
  +-- "Help me set up configs"          --> Hydra Config
```

## Failure Paths

### Data failures
- **CSV format wrong**: CSVDataset expects headers `images, x, y, labels` for points or `images, x_min, y_min, x_max, y_max, labels` for boxes. Labels must be 1-indexed integers. Route to Data Prep agent.
- **num_classes mismatch**: CRITICAL -- `num_classes` includes background. Binary iguana detection requires `num_classes=3` (background + iguana + hard_negative). This is the single most common configuration error.
- **Annotation coordinates out of bounds**: Points outside image dimensions after cropping. Check patcher output and ObjectAwareRandomCrop behavior.

### Config failures
- **Hydra interpolation errors**: `${model.kwargs.down_ratio}` must resolve. Verify the 4-level config tree (main -> model/, datasets/, losses/, training_settings/) loads correctly.
- **down_ratio propagation**: down_ratio must be consistent across model, FIDT transform, stitcher, and visualizer. Use `${model.kwargs.down_ratio}` interpolation everywhere.
- **Loss weight dimension mismatch**: CrossEntropyLoss weight list length must equal `num_classes` (including background).

### Training failures
- **Loss is NaN**: Usually learning rate too high, or FIDT targets contain invalid values. Check `lr`, `backbone_lr`, `warmup_iters`.
- **F1 score stuck at 0**: matching_radius too small (default 25px is often wrong; optimal is 75px for iguana data). Check LMDS `adapt_ts` and `kernel_size`.
- **Overfitting**: DLA-60/102/169 all overfit on iguana data. Switch to DLA-34. Reduce `head_conv` from 128 to 64.
- **Memory overflow**: Reduce batch_size, use `down_ratio=4` instead of 2, enable gradient checkpointing for DINOv2.

### Inference failures
- **Stitcher output is blank**: Check that `up=False` when using HerdNetStitcher with `reduction='mean'`. The stitcher handles upsampling internally.
- **Double-counting at tile borders**: Increase overlap (recommended: 120px) and use `reduction='mean'` with Hann windowing.
- **LMDS returns no detections**: `adapt_ts` too high or `neg_ts` too high. Start with `adapt_ts=0.5`, `kernel_size=(5,5)`.

## Reference Documents
See `references/` directory for detailed technical documentation on each component.

## Templates
See `templates/` directory for ready-to-use config files, scripts, and report formats.

## Canonical Benchmarks (Miesner 2025 Thesis)

These are the verified optimal parameters and results that students should compare against:

| Parameter | Optimal Value | Default Value | Notes |
|-----------|--------------|---------------|-------|
| down_ratio | 4 | 2 | Reduces memory, improves generalization |
| LMDS kernel_size | (5, 5) | (3, 3) | Better for iguana spacing |
| LMDS adapt_ts | 0.5 | 0.3 | Higher threshold reduces FP |
| matching_radius | 75px | 25px | Critical for correct F1 computation |
| FocalLoss beta | 5 | 4 | Penalizes background more |
| weight_decay | 3.25e-4 | 1.6e-4 | Stronger regularization |
| background_class_weight | 0.1 | 0.1 | Low weight for background in CE loss |
| class_weight (iguana) | 4-5 | 1.0 | Compensates class imbalance |
| Backbone | DLA-34 | DLA-34 | DLA-60/102/169 overfit |

| Island | F1 | Precision | Recall | Epochs |
|--------|-----|-----------|--------|--------|
| Floreana | 0.934 | -- | -- | 8-11 |
| Fernandina | 0.843 | -- | -- | 8-11 |

Key findings:
- Body-center annotations outperform head annotations by F1 = +0.10
- Cross-island training degrades performance -- train per island
- Learning curves plateau at 2000-2500 annotations
- Pix4D > DroneDeploy for orthomosaic quality (F1 delta ~0.07)
