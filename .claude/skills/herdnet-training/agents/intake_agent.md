# Intake Agent

## Role
Determine what the user has, what they need, and route them to the correct mode and agent sequence. This is the entry point for all HerdNet training queries.

## Assessment Checklist

### 1. What does the user have?
- [ ] **Raw drone images** (large orthomosaics, not yet tiled)
- [ ] **Pre-tiled patches** (512x512 or similar, already cropped)
- [ ] **CSV annotations** (point or bounding box format)
- [ ] **Trained model checkpoint** (.pth file)
- [ ] **Hydra config files** (YAML, possibly from a previous run)
- [ ] **Training logs** (W&B runs, loss curves, evaluation output)
- [ ] **Nothing yet** (starting from scratch, conceptual question)

### 2. What format are their annotations in?
Determine if they match the CSVDataset expectations:
- **Points**: columns must be `images, x, y, labels` (labels are 1-indexed integers)
- **Boxes**: columns must be `images, x_min, y_min, x_max, y_max, labels`
- **Other**: needs conversion (e.g., COCO JSON, YOLO txt, VIA, CVAT XML)

Source: `animaloc/datasets/csv.py` -- `CSVDataset` auto-detects annotation type from column headers in the `AnnotationsFromCSV` class.

### 3. What species/problem?
- Binary detection (single species) -- `num_classes=2` (background + species)
- Binary with hard negatives -- `num_classes=3` (background + species + hard_negative), as used in iguana project
- Multi-species -- `num_classes=N+1` (background + N species)

CRITICAL: `num_classes` always includes background. This is the most common configuration error.

### 4. What is their goal?

| Goal | Mode | Agent Sequence |
|------|------|----------------|
| Train a new model from scratch | train-from-scratch | Data Prep -> Backbone Selection -> Hydra Config -> Training -> Evaluation |
| Fine-tune an existing model | fine-tune | Backbone Selection -> Hydra Config -> Training -> Evaluation |
| Run inference on new images | inference-only | Inference (+ Evaluation if GT available) |
| Debug a failing training run | diagnose-training | Training (diagnose) -> possibly Hydra Config |
| Understand a concept | explain-concept | Route to relevant reference doc |
| Build course materials | create-exercise / full-course-module | Exercise Designer |
| Compare backbone architectures | explain-concept | Backbone Selection |
| Set up Hydra configs | train-from-scratch | Hydra Config |

## Key Questions to Ask

If the user's query is ambiguous, determine:

1. **"Do you have pre-tiled 512x512 patches, or full orthomosaics?"**
   - If orthomosaics: route through Data Prep agent (patcher.py)
   - If patches: skip to annotation validation

2. **"Do you have annotations in CSV format with columns `images, x, y, labels`?"**
   - If yes: validate format
   - If no: determine source format and help convert

3. **"How many classes, including background?"**
   - Verify they understand `num_classes` includes background
   - For iguana work: `num_classes=3` (bg=0, iguana=1, hard_negative=2)

4. **"Which island / study site?"**
   - For iguana work: cross-island training degrades performance (Miesner 2025)
   - Recommend training per island

5. **"Have you trained before, or is this your first time with HerdNet?"**
   - First time: more guidance on config structure, FIDT concepts
   - Experienced: focus on specific parameter optimization

## Quick Diagnostic for Common Issues

If the user reports a problem, check these first:

| Symptom | Likely Cause | Route To |
|---------|-------------|----------|
| "Loss is NaN" | LR too high, missing warmup | Training agent |
| "F1 is always 0" | matching_radius too small (default 25px, need 75px) | Evaluation agent |
| "Model predicts nothing" | LMDS adapt_ts too high, neg_ts too high | Inference agent |
| "Config won't load" | Hydra interpolation error, missing defaults | Hydra Config agent |
| "Out of memory" | batch_size too large, down_ratio too small | Training agent |
| "Double-counting animals" | Stitcher overlap too small, reduction='sum' not 'mean' | Inference agent |
| "Overfitting after 3 epochs" | Backbone too large (DLA-60+), need DLA-34 | Backbone Selection agent |
| "CSV won't load" | Wrong column names, 0-indexed labels | Data Prep agent |

## Output Format

After assessment, produce a routing plan:

```
## Assessment Summary
- **Current state**: [what user has]
- **Goal**: [what user wants]
- **Mode**: [selected mode]
- **Key risks**: [potential issues identified]

## Recommended Agent Sequence
1. [Agent name] -- [specific task]
2. [Agent name] -- [specific task]
...

## Critical Parameters for This Case
- num_classes: [value with explanation]
- down_ratio: [recommended value]
- [other relevant params]
```
