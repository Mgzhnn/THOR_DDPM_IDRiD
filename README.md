# THOR_DDPM_IDRiD

Adaptation of THOR_DDPM for anomaly detection in retinal fundus images with diffusion reconstruction.

## Upstream Credit
This project is adapted from the original **THOR_DDPM** repository by **Cosmin I. Bercea, Benedikt Wiestler, Daniel Rueckert, and Julia Schnabel**.

The core method and original implementation are from the upstream authors; this repository adds a retinal fundus configuration and evaluation workflow.

Upstream source: https://github.com/ci-ber/THOR_DDPM

## Example Output
![THOR_DDPM_IDRiD sample](assets/Thor_IDRID.png)

The example images on this page were produced before the 2026-10-21 fix to the reconstruction range
(reconstructions were squeezed into [0.5, 1], hence the washed-out colours). Regenerate them by re-running the evaluation.

## Repository Layout
```
main.py                      entry point (loads the YAML config, runs train or test)
configs/thor_fundus.yaml     the one experiment config
requirements.txt
thor/                        the Python package
  config.py                  YAML validation, dynamic class import, seeding
  configurator.py            wires model, trainer, data and evaluators from the config
  data/
    dataset.py               DefaultDataset / DefaultDataLoader (CSV-driven)
    fundus.py                FundusLoader: RGB read, resize, flips/affine augmentation
    transforms.py            image transforms shared by the loaders
    csv_utils.py             CSV split helpers
  models/
    ddpm.py                  DDPM wrapper with the THOR inference (harmonization) path
    unet.py                  diffusion U-Net
    schedulers/              base scheduler, DDPM and DDIM schedulers
    simplex_noise.py         simplex noise generator
    vgg.py                   VGG encoder for perceptual loss
  losses/
    ln.py                    L1 / L2
    image.py                 perceptual, SSIM-style and NCC losses
  training/
    base.py                  generic training loop, checkpoints, early stopping
    ddpm_trainer.py          DDPMTrainer: the diffusion training step and validation
  evaluation/
    base.py                  evaluator interface
    fundus_evaluator.py      FundusEvaluator: scores, threshold, metrics, figures
notebooks/fov_mask_morphology.ipynb   field-of-view mask exploration
assets/                      figures used in this README
```

Classes are referenced from the config by module and class name, for example
`thor.training.ddpm_trainer` / `DDPMTrainer` and `thor.evaluation.fundus_evaluator` / `FundusEvaluator`.
Everything from the upstream framework that this project does not use (other datasets' loaders,
the DDIM and anomaly trainers, the Atlas and X-ray evaluators, discriminators, standalone inferers) was removed.

Key config fields: `experiment.task`, `experiment.weights`, `trainer.params.checkpoint_path`,
`model.params.inference_type`, `model.params.noise_level_recon`, `model.params.t_harmonization`.

## Setup
1. Create environment:
```bash
conda create -n thor python=3.8
conda activate thor
```
2. Install PyTorch for your system (CPU/CUDA).
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Optional: install an `xformers` build that matches your installed PyTorch for memory-efficient attention.
   The U-Net falls back to standard attention without it.

Logging is local (`weights/`, `results/`). `wandb` is not used.

## Data Preparation
The dataset is not included. `IDRiD` in the repository name and in the data paths is historical; any retinal fundus images in this layout work.

Use `configs/thor_fundus.yaml` and prepare these CSV splits:
- `data/IDRiD_normalize/splits/train_normal.csv`
- `data/IDRiD_normalize/splits/train_idrid_only.csv`
- `data/IDRiD_normalize/splits/val_normal.csv`
- `data/IDRiD_normalize/splits/test_normal.csv`
- `data/IDRiD_normalize/splits/test_DR.csv`

CSV format:
```csv
filename
/path/to/image_001.png
/path/to/image_002.png
```

Label rule in evaluator:
- path containing `/normal/` or `/good/` -> `label=0`
- otherwise -> `label=1`

The evaluator logs how many images got each label; check it, because a split whose paths lack those folders is labelled all anomalous.
`train_normal.csv` and `train_idrid_only.csv` must contain healthy images only, and no image may appear in both a train and a test split.

## Run
Run from the repository root so that the `thor` package is importable:
```bash
python main.py --config_path configs/thor_fundus.yaml
```

For training:
- set `experiment.task: train`
- set `experiment.weights: null` to train from scratch, or a checkpoint path to fine-tune from it
- set `experiment.resume: true` only to continue an interrupted run (restores the optimizer and starts after the saved epoch)

For evaluation:
- set `experiment.task: test`
- set `experiment.weights: <path_to_best_model.pt>`

## Anomaly Score
- The THOR reconstruction is compared with the input inside a field-of-view mask; the score is the masked mean squared error.
  THOR's anomaly maps are used for harmonization and visualization, not for the score.
- The decision threshold is the 95th percentile of validation normals (`val_normal.csv`, `val_data_loader` in the config).
  Without validation data it falls back to the test normals, and `global_metrics.txt` warns that `Pred_Label` is then optimistic.

## Outputs
Checkpoint folder:
- `weights/thor/idrid/<timestamp>/`
- includes `latest_model.pt`, `best_model.pt`, `training.log`, `Val_epoch_*.png`
- validation runs every `val_interval` epochs and after the last epoch; `best_model.pt` has the lowest validation loss

Evaluation folder:
- `results/<timestamp>/`
- `per_image_scores.csv` (always saved; `Pred_Label` and `Correct` appear only when threshold is available)
- `global_metrics.txt` (always saved; AUROC/AUPRC only when both classes exist; states whether the threshold came from validation or test normals)
- `roc_curve.png` (only when both classes exist)
- `score_overlap.png` (only when overlap is computable)
- `visualizations/*.png`

## Metric Examples
ROC curve example:

![ROC curve example](assets/roc_curve_metric.png)

Anomaly score distribution (custom analysis example):

![Anomaly score distribution example](assets/anomaly_score_distribution_metric.png)

## Notes
- Data, checkpoints, and results are git-ignored.
