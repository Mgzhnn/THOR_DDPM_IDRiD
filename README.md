# THOR_DDPM (IDRiD-focused)

This repository is a trimmed-down copy of the THOR_DDPM codebase adapted for IDRiD downstream evaluation.

## Attribution
This work is based on the original THOR_DDPM repository by Cosmin I. Bercea, Benedikt Wiestler, Daniel Rueckert, and Julia Schnabel:
https://github.com/ci-ber/THOR_DDPM

## Citation
If you use this work, please cite the original paper:

```
@misc{Bercea2024diffusion,
    title={Diffusion Models with Implicit Guidance for Medical Anomaly Detection},
    author={Cosmin I. Bercea and Benedikt Wiestler and Daniel Rueckert and Julia Schnabel},
    year={2024},
    month={3},
    eprint={2403.08464},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Setup and Run (IDRiD)
These steps mirror the upstream README, adapted for this IDRiD-focused copy.

1) Create a Python environment (optional):
```
conda create --name thor python=3.8
conda activate thor
```

2) Install PyTorch (pick a CUDA or CPU build that matches your system).

3) Install requirements:
```
pip install -r pip_requirements.txt
```

4) (Optional) Set up Weights & Biases (https://docs.wandb.ai/quickstart):
```
wandb login
```
Sign up for a free account and paste your API key when prompted. You can skip this if you are not logging to wandb.

5) Prepare your data and CSV splits:
   - Place your data in a local folder and update the paths in your config YAML.
   - Ensure any CSV split files you use point to existing files.
   - Datasets are not included in this repo.

6) Run the pipeline:
```
python core/Main.py --config_path projects/thor/configs/IDRiD/thor.yaml
```

## Notes
- Upstream repository does not include a top-level license. This copy is intended for private use unless explicit permission is granted.
- Datasets, results, and checkpoints are excluded from version control.
