# VisionGuard AI

Production-oriented Week 1 foundation for industrial visual anomaly detection. It uses PatchCore
to learn normal product appearance, then produces anomaly scores and localization maps for unseen
images. The initial configuration targets one MVTec AD 2 category so the four-week MVP remains
achievable.

## Week 1 scope

- Typed, validated YAML configuration
- MVTec AD 2 data integration through Anomalib
- PatchCore training, evaluation, and single-image inference commands
- Structured console/file logging and isolated artifact directories
- Unit tests, linting, type checking, and reproducible dependency bounds
- Git-safe handling of datasets, checkpoints, generated outputs, and secrets

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- NVIDIA GPU with a compatible PyTorch build is recommended, but CPU works for smoke tests

## Setup (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch installation varies by CPU/CUDA platform. If Anomalib's dependency resolution does not
select the desired build, install PyTorch first using the command generated at
https://pytorch.org/get-started/locally/ and then run the editable install.

## Dataset

Download **MVTec AD 2** from its official source and accept its license. Extract it beneath
`data/mvtecad2/`, then confirm that `data.category` in
`configs/patchcore_mvtecad2.yaml` matches an extracted category. The default is `sheet_metal`.
Dataset content is never committed.

## Commands

```powershell
# Validate the lightweight project foundation
python -m pytest
python -m ruff check .
python -m mypy src

# Build the PatchCore memory bank and checkpoint
visionguard train --config configs/patchcore_mvtecad2.yaml

# Evaluate the newest checkpoint
visionguard evaluate --config configs/patchcore_mvtecad2.yaml

# Inspect one image (writes result.json and heatmap_overlay.png)
visionguard predict --config configs/patchcore_mvtecad2.yaml `
  --image path\to\sample.png --output outputs\sample
```

The thin scripts in `scripts/` provide equivalent entry points. Prefer the installed
`visionguard` command for normal use.

## Repository layout

```text
configs/                 Experiment configuration
data/                    Local datasets (ignored)
outputs/                 User-facing predictions (ignored)
scripts/                 Convenience entry points
src/visionguard/         Application package
tests/                   Fast unit tests
artifacts/                Checkpoints, metrics, and logs (ignored)
```

## Reproducibility notes

The YAML file is the source of truth for an experiment. Keep the exact config and Git commit hash
with reported results. PatchCore is primarily a feature-memory method rather than a conventional
gradient-trained network; “training” builds its representative memory bank from normal images.

## Definition of done for Week 1

1. A single MVTec AD 2 category is available locally.
2. `visionguard train` produces a checkpoint under `artifacts/`.
3. `visionguard evaluate` prints image- and pixel-level metrics supplied by Anomalib.
4. `visionguard predict` writes a JSON result for a known good and defective image.
5. Tests and lint checks pass before committing.
