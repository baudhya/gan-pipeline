# GAN Pipeline

Production-grade SAR→EO image translation pipeline built on **pix2pix** with a **U-Net generator**, **multi-scale PatchGAN discriminator**, and **hinge loss**. Supports unconditional DCGAN training as well.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [U-Net Generator](#u-net-generator)
   - [Multi-Scale PatchGAN Discriminator](#multi-scale-patchgan-discriminator)
   - [Loss Functions](#loss-functions)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Configuration System](#configuration-system)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Inference](#inference)
10. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
11. [Docker](#docker)
12. [Running Tests](#running-tests)
13. [Code Reference](#code-reference)

---

## Overview

This pipeline translates **SAR (Synthetic Aperture Radar)** images to **EO (Electro-Optical / optical)** images using a conditional GAN. SAR sensors image the Earth using microwave radar, producing grayscale amplitude images regardless of cloud cover or lighting. EO sensors capture visible-light images, which are more interpretable but affected by weather and time of day. The model learns to synthesize a plausible optical appearance from a radar input.

**Why pix2pix?**
Pix2pix is the standard conditional image-to-image translation framework. Unlike unpaired methods (CycleGAN), it trains on aligned SAR/EO pairs, which are available from satellite sources such as Sentinel-1 (SAR) + Sentinel-2 (optical).

**What makes this production-grade?**
- All hyperparameters are externalized to YAML via [Hydra](https://hydra.cc/) — no hardcoded values
- Experiment tracking via [MLflow](https://mlflow.org/) out of the box
- Multi-scale discriminator for both local texture and global structure discrimination
- Synchronized data augmentation across SAR/EO pairs
- Full pytest suite with parametrized coverage of shapes, loss types, and scales
- Docker + docker-compose for reproducible deployment
- GitHub Actions CI pipeline (lint → typecheck → test)

---

## Architecture

### U-Net Generator

**File:** `src/gan_pipeline/models/unet.py`  
**Class:** `UNetGenerator(in_channels, out_channels, base_features=64)`

The generator is an **8-level U-Net** with skip connections. It takes a SAR image as input and produces a synthetic EO image.

```
Input SAR  (1 or 3 ch, 256×256)
     │
  ┌──┴──────────────────────────────────────────────────┐
  │  Encoder (8 blocks, each halves spatial dims)        │
  │                                                      │
  │  enc1: (in_ch → 64,  256→128)  no BatchNorm         │
  │  enc2: (64 → 128,    128→64)                         │
  │  enc3: (128 → 256,    64→32)                         │
  │  enc4: (256 → 512,    32→16)                         │
  │  enc5: (512 → 512,    16→8)                          │
  │  enc6: (512 → 512,     8→4)                          │
  │  enc7: (512 → 512,     4→2)                          │
  │  enc8: (512 → 512,     2→1)   no BatchNorm           │
  │                   (bottleneck)                       │
  └──────────────────────────────────────────────────────┘
     │ (512, 1, 1) + skip connections from each encoder level
  ┌──┴──────────────────────────────────────────────────┐
  │  Decoder (8 blocks, each doubles spatial dims)       │
  │                                                      │
  │  dec1: ConvT(512→512,    1→2)   Dropout 0.5         │ ← concat e7 → 1024 ch
  │  dec2: ConvT(1024→512,   2→4)   Dropout 0.5         │ ← concat e6
  │  dec3: ConvT(1024→512,   4→8)   Dropout 0.5         │ ← concat e5
  │  dec4: ConvT(1024→512,   8→16)                      │ ← concat e4
  │  dec5: ConvT(1024→256,  16→32)                      │ ← concat e3
  │  dec6: ConvT(512→128,   32→64)                      │ ← concat e2
  │  dec7: ConvT(256→64,   64→128)                      │ ← concat e1
  │  out:  ConvT(128→out_ch, 128→256)  Tanh             │
  └──────────────────────────────────────────────────────┘
     │
  Output EO  (3 ch, 256×256, range [-1, 1])
```

**Key design choices:**
- Encoder uses `Conv2d(4×4, stride=2)` + `BatchNorm` + `LeakyReLU(0.2)`
- The bottleneck (`enc8`) has no BatchNorm, following the original pix2pix paper
- Decoder uses `ConvTranspose2d(4×4, stride=2)` + `BatchNorm` + `ReLU` (non-inplace to avoid corrupting skip-connection gradients)
- First 3 decoder blocks apply `Dropout(0.5)` to encourage stochastic output and prevent mode collapse
- Skip connections concatenate encoder features with decoder features at matching resolutions — this is what allows the model to preserve fine spatial structure (edges, textures) while the bottleneck captures global context
- Weights initialized from `N(0, 0.02)` following pix2pix convention

**Parameter count (default config):** ~54 million

---

### Multi-Scale PatchGAN Discriminator

**Files:**
- `src/gan_pipeline/models/patchgan.py` — single-scale PatchGAN
- `src/gan_pipeline/models/multiscale_disc.py` — multi-scale wrapper

**Classes:**
- `PatchGANDiscriminator(sar_channels, eo_channels, base_features=64)`
- `MultiScaleDiscriminator(sar_channels, eo_channels, base_features=64, n_scales=3)`

#### Why PatchGAN?

A standard discriminator outputs a single real/fake scalar per image. PatchGAN instead outputs a **grid of scalars**, each representing whether a corresponding patch of the image is real or fake. The discriminator only sees local context within a receptive field.

This has three advantages:
1. **Fewer parameters** than a full-image discriminator
2. **Sharper textures** — the generator is penalized locally, not just globally
3. **Works at any resolution** — the patch-level objective is scale-independent

#### 70×70 PatchGAN Architecture

Input: `cat([SAR, fake_EO])` or `cat([SAR, real_EO])` — shape `(B, sar_ch+eo_ch, H, W)`

```
L1: Conv(in_ch, 64,   4×4 s=2 p=1)  LReLU(0.2)  no BN   256→128
L2: Conv(64,    128,  4×4 s=2 p=1)  BN  LReLU(0.2)       128→64
L3: Conv(128,   256,  4×4 s=2 p=1)  BN  LReLU(0.2)        64→32
L4: Conv(256,   512,  4×4 s=1 p=1)  BN  LReLU(0.2)        32→31
L5: Conv(512,   1,    4×4 s=1 p=1)                         31→30
```

Output: `(B, 1, 30, 30)` — each of the 900 values scores one ~70×70 patch.

The 70×70 receptive field is calculated as:
`RF = 1 + (4-1)×1 + (4-1)×1 + (4-1)×2 + (4-1)×4 + (4-1)×8 = 70`

#### Multi-Scale Discriminator

**File:** `src/gan_pipeline/models/multiscale_disc.py`

The `MultiScaleDiscriminator` runs N independent `PatchGANDiscriminator` instances on N progressively downsampled versions of the input:

```
Input pair (B, 4, 256, 256)
     │
     ├──→  D₀  (256×256) → patch map (~30×30)   fine detail
     │
    AvgPool(3, s=2)
     │
     ├──→  D₁  (128×128) → patch map (~14×14)   mid-level structure
     │
    AvgPool(3, s=2)
     │
     └──→  D₂   (64×64)  → patch map ( ~6×6)    global layout
```

- Each `Dᵢ` has its own independent weights
- `AvgPool2d(kernel=3, stride=2, padding=1, count_include_pad=False)` — smooth downsampling avoids aliasing artifacts
- Losses from all scales are averaged: `L_D = mean(L_D₀, L_D₁, L_D₂)`

**Why multi-scale?**
With a single 70×70 PatchGAN on 256×256 images, the discriminator only evaluates small patches and can miss large-scale coherence (e.g., building layout, road network orientation). The coarser-scale discriminators fill this gap while the fine-scale discriminator preserves texture quality.

Configure the number of scales with `model.discriminator.n_scales` (default: 3).

---

### Loss Functions

**File:** `src/gan_pipeline/models/losses.py`

Three loss types are supported, selectable via `training.loss_type`.

#### Hinge Loss (default)

The **recommended** loss for pix2pix with PatchGAN. Derived from the SVM margin objective.

```
L_D = mean(relu(1 − real_logits)) + mean(relu(1 + fake_logits))
L_G = −mean(fake_logits)
```

- Real logits ≥ 1 contribute zero discriminator loss (they're already confident)
- Fake logits ≤ −1 contribute zero discriminator loss (also confident)
- The generator maximizes discriminator output without a saturating ceiling
- Tends to produce stable gradients throughout training

#### BCE Loss

Standard binary cross-entropy. Real patches are labeled 1, fake patches are labeled 0.

```
L_D = 0.5 × (BCE(real_logits, 1) + BCE(fake_logits, 0))
L_G = BCE(fake_logits, 1)
```

Can suffer from vanishing gradients when the discriminator is too confident early in training.

#### Wasserstein Loss

```
L_D = mean(fake_logits) − mean(real_logits)
L_G = −mean(fake_logits)
```

Requires gradient clipping or gradient penalty (toggle with `training.gradient_penalty: true`, weight with `training.gradient_penalty_lambda`).

#### Multi-Scale Wrappers

```python
multiscale_discriminator_loss(real_maps_list, fake_maps_list, loss_type)
multiscale_generator_loss(fake_maps_list, loss_type)
```

Both take the list of patch maps from `MultiScaleDiscriminator.forward()` and return a single scalar — the mean loss across all scales.

#### Full Generator Loss

```
L_G_total = L_G_adv + λ_L1 × L1(fake_EO, real_EO)
```

`λ_L1 = 100` by default. The L1 term pushes the generator toward the correct pixel values; the adversarial term sharpens the result beyond what pixel-level regression alone would produce.

---

## Project Structure

```
gan-pipeline/
│
├── src/gan_pipeline/               # Installable Python package
│   ├── data/
│   │   ├── dataset.py              # Standard ImageFolder dataloader (DCGAN)
│   │   ├── paired_dataset.py       # SAR/EO paired dataset (side-by-side or separate dirs)
│   │   └── transforms.py           # torchvision transform pipeline
│   │
│   ├── models/
│   │   ├── base.py                 # BaseGenerator, BaseDiscriminator ABCs
│   │   ├── dcgan.py                # DCGAN generator + discriminator (unconditional)
│   │   ├── unet.py                 # U-Net generator (conditional)
│   │   ├── patchgan.py             # 70×70 PatchGAN discriminator
│   │   ├── multiscale_disc.py      # Multi-scale PatchGAN wrapper
│   │   └── losses.py               # BCE, Wasserstein, Hinge; multi-scale helpers
│   │
│   ├── training/
│   │   ├── trainer.py              # GANTrainer — unconditional DCGAN training loop
│   │   └── pix2pix_trainer.py      # Pix2PixTrainer — conditional SAR→EO loop
│   │
│   ├── evaluation/
│   │   └── metrics.py              # FID and Inception Score (via torch-fidelity)
│   │
│   ├── inference/
│   │   └── generate.py             # Load checkpoint, generate and save images
│   │
│   └── utils/
│       ├── checkpointing.py        # save_checkpoint / load_checkpoint
│       └── logging.py              # Loguru setup (stderr + rotating file)
│
├── configs/                        # Hydra configuration tree
│   ├── config.yaml                 # Root config: selects model/training/data defaults
│   ├── model/
│   │   ├── pix2pix.yaml            # U-Net + multi-scale PatchGAN (DEFAULT)
│   │   └── dcgan.yaml              # DCGAN architecture
│   ├── training/
│   │   ├── pix2pix.yaml            # pix2pix hyperparameters (DEFAULT)
│   │   └── default.yaml            # DCGAN hyperparameters
│   └── data/
│       ├── sar_eo.yaml             # SAR→EO dataset (DEFAULT)
│       └── celeba.yaml             # CelebA (DCGAN example)
│
├── scripts/
│   ├── train_pix2pix.py            # Entry point: SAR→EO pix2pix training
│   ├── train.py                    # Entry point: unconditional DCGAN training
│   ├── evaluate.py                 # Compute FID / IS from a checkpoint
│   └── generate.py                 # Generate images from a DCGAN checkpoint
│
├── tests/
│   ├── conftest.py                 # Shared fixtures (device, cfg)
│   ├── test_data.py                # Transform shape/range tests
│   ├── test_models.py              # DCGAN shapes, losses, gradient penalty
│   ├── test_pix2pix.py             # U-Net, PatchGAN, multi-scale, dataset, trainer
│   └── test_training.py            # Checkpoint save/load, DCGAN trainer step
│
├── docker/
│   ├── Dockerfile                  # PyTorch CUDA runtime image
│   └── docker-compose.yml          # Services: train, mlflow, generate
│
├── .github/workflows/ci.yml        # GitHub Actions: lint → typecheck → pytest
├── pyproject.toml                  # Package metadata, deps, tool config
├── Makefile                        # Common dev commands
└── .env.example                    # Environment variable template
```

---

## Installation

**Requirements:** Python ≥ 3.10

```bash
# Clone the repo
git clone <repo-url>
cd gan-pipeline

# Install with dev dependencies
pip install -e ".[dev]"

# For FID/IS evaluation (optional)
pip install -e ".[eval]"
```

For **CPU-only** environments (e.g. CI, development machines without GPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

For **CUDA** (production training):

```bash
pip install torch torchvision  # picks up CUDA automatically
pip install -e ".[dev]"
```

---

## Dataset Preparation

### Format 1: Side-by-side (default)

The standard pix2pix format. Each file is a single image where the **left half is SAR** and the **right half is EO**. Files can be `.jpg`, `.png`, `.tif`, or `.tiff`.

```
data/sar_eo/
├── train/
│   ├── 00001.png    # 512×256: [SAR_256×256 | EO_256×256]
│   ├── 00002.png
│   └── ...
└── val/
    ├── 00001.png
    └── ...
```

To use this format (already the default):

```yaml
# configs/data/sar_eo.yaml
dataset_format: side_by_side
```

### Format 2: Separate directories

SAR and EO images in separate folders. **Filenames must match** across both directories.

```
data/sar_eo/
├── trainA/          # SAR images
│   ├── 00001.png
│   └── ...
├── trainB/          # EO images (same filenames as trainA)
│   ├── 00001.png
│   └── ...
├── valA/
└── valB/
```

To use this format:

```yaml
# configs/data/sar_eo.yaml  (or override on command line)
dataset_format: separate_dirs
```

### SAR channels

| `sar_channels` | Interpretation |
|---|---|
| `1` (default) | Single-polarization (e.g. Sentinel-1 VV or VH) |
| `3` | Multi-polarization (HH, HV, VV stacked as 3-channel image) |

Update in `configs/data/sar_eo.yaml` or override at runtime:

```bash
python scripts/train_pix2pix.py data.sar_channels=3
```

### Data augmentation (training only)

Applied automatically during training, synchronized across SAR and EO:

1. Resize to `image_size × 1.12` (≈286×286 for 256 target)
2. Random crop back to `image_size × image_size`
3. Random horizontal flip (50% probability)
4. Normalize to `[-1, 1]` using `mean=0.5, std=0.5`

Disabled for validation/inference (`augment=false`).

---

## Configuration System

This project uses [Hydra](https://hydra.cc/) for configuration. All hyperparameters live in YAML files under `configs/`. The active config is assembled at runtime from the defaults declared in `configs/config.yaml`.

### Default config (`configs/config.yaml`)

```yaml
defaults:
  - model: pix2pix        # loads configs/model/pix2pix.yaml
  - training: pix2pix     # loads configs/training/pix2pix.yaml
  - data: sar_eo          # loads configs/data/sar_eo.yaml
  - _self_

experiment_name: sar_eo_pix2pix
seed: 42
output_dir: outputs/${experiment_name}
resume: null
```

### Overriding from the command line

Any config key can be overridden at runtime using `key=value` syntax:

```bash
# Change number of discriminator scales
python scripts/train_pix2pix.py model.discriminator.n_scales=2

# Change loss type
python scripts/train_pix2pix.py training.loss_type=bce

# Change batch size and learning rate
python scripts/train_pix2pix.py training.batch_size=4 training.lr_generator=0.0001

# Use a different data root
python scripts/train_pix2pix.py data.root=/path/to/my/dataset

# Switch to DCGAN (unconditional) config group
python scripts/train.py model=dcgan training=default data=celeba
```

### Key config parameters

#### `configs/model/pix2pix.yaml`

| Key | Default | Description |
|---|---|---|
| `model.generator.base_features` | `64` | Base channel count for U-Net; doubled at each encoder level up to 512 |
| `model.discriminator.base_features` | `64` | Base channel count for each PatchGAN |
| `model.discriminator.n_scales` | `3` | Number of discriminator scales |

#### `configs/training/pix2pix.yaml`

| Key | Default | Description |
|---|---|---|
| `training.epochs` | `200` | Total training epochs |
| `training.batch_size` | `1` | Batch size (pix2pix typically uses 1) |
| `training.lr_generator` | `0.0002` | Generator Adam learning rate |
| `training.lr_discriminator` | `0.0002` | Discriminator Adam learning rate |
| `training.beta1` | `0.5` | Adam β₁ (lower than default 0.9 for GAN stability) |
| `training.beta2` | `0.999` | Adam β₂ |
| `training.loss_type` | `hinge` | Loss function: `hinge`, `bce`, or `wasserstein` |
| `training.lambda_l1` | `100.0` | Weight of pixel-level L1 loss term |
| `training.save_every` | `10` | Save checkpoint every N epochs |
| `training.sample_every` | `5` | Save sample grid every N epochs |
| `training.log_every` | `100` | Log to console every N batches |
| `training.num_workers` | `4` | DataLoader worker processes |

#### `configs/data/sar_eo.yaml`

| Key | Default | Description |
|---|---|---|
| `data.root` | `data/sar_eo` | Path to dataset root |
| `data.image_size` | `256` | Spatial resolution (must be power of 2, ≥ 32) |
| `data.sar_channels` | `1` | SAR input channels (1 or 3) |
| `data.eo_channels` | `3` | EO output channels (3 for RGB) |
| `data.dataset_format` | `side_by_side` | `side_by_side` or `separate_dirs` |
| `data.augment_train` | `true` | Enable crop + flip augmentation for training |

---

## Training

### SAR→EO (pix2pix, recommended)

```bash
python scripts/train_pix2pix.py
```

Outputs are written to `outputs/sar_eo_pix2pix/`:

```
outputs/sar_eo_pix2pix/
├── train.log                        # Full debug log (rotates at 50 MB)
├── samples/
│   ├── epoch_0000.png               # Grid: [SAR | fake EO | real EO]
│   ├── epoch_0005.png
│   └── ...
└── checkpoints/
    ├── epoch_0010.pt
    ├── epoch_0020.pt
    └── ...
```

### Resume from checkpoint

```bash
python scripts/train_pix2pix.py resume=outputs/sar_eo_pix2pix/checkpoints/epoch_0050.pt
```

### Named experiments

```bash
python scripts/train_pix2pix.py \
  experiment_name=run_hinge_3scale \
  training.loss_type=hinge \
  model.discriminator.n_scales=3
```

Each experiment gets its own output directory: `outputs/run_hinge_3scale/`.

### Unconditional DCGAN

```bash
# Train on CelebA (or any ImageFolder-compatible dataset)
python scripts/train.py model=dcgan training=default data=celeba
```

Or with the Makefile shortcut:

```bash
make train-dcgan
```

---

## Evaluation

Compute **FID** (Fréchet Inception Distance) and **IS** (Inception Score) against a held-out set of real EO images.

```bash
# Requires: pip install -e ".[eval]"   (installs torch-fidelity)
python scripts/evaluate.py \
  checkpoint=outputs/sar_eo_pix2pix/checkpoints/epoch_0199.pt \
  real_dir=data/sar_eo/val/B \
  eval_samples=5000
```

Lower FID is better (measures distance between real and generated distributions). Higher IS is better (measures sharpness and diversity).

---

## Inference

Generate images from a trained checkpoint without training infrastructure:

```bash
python scripts/generate.py \
  checkpoint=outputs/sar_eo_pix2pix/checkpoints/epoch_0199.pt \
  n_samples=64 \
  gen_output_dir=outputs/generated
```

Saves a PNG grid to `outputs/generated/generated.png`.

---

## Experiment Tracking with MLflow

Every training run automatically logs to MLflow:

**Logged parameters:** model name, loss type, λ_L1, number of scales, learning rates, batch size  
**Logged metrics per epoch:** `d_loss`, `g_adv`, `g_l1`

### Start the MLflow UI

```bash
mlflow ui --port 5000
# Then open http://localhost:5000
```

Or set a remote tracking server:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
python scripts/train_pix2pix.py
```

---

## Docker

Build and run with docker-compose. The compose file defines three services:

| Service | Description |
|---|---|
| `train` | Runs `scripts/train_pix2pix.py` with GPU access |
| `mlflow` | MLflow tracking server at `http://localhost:5000` |
| `generate` | Runs `scripts/generate.py` (activated with `--profile generate`) |

### Build

```bash
docker compose -f docker/docker-compose.yml build
```

### Train

```bash
docker compose -f docker/docker-compose.yml up train
```

Volumes are mounted so that `data/`, `outputs/`, and `mlruns/` persist on the host.

### Generate images from a trained model

```bash
docker compose -f docker/docker-compose.yml \
  --profile generate run generate \
  checkpoint=outputs/sar_eo_pix2pix/checkpoints/epoch_0199.pt
```

### Environment variables

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

```
MLFLOW_TRACKING_URI=http://localhost:5000
CUDA_VISIBLE_DEVICES=0
```

---

## Running Tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# Single test file
pytest tests/test_pix2pix.py -v

# Run tests matching a pattern
pytest -k "multiscale" -v

# With coverage report
pytest --cov=gan_pipeline --cov-report=term-missing
```

### Test coverage

| File | What's tested |
|---|---|
| `test_data.py` | Transform output shape (32/64/128), pixel range [-1, 1] |
| `test_models.py` | DCGAN generator/discriminator shapes; BCE/Wasserstein/Hinge losses; gradient penalty; `sample()` |
| `test_pix2pix.py` | U-Net output shape (1→3, 3→3, 1→1 ch); skip-connection gradients; PatchGAN patch map shape (~30×30); multi-scale output lengths; decreasing patch sizes across scales; all loss types on multi-scale maps; train step (hinge×3scale, bce×1scale, hinge×2scale); side-by-side dataset loading and augmentation |
| `test_training.py` | Checkpoint save/load round-trip; DCGAN trainer step (loss is finite float) |

### Makefile shortcuts

```bash
make install      # pip install -e ".[dev]"
make test         # pytest
make lint         # ruff + black --check + isort --check
make format       # black + isort + ruff --fix (in-place)
make typecheck    # mypy src/
make train        # python scripts/train_pix2pix.py
make train-dcgan  # python scripts/train.py model=dcgan ...
make clean        # remove __pycache__, .pytest_cache, .mypy_cache, egg-info
```

---

## Code Reference

### Adding a new model

1. Create `src/gan_pipeline/models/mymodel.py` with classes inheriting `BaseGenerator` / `BaseDiscriminator`
2. Export from `src/gan_pipeline/models/__init__.py`
3. Add `configs/model/mymodel.yaml`
4. Instantiate in the training script

### Adding a new loss

Add a new enum value to `LossType` in `losses.py` and handle it in `generator_loss` and `discriminator_loss`. The multi-scale wrappers pick it up automatically.

### Adding a new dataset format

Implement a new `Dataset` subclass in `src/gan_pipeline/data/paired_dataset.py` and add a branch in `get_paired_dataloader`.

### Checkpoint format

Checkpoints are plain PyTorch `.pt` files containing:

```python
{
    "epoch": int,
    "generator": OrderedDict,       # generator.state_dict()
    "discriminator": OrderedDict,   # discriminator.state_dict()
    "opt_g": dict,                  # optimizer state
    "opt_d": dict,
    "metrics": {"d_loss": float, "g_adv": float, "g_l1": float},
}
```

Load with:

```python
from gan_pipeline.utils.checkpointing import load_checkpoint
state = load_checkpoint(Path("epoch_0199.pt"), device)
generator.load_state_dict(state["generator"])
```
