# GAN Pipeline

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/baudhya/gan-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/baudhya/gan-pipeline/actions/workflows/ci.yml)
[![mypy](https://github.com/baudhya/gan-pipeline/actions/workflows/mypy.yml/badge.svg)](https://github.com/baudhya/gan-pipeline/actions/workflows/mypy.yml)
[![ruff](https://github.com/baudhya/gan-pipeline/actions/workflows/ruff.yml/badge.svg)](https://github.com/baudhya/gan-pipeline/actions/workflows/ruff.yml)
[![pytest](https://github.com/baudhya/gan-pipeline/actions/workflows/pytest.yml/badge.svg)](https://github.com/baudhya/gan-pipeline/actions/workflows/pytest.yml)
[![pre-commit](https://github.com/baudhya/gan-pipeline/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/baudhya/gan-pipeline/actions/workflows/pre-commit.yml)

Production-grade SAR→EO image translation pipeline built on **pix2pix** with a **U-Net generator**, **multi-scale PatchGAN discriminator**, **hinge loss**, **VGG perceptual loss**, and **feature matching loss**. Supports unconditional DCGAN training as well.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [U-Net Generator](#u-net-generator)
   - [Multi-Scale PatchGAN Discriminator](#multi-scale-patchgan-discriminator)
   - [Loss Functions](#loss-functions)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
   - [Pre-commit hooks](#pre-commit-hooks)
5. [Data Preparation](#data-preparation)
   - [prepare_data.py — Sentinel-1/2 pipeline](#prepare_datapy--sentinel-12-pipeline)
   - [SAR preprocessing](#sar-preprocessing)
   - [EO preprocessing](#eo-preprocessing)
   - [Training data formats](#training-data-formats)
   - [Data augmentation](#data-augmentation)
   - [Sentinel-1/2 paired data with land-cover classes](#sentinel-12-paired-data-with-land-cover-classes)
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
- End-to-end data pipeline from raw Sentinel GeoTIFFs → training-ready PNGs
- All hyperparameters externalized to YAML via [Hydra](https://hydra.cc/) — no hardcoded values
- Experiment tracking via [MLflow](https://mlflow.org/) out of the box
- Multi-scale discriminator for both local texture and global structure discrimination
- Synchronized data augmentation across SAR/EO pairs
- 68 passing tests with parametrized coverage of shapes, loss types, and scales
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
- Decoder uses `ConvTranspose2d(4×4, stride=2)` + `BatchNorm` + `ReLU` (non-inplace to avoid corrupting skip-connection gradients during backward)
- First 3 decoder blocks apply `Dropout(0.5)` to encourage stochastic output and prevent mode collapse
- Skip connections concatenate encoder features with decoder features at matching resolutions — this preserves fine spatial structure (edges, textures) while the bottleneck captures global context
- Weights initialized from `N(0, 0.02)` following pix2pix convention

**Parameter count (default config):** ~54 million

---

### Multi-Scale PatchGAN Discriminator

**Files:**
- `src/gan_pipeline/models/patchgan.py` — single-scale PatchGAN
- `src/gan_pipeline/models/multiscale_disc.py` — multi-scale wrapper

**Classes:**
- `PatchGANDiscriminator(sar_channels, eo_channels, base_features=64, spectral_norm=False)`
- `MultiScaleDiscriminator(sar_channels, eo_channels, base_features=64, n_scales=3, spectral_norm=False)`

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

#### Spectral Normalisation

When `spectral_norm=True`, `nn.utils.spectral_norm` is applied to every `Conv2d` in the discriminator after weight initialisation. This constrains the Lipschitz constant of each layer, providing cheap gradient stability without requiring a gradient penalty or changes to the loss function.

```
Before SN:  Conv2d.weight  ← N(0, 0.02) parameter
After SN:   Conv2d.weight_orig  ← same initialised parameter
            Conv2d.weight       ← weight_orig / σ_max  (computed per forward pass)
```

SN is applied **after** `_init_weights()` so `weight_orig` retains the pix2pix N(0, 0.02) initialisation. The Python default is `False` (backward compatible); the pix2pix config sets it to `true` for all production runs.

---

### Loss Functions

**File:** `src/gan_pipeline/models/losses.py`

Three adversarial loss types are supported, selectable via `training.loss_type`. VGG perceptual loss and feature matching loss can be layered on top of any of them.

#### Hinge Loss (default)

The **recommended** loss for pix2pix with PatchGAN. Derived from the SVM margin objective.

```
L_D = mean(relu(1 − real_logits)) + mean(relu(1 + fake_logits))
L_G = −mean(fake_logits)
```

- Real logits ≥ 1 contribute zero discriminator loss (already confident)
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

#### VGG Perceptual Loss

**Class:** `VGGPerceptualLoss` in `src/gan_pipeline/models/losses.py`

Computes feature-space distance between the generated and real EO image using a frozen VGG16 backbone pretrained on ImageNet. Features are extracted at four intermediate activations — `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3` — and the L1 distance is summed across all four levels.

```
L_VGG = Σ_{i=1}^{4}  L1( VGG_i(fake_EO),  VGG_i(real_EO) )
```

The VGG network is frozen (`requires_grad=False`) and moved to the training device automatically. Inputs in `[-1, 1]` are rescaled to ImageNet-normalized `[0, 1]` internally. Single-channel (grayscale) SAR inputs passed by accident are expanded to 3 channels; tensors with more than 3 channels are truncated to the first three (RGB).

Setting `lambda_vgg: 0.0` disables the loss entirely without instantiating the VGG network — useful for CI or low-memory environments.

**Offline / air-gapped use:** by default, VGG16 weights are downloaded from PyTorch Hub on first use and cached at `~/.cache/torch/hub/checkpoints/vgg16-397923af.pth`. On machines without internet access, pre-download the file on a connected machine and point the config at it:

```bash
# On a machine with internet — download and locate the cache file
python -c "import torchvision; torchvision.models.vgg16(weights='IMAGENET1K_V1')"
# → ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# Copy to the air-gapped machine, then set in config:
#   training.vgg_weights_path: /path/to/vgg16-397923af.pth
# or pass as a CLI override:
python scripts/train_pix2pix.py training.vgg_weights_path=/path/to/vgg16-397923af.pth
```

When `vgg_weights_path` is set, the file is loaded with `torch.load(..., weights_only=True)`; no network access is required. Leave it as `null` (the default) for the standard online behaviour.

#### Feature Matching Loss

**Function:** `feature_matching_loss` in `src/gan_pipeline/models/losses.py`  
**Methods:** `PatchGANDiscriminator.forward_with_features()`, `MultiScaleDiscriminator.forward_with_features()`

Introduced in Pix2PixHD, feature matching pushes the generator to produce activations inside the discriminator that match those produced by real pairs — giving it a dense, learned training signal beyond the single scalar adversarial loss.

`forward_with_features()` runs the discriminator and returns both the final patch logit map **and** the output of each intermediate conv block:

```
PatchGANDiscriminator.forward_with_features(x)
  → (logit_map,  [feat_block1,  feat_block2,  feat_block3,  feat_block4])
       (B,1,H,W)  (B,64,...)     (B,128,...)    (B,256,...)    (B,512,...)
```

`MultiScaleDiscriminator.forward_with_features(x)` delegates to each scale and returns:

```
(logits_list,  features_per_scale)
 list[Tensor]  list[list[Tensor]]   — outer: scale, inner: conv block
```

The loss averages L1 distance over all `(scale, layer)` pairs:

```
L_FM = mean over (scale s, layer l) of  L1( D_s_l(fake_pair),  D_s_l(real_pair).detach() )
```

Real features are detached so gradients only flow through the fake path to update G. When `lambda_fm > 0`, `forward_with_features` is used for the generator step (one forward pass yields both logits and features); real features are extracted under `torch.no_grad()`. Setting `lambda_fm: 0.0` skips both the feature extraction and the extra discriminator forward pass on the real pair.

#### Full Generator Loss

```
L_G_total = L_G_adv + λ_L1 × L1(fake_EO, real_EO) + λ_VGG × L_VGG(fake_EO, real_EO) + λ_FM × L_FM
```

| Term | Default weight | Role |
|---|---|---|
| `L_G_adv` | 1.0 | Adversarial sharpness signal from multi-scale discriminator |
| `λ_L1 × L1` | 100.0 | Low-frequency fidelity; anchors color and structure |
| `λ_VGG × L_VGG` | 10.0 | Perceptual texture quality; reduces checkerboard and blurry artifacts |
| `λ_FM × L_FM` | 10.0 | Discriminator feature alignment; stabilises training and improves fine detail |

---

## Project Structure

```
gan-pipeline/
│
├── src/gan_pipeline/               # Installable Python package
│   ├── data/
│   │   ├── dataset.py              # Standard ImageFolder dataloader (DCGAN)
│   │   ├── paired_dataset.py       # SAR/EO paired dataset (side-by-side or separate dirs)
│   │   ├── transforms.py           # torchvision transform pipeline
│   │   └── sentinel_utils.py       # Sentinel-1/2 preprocessing utilities (pure numpy)
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
│   ├── prepare_data.py             # Sentinel-1/2 data preparation (argparse CLI)
│   ├── train_pix2pix.py            # Entry point: SAR→EO pix2pix training
│   ├── train.py                    # Entry point: unconditional DCGAN training
│   ├── evaluate.py                 # Compute FID / IS from a checkpoint
│   └── generate.py                 # Generate images from a checkpoint
│
├── tests/
│   ├── conftest.py                 # Shared fixtures (device, cfg)
│   ├── test_data.py                # Transform shape/range tests
│   ├── test_models.py              # DCGAN shapes, losses, gradient penalty
│   ├── test_pix2pix.py             # U-Net, PatchGAN, multi-scale, dataset, trainer
│   ├── test_sentinel_utils.py      # SAR/EO preprocessing: normalize, assemble, validate
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
git clone https://github.com/baudhya/gan-pipeline.git
cd gan-pipeline

# Core install with dev dependencies
pip install -e ".[dev]"

# Geospatial extras — required for prepare_data.py
pip install -e ".[geo]"

# FID/IS evaluation (optional)
pip install -e ".[eval]"
```

For **CPU-only** environments (CI, development machines without GPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,geo]"
```

For **CUDA** (production training):

```bash
pip install torch torchvision  # picks up CUDA automatically
pip install -e ".[dev,geo]"
```

### Optional extras summary

| Extra | Installs | Required for |
|---|---|---|
| `dev` | pytest, black, ruff, mypy, isort, pre-commit | Development and CI |
| `geo` | rasterio, h5py | `scripts/prepare_data.py` |
| `eval` | torch-fidelity | `scripts/evaluate.py` (FID/IS) |

---

### Pre-commit hooks

The repo ships a `.pre-commit-config.yaml` that runs **ruff**, **black**, **isort**, and **mypy** automatically before every `git commit`. Once installed, a commit is rejected if any check fails — the same gates that run in CI.

**One-time setup** (after cloning or after `pip install -e ".[dev]"`):

```bash
pre-commit install
```

That's it. The hooks run on the files you've staged each time you commit.

**Run manually against all files** (useful before opening a PR):

```bash
pre-commit run --all-files
```

**Skip hooks in an emergency** (e.g. a WIP commit you intend to fix up):

```bash
git commit --no-verify -m "wip: ..."
```

**What each hook checks:**

| Hook | Command | Blocks commit if… |
|---|---|---|
| `ruff lint` | `ruff check src tests scripts` | Any lint error (unused import, bad style, etc.) |
| `black format check` | `black --check src tests scripts` | Any file would be reformatted |
| `isort import check` | `isort --check-only src tests scripts` | Imports are not sorted |
| `mypy type check` | `mypy src` | Any type error in `src/` |

To fix formatting issues automatically before committing, run `make format` first, then re-stage and commit.

---

## Data Preparation

### `prepare_data.py` — Sentinel-1/2 pipeline

**File:** `scripts/prepare_data.py`

Converts raw Sentinel-1 (SAR) and Sentinel-2 (EO) GeoTIFF data into the side-by-side PNG format consumed by the training pipeline. Requires the `[geo]` extra (`rasterio`).

#### Two input modes

**`sen12ms`** — Processes the [SEN12MS benchmark dataset](https://mediatum.ub.tum.de/1474000), which ships as pre-cropped 256×256 GeoTIFF patch pairs. This is the recommended starting point for experiments.

```
SEN12MS/
├── s1/
│   └── ROIs1158_spring_s1_1/
│       ├── s1_1.tif      # (2, 256, 256) float32 — VV, VH in dB
│       ├── s1_2.tif
│       └── ...
└── s2/
    └── ROIs1158_spring_s2_1/
        ├── s2_1.tif      # (13, 256, 256) uint16 — all Sentinel-2 bands ×10000
        ├── s2_2.tif
        └── ...
```

Pairs are matched by `(roi_id, season, scene_id, patch_id)`. Unmatched files are skipped silently.

```bash
python scripts/prepare_data.py \
  --mode sen12ms \
  --s1-dir /data/SEN12MS/s1 \
  --s2-dir /data/SEN12MS/s2 \
  --output-dir data/sar_eo \
  --sar-already-db \       # SEN12MS S1 data is already in dB
  --sar-channels 1 \       # VV only; use 2 for VV+VH
  --val-split 0.1 \
  --test-split 0.1
```

**`scenes`** — Chips co-registered Sentinel-1 and Sentinel-2 scenes into 256×256 patches using a sliding window. Both scenes must be in the **same CRS and geographic extent** before running this script. Use [ESA SNAP](https://step.esa.int/main/toolboxes/snap/) or GDAL to coregister first.

```
raw/
├── sentinel1/
│   ├── scene_001.tif     # full S1 GRD scene, float32, linear power scale
│   └── scene_002.tif
└── sentinel2/
    ├── scene_001.tif     # full S2 L2A scene, uint16, bands × 10000
    └── scene_002.tif
```

```bash
python scripts/prepare_data.py \
  --mode scenes \
  --s1-dir /data/raw/sentinel1 \
  --s2-dir /data/raw/sentinel2 \
  --output-dir data/sar_eo \
  --image-size 256 \
  --stride 128 \           # 50% overlap between patches
  --min-valid-fraction 0.9
```

#### Output format

Both modes produce the same output structure:

```
data/sar_eo/
├── train/
│   ├── 000001.png    # 512×256 RGB: [SAR_256×256 | EO_256×256]
│   ├── 000002.png
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each PNG is a side-by-side composite: **left half = SAR, right half = EO**, both normalized to uint8. The model's dataset loader splits them back at load time.

---

### SAR preprocessing

**File:** `src/gan_pipeline/data/sentinel_utils.py` — `normalize_sar`, `make_sar_image`

Sentinel-1 GRD backscatter follows this normalization pipeline:

```
raw (linear power, float32)
  │
  ├── [if not already_db]  10·log₁₀(max(x, 1e-10))
  │
  ▼
dB values
  │
  clip to [sar_min_db, sar_max_db]     default: [-25, 0] dB
  │
  ▼
(x − min_db) / (max_db − min_db)      → [0, 1]
  │
  × 255  →  uint8                      → [0, 255]
```

**Why [-25, 0] dB?**  
For Sentinel-1 GRD over land, backscatter typically falls in the range -20 to -5 dB for urban areas, -15 to -8 dB for vegetation, and -25 to -15 dB for water. Clipping at -25 dB captures almost all land cover while eliminating noise at very low values. Values above 0 dB (bright targets like corner reflectors) are saturated — acceptable for translation training.

**Channel options (`--sar-channels`):**

| Value | Output | Contents |
|---|---|---|
| `1` (default) | `(H, W, 1)` grayscale | VV polarization |
| `3` | `(H, W, 3)` pseudo-RGB | VV / VH / VV stacked |

The 3-channel stacking (VV, VH, VV) is a common visualization convention that makes VV visible in red and blue channels and VH in green, providing contrast between surfaces.

**`--sar-already-db`:** Skip the linear→dB conversion. Required for SEN12MS, which stores S1 data already in dB. Raw Sentinel-1 GRD products from ESA Copernicus are in linear power scale and do not need this flag.

---

### EO preprocessing

**File:** `src/gan_pipeline/data/sentinel_utils.py` — `normalize_eo`, `make_eo_image`

Sentinel-2 L2A (Bottom-of-Atmosphere) reflectance pipeline:

```
raw (uint16, reflectance × 10000)
  │
  ÷ 10000                              → physical reflectance [0, ~1]
  │
  clip to [0, refl_cap]               default: refl_cap = 0.3
  │
  ÷ refl_cap                          → [0, 1]
  │
  × 255  →  uint8                     → [0, 255]
```

**Why clip at 0.3?**  
Most vegetated and urban surfaces have reflectance below 0.3 in the visible bands. Clipping there maximizes contrast for typical land scenes. Snow, bright sand, or cloud (reflectance > 0.5) will saturate — this is acceptable since cloudy patches are filtered out by `--min-valid-fraction`.

**RGB band selection:**

Sentinel-2 has 13 spectral bands. We select three for RGB output:

| Natural colour channel | Sentinel-2 band | Wavelength |
|---|---|---|
| Red | B04 | 665 nm |
| Green | B03 | 560 nm |
| Blue | B02 | 490 nm |

Band indices depend on the dataset's band ordering:
- **SEN12MS** (default): bands are ordered B02, B03, B04, … → RGB indices `(2, 1, 0)`
- **Standard ESA**: bands ordered B01, B02, B03, B04, … → use `--s2-standard-order`, indices `(3, 2, 1)`

---

### Training data formats

After running `prepare_data.py`, the model expects one of two layouts, set in `configs/data/sar_eo.yaml`:

**Format 1: Side-by-side** (`dataset_format: side_by_side`, default)

Each file is a single image with SAR on the left and EO on the right. This is what `prepare_data.py` produces.

```
data/sar_eo/
├── train/00001.png    # 512×256: [SAR | EO]
├── val/00001.png
└── test/00001.png
```

**Format 2: Separate directories** (`dataset_format: separate_dirs`)

SAR and EO in separate folders; filenames must match.

```
data/sar_eo/
├── trainA/00001.png   # SAR
├── trainB/00001.png   # EO (same filename)
├── valA/ …  valB/ …
```

**SAR channel config** (`data.sar_channels`):

| Value | Description |
|---|---|
| `1` (default) | Grayscale SAR — single polarization |
| `3` | Multi-channel SAR — VV/VH/VV stacked to RGB |

---

### Data augmentation

Applied automatically during training, **synchronized** across SAR and EO so both halves of each pair receive identical spatial transforms:

1. Resize to `round(image_size × 1.12)` — e.g. 286×286 for 256 target
2. `RandomCrop(image_size)` — same `(i, j, h, w)` applied to both
3. `RandomHorizontalFlip(p=0.5)` — same flip decision for both
4. `Normalize(mean=0.5, std=0.5)` — maps uint8 [0, 255] → float [-1, 1]

Augmentation is disabled for validation and inference (`augment=false`).

---

### Sentinel-1/2 paired data with land-cover classes

SEN12MS ships four co-registered data streams per tile:

```
SEN12MS/
├── s1/   ROIs<roi>_<season>_s1_<scene>/   s1_<patch>.tif   # (2, 256, 256) float32 — VV, VH already in dB
├── s2/   ROIs<roi>_<season>_s2_<scene>/   s2_<patch>.tif   # (13, 256, 256) uint16 — all 13 S2 bands ×10000
├── lc/   ROIs<roi>_<season>_lc_<scene>/   lc_<patch>.tif   # (1, 256, 256) uint8  — MODIS LC class per pixel
└── dem/  ROIs<roi>_<season>_dem_<scene>/  dem_<patch>.tif  # (1, 256, 256) float32 — elevation in metres
```

Pairs are identified by matching `(roi_id, season, scene_id, patch_id)` across directories. The `lc/` band is the key resource for class-balanced training — it is not consumed by the model but used during preprocessing to label each tile with its dominant land-cover class.

#### MODIS IGBP land-cover classes

SEN12MS uses the MODIS MCD12Q1 product (IGBP classification scheme):

| Class ID | Label | Typical SAR signature |
|---|---|---|
| 1–5 | Forests (needleleaf / broadleaf / mixed) | Medium–high backscatter, volume scattering |
| 6–7 | Shrublands | Low–medium backscatter |
| 8–9 | Savannas / woody savannas | Seasonal variation in backscatter |
| 10 | Grasslands | Low, smooth backscatter |
| 11 | Permanent wetlands | Double-bounce + specular; temporally variable |
| 12 | Croplands | Strong seasonal signal — phenology visible in S1 |
| 13 | Urban | Strong double-bounce; very high backscatter |
| 14 | Cropland / natural mosaic | Mixed |
| 15 | Snow and ice | Very low backscatter (specular) |
| 16 | Barren | Low, rough-surface backscatter |
| 17 | Water bodies | Near-zero backscatter (specular) |

Classes 15 (snow), 16 (barren), and 17 (water) are often underrepresented in global datasets. The SAR→EO mapping is hardest to learn for these classes because their optical appearances differ strongly from the dominant forest/cropland majority — which is exactly when the model is most likely to hallucinate.

#### Step 1 — extract LC labels during preprocessing

Pass `--lc-dir` to `prepare_data.py` in `sen12ms` mode. The script reads the corresponding `lc_<patch>.tif`, computes the dominant class (mode over the 256×256 tile), and appends it to a `manifest.csv` in the output directory:

```bash
python scripts/prepare_data.py \
  --mode sen12ms \
  --s1-dir /data/SEN12MS/s1 \
  --s2-dir /data/SEN12MS/s2 \
  --lc-dir /data/SEN12MS/lc \
  --output-dir data/sar_eo \
  --sar-already-db \
  --sar-channels 1 \
  --val-split 0.1 \
  --test-split 0.1
```

The manifest records one row per tile:

```
data/sar_eo/
├── train/
│   ├── 000001.png
│   └── ...
├── val/
├── test/
└── manifest.csv          ← one row per tile
```

```
filename,split,lc_class,lc_name
train/000001.png,train,12,Croplands
train/000002.png,train,1,Evergreen Needleleaf Forests
train/000003.png,train,17,Water Bodies
...
```

#### Step 2 — class-balanced sampling with WeightedRandomSampler

With the manifest in hand, replace the default `shuffle=True` DataLoader with a `WeightedRandomSampler` that upsamples rare classes:

```python
import csv
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from gan_pipeline.data.paired_dataset import SideBySidePairedDataset

# 1. Load the manifest for the training split
manifest_path = "data/sar_eo/manifest.csv"
train_classes: list[int] = []
with open(manifest_path) as f:
    for row in csv.DictReader(f):
        if row["split"] == "train":
            train_classes.append(int(row["lc_class"]))

# 2. Compute per-class weights (inverse frequency)
class_counts = Counter(train_classes)
total = sum(class_counts.values())
class_weight = {cls: total / count for cls, count in class_counts.items()}

# 3. Assign a weight to every sample
sample_weights = [class_weight[cls] for cls in train_classes]

# 4. Build the sampler — replacement=True so rare classes are oversampled
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# 5. Build the dataset and DataLoader (shuffle must be False when using a sampler)
dataset = SideBySidePairedDataset(
    root="data/sar_eo",
    split="train",
    image_size=256,
    sar_channels=1,
    eo_channels=3,
    augment=True,
)

from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=1,
    sampler=sampler,       # replaces shuffle=True
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)
```

The sampler ensures that every class is seen at approximately equal frequency per epoch, preventing the model from mode-collapsing to the majority class (croplands / forests) and neglecting water, snow, and urban patches where hallucination risk is highest.

#### Why this matters for SAR→EO translation

The SAR→EO mapping is class-dependent:

- **Forest / cropland** (≈60–70% of SEN12MS): abundant training signal; model learns reliably.
- **Urban** (~3%): strong double-bounce SAR signature maps to a very distinct optical appearance; undersampled but learnable with balanced sampling.
- **Water / snow** (<2%): near-specular SAR (near-zero backscatter) is consistent across many different optical scenes — the one-to-many ambiguity is worst here. Without balancing, the model rarely trains on these classes and produces blurry or wrong-colour outputs for water bodies and snow-covered terrain.

Without class-balanced sampling, training metrics (FID / LPIPS) will look healthy because they are dominated by the majority classes. The failure on water and snow only becomes visible in stratified per-class evaluation — which is why the hallucination audit (Section 8.4 of the production guide) reports metrics stratified by land-cover class, not as a single aggregate.

#### Verifying class balance

After preparing the dataset, inspect the class distribution before training:

```python
import csv
from collections import Counter

with open("data/sar_eo/manifest.csv") as f:
    rows = list(csv.DictReader(f))

train_rows = [r for r in rows if r["split"] == "train"]
counts = Counter(r["lc_name"] for r in train_rows)
for name, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {n:6d}  {n/len(train_rows)*100:5.1f}%  {name}")
```

A heavily skewed distribution (e.g. Croplands at 35%, Water at 0.8%) confirms that class-balanced sampling is needed.

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
| `model.discriminator.spectral_norm` | `true` | Apply spectral norm to all D conv layers |

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
| `training.lambda_vgg` | `10.0` | Weight of VGG perceptual loss; set to `0.0` to disable |
| `training.vgg_weights_path` | `null` | Local path to `vgg16-*.pth` for offline use; `null` = download on first run |
| `training.lambda_fm` | `10.0` | Weight of feature matching loss; set to `0.0` to disable |
| `training.save_every` | `10` | Save checkpoint every N epochs |
| `training.sample_every` | `5` | Save sample grid every N epochs |
| `training.log_every` | `100` | Log to console every N batches |
| `training.num_workers` | `4` | DataLoader worker processes |

#### `configs/data/sar_eo.yaml`

| Key | Default | Description |
|---|---|---|
| `data.root` | `data/sar_eo` | Path to dataset root (output of `prepare_data.py`) |
| `data.image_size` | `256` | Spatial resolution (must be power of 2, ≥ 32) |
| `data.sar_channels` | `1` | SAR input channels (1 or 3) |
| `data.eo_channels` | `3` | EO output channels (3 for RGB) |
| `data.dataset_format` | `side_by_side` | `side_by_side` or `separate_dirs` |
| `data.augment_train` | `true` | Enable crop + flip augmentation for training |

### `prepare_data.py` CLI reference

`prepare_data.py` uses argparse (not Hydra) since it is a one-off preprocessing step, not a training loop.

```
--mode              sen12ms | scenes                   (required)
--s1-dir            path to Sentinel-1 root            (required)
--s2-dir            path to Sentinel-2 root            (required)
--output-dir        destination for PNGs               (default: data/sar_eo)

SAR options:
  --sar-channels    1 or 3                             (default: 1)
  --sar-min-db      lower dB clip                      (default: -25)
  --sar-max-db      upper dB clip                      (default: 0)
  --sar-already-db  skip linear→dB (use for SEN12MS)

EO options:
  --s2-scale        raw→reflectance divisor            (default: 10000)
  --refl-cap        reflectance saturation point       (default: 0.3)
  --s2-standard-order  use ESA band order (B01 first)

Patch options (scenes mode only):
  --image-size      patch size in pixels               (default: 256)
  --stride          sliding window stride              (default: image-size)
  --min-valid-fraction  NaN/zero rejection threshold   (default: 0.8)

Split:
  --val-split       fraction for validation            (default: 0.1)
  --test-split      fraction for test                  (default: 0.1)
  --seed            random seed                        (default: 42)
```

---

## Training

### Step 0: prepare data

```bash
# SEN12MS (recommended for experiments)
python scripts/prepare_data.py \
  --mode sen12ms \
  --s1-dir /data/SEN12MS/s1 \
  --s2-dir /data/SEN12MS/s2 \
  --output-dir data/sar_eo \
  --sar-already-db

# Raw co-registered scenes
python scripts/prepare_data.py \
  --mode scenes \
  --s1-dir /data/raw/s1 \
  --s2-dir /data/raw/s2 \
  --output-dir data/sar_eo \
  --stride 128
```

Or use the Makefile shortcut (edit paths in `Makefile` first):

```bash
make prepare-data
```

### Step 1: train

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
python scripts/train.py model=dcgan training=default data=celeba
# or:
make train-dcgan
```

---

## Evaluation

Compute **FID** (Fréchet Inception Distance) and **IS** (Inception Score) against a held-out set of real EO images.

```bash
pip install -e ".[eval]"   # installs torch-fidelity

python scripts/evaluate.py \
  checkpoint=outputs/sar_eo_pix2pix/checkpoints/epoch_0199.pt \
  real_dir=data/sar_eo/test \
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

**Logged parameters:** model name, loss type, λ_L1, λ_VGG, λ_FM, number of discriminator scales, learning rates, batch size  
**Logged metrics per epoch:** `d_loss`, `g_adv`, `g_l1`, `g_vgg`, `g_fm`

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
# All tests (68 total)
pytest

# Verbose output
pytest -v

# Single test file
pytest tests/test_pix2pix.py -v
pytest tests/test_sentinel_utils.py -v

# Run tests matching a pattern
pytest -k "multiscale or sentinel" -v

# With coverage report
pytest --cov=gan_pipeline --cov-report=term-missing
```

### Test coverage

| File | Tests | What's covered |
|---|---|---|
| `test_data.py` | 4 | Transform output shape (32/64/128), pixel range [-1, 1] |
| `test_models.py` | 8 | DCGAN generator/discriminator shapes; BCE/Wasserstein/Hinge losses; gradient penalty; `sample()` |
| `test_pix2pix.py` | 29 | U-Net output shape (1→3, 3→3, 1→1 ch); skip-connection gradients; PatchGAN patch map shape (~30×30), spectral norm on/off (weight_orig presence), and `forward_with_features` structure; multi-scale output lengths, decreasing spatial sizes, SN threading, and `forward_with_features` per scale; all loss types on multi-scale maps; VGG perceptual loss (1/3/4-channel inputs, zero on identical, offline weights_path); feature matching loss (scalar, finite, zero on identical); train step (hinge×3scale, bce×1scale, hinge×2scale); side-by-side dataset load and augmentation |
| `test_sentinel_utils.py` | 19 | `linear_to_db` correctness and zero-safety; SAR/EO normalization ranges and clipping; `make_sar_image` channel configs (1/3ch) and input layouts (CHW/HWC); `make_eo_image` shape; `is_valid_patch` NaN/zero rejection; `make_side_by_side` shape, broadcast, spatial mismatch error, SAR-on-left |
| `test_training.py` | 2 | Checkpoint save/load round-trip; DCGAN trainer step (finite float losses) |

### Makefile shortcuts

```bash
make install        # pip install -e ".[dev]"
make test           # pytest
make lint           # ruff + black --check + isort --check
make format         # black + isort + ruff --fix (in-place)
make typecheck      # mypy src/
make prepare-data   # python scripts/prepare_data.py (sen12ms mode, edit paths first)
make train          # python scripts/train_pix2pix.py
make train-dcgan    # python scripts/train.py model=dcgan ...
make clean          # remove __pycache__, .pytest_cache, .mypy_cache, egg-info
```

**Pre-commit:**

```bash
pre-commit install          # activate hooks (run once after cloning)
pre-commit run --all-files  # run all hooks manually
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

### Adding a new SAR/EO normalization scheme

All preprocessing math is isolated in `sentinel_utils.py`. Add a new function there and call it from `prepare_data.py`. The training pipeline is unaffected since it only sees uint8 PNGs.

### Checkpoint format

Checkpoints are plain PyTorch `.pt` files:

```python
{
    "epoch": int,
    "generator": OrderedDict,       # generator.state_dict()
    "discriminator": OrderedDict,   # discriminator.state_dict()
    "opt_g": dict,                  # optimizer state
    "opt_d": dict,
    "metrics": {"d_loss": float, "g_adv": float, "g_l1": float, "g_vgg": float, "g_fm": float},
}
```

Load with:

```python
from gan_pipeline.utils.checkpointing import load_checkpoint
state = load_checkpoint(Path("epoch_0199.pt"), device)
generator.load_state_dict(state["generator"])
```
