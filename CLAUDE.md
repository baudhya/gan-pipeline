# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (core + dev tools)
pip install -e ".[dev]"

# Add geospatial deps (required for scripts/prepare_data.py)
pip install -e ".[geo]"

# Add FID/IS evaluation
pip install -e ".[eval]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_pix2pix.py -v

# Run tests matching a pattern
pytest -k "multiscale or sentinel" -v

# Lint / format / typecheck
make lint        # ruff + black --check + isort --check
make format      # black + isort + ruff --fix (in-place)
make typecheck   # mypy src/

# Train pix2pix (SAR→EO)
python scripts/train_pix2pix.py

# Unconditional DCGAN
python scripts/train.py model=dcgan training=default data=celeba

# Prepare Sentinel data
make prepare-data   # edit paths in Makefile first
```

## Architecture

This repo contains two independent training pipelines sharing models and utilities.

### Conditional pipeline (pix2pix) — main use case

**Entry point:** `scripts/train_pix2pix.py`  
**Trainer:** `src/gan_pipeline/training/pix2pix_trainer.py` — `Pix2PixTrainer`

Data flow:
1. `get_paired_dataloader` → yields `{"sar": Tensor, "eo": Tensor}` dicts
2. `UNetGenerator(sar) → fake_eo`
3. `MultiScaleDiscriminator(cat([sar, eo]))` → list of N patch maps (finest→coarsest)
4. Losses averaged across scales via `multiscale_discriminator_loss` / `multiscale_generator_loss`
5. Generator total loss: `g_adv + lambda_l1 * F.l1_loss(fake_eo, real_eo)`

### Unconditional pipeline (DCGAN)

**Entry point:** `scripts/train.py`  
**Trainer:** `src/gan_pipeline/training/trainer.py` — `GANTrainer`  
Uses `DCGANGenerator` + `DCGANDiscriminator` from `models/dcgan.py`. Latent vector as input, no conditioning.

### Model hierarchy

```
BaseGenerator / BaseDiscriminator   (models/base.py — ABCs)
  ├── UNetGenerator                 (models/unet.py — 8-level encoder/decoder, skip connections)
  ├── DCGANGenerator                (models/dcgan.py)
  ├── PatchGANDiscriminator         (models/patchgan.py — 70×70 receptive field)
  │     └── wrapped by MultiScaleDiscriminator  (models/multiscale_disc.py)
  └── DCGANDiscriminator            (models/dcgan.py)
```

`MultiScaleDiscriminator.forward()` returns `list[Tensor]` (one patch map per scale), not a single tensor — this is why there are separate `multiscale_*_loss` functions in `models/losses.py` distinct from the single-scale `generator_loss` / `discriminator_loss`.

### Configuration

All hyperparameters are managed by **Hydra**. The config tree under `configs/` is composed at runtime:

```
configs/config.yaml         ← root; selects defaults
configs/model/pix2pix.yaml  ← generator/discriminator features, n_scales
configs/training/pix2pix.yaml ← lr, loss_type, lambda_l1, epochs, etc.
configs/data/sar_eo.yaml    ← root path, image_size, sar_channels, dataset_format
```

CLI overrides: `python scripts/train_pix2pix.py training.loss_type=bce model.discriminator.n_scales=2`

### Data pipeline

**Training data format:** side-by-side PNGs — SAR on the left half, EO on the right half (512×256 for 256×256 target). Produced by `scripts/prepare_data.py`.

`SideBySidePairedDataset` (`data/paired_dataset.py`) crops the image at `w//2` to recover each half. Augmentation (resize→crop→hflip) is synchronized across both halves using `torchvision.transforms.functional` with shared `(i, j, h, w)` parameters. All tensors are normalized to `[-1, 1]`.

`sentinel_utils.py` contains pure-numpy preprocessing (no rasterio dependency): SAR linear→dB→clip→uint8, EO reflectance→clip→uint8, side-by-side assembly. These are called by `prepare_data.py` but are importable standalone.

### Key non-obvious constraints

- **`ReLU(inplace=False)` in `_dec_block` (unet.py):** inplace ReLU in the decoder corrupts encoder skip tensors that LeakyReLU backward needs — causes `RuntimeError: ... is at version 2; expected version 1`. Never change these to inplace.
- **`/data/` in `.gitignore`** is root-anchored intentionally — `data/` would also exclude `src/gan_pipeline/data/`.
- **Checkpoint format:** `{"epoch", "generator", "discriminator", "opt_g", "opt_d", "metrics"}` — see `utils/checkpointing.py`.
- **MLflow** is logged automatically during every training run; no separate setup needed for local tracking (`mlruns/` directory).
- **Do not add `Co-Authored-By:` lines** to git commits — only the repo owner is the commit author.
