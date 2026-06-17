# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project overview

Production-grade **SAR→EO image translation** pipeline built on pix2pix. The main model translates Sentinel-1 SAR images into Sentinel-2 optical images using a U-Net generator and a multi-scale PatchGAN discriminator. A secondary unconditional DCGAN pipeline shares models and utilities.

**Owner:** Siddharth Baudh  
**Primary language:** Python 3.10+  
**Package:** `gan_pipeline` (installed as `pip install -e ".[dev]"`)

---

## Environment setup

```bash
# One-command setup (recommended)
bash init.sh                        # checks Python ≥3.10, creates .venv, installs deps, activates pre-commit

# Activate virtual environment
source .venv/bin/activate

# Manual setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Optional extras
pip install -e ".[geo]"             # rasterio + h5py — required for scripts/prepare_data.py
pip install -e ".[eval]"            # torch-fidelity — required for scripts/evaluate.py

# Makefile shortcuts
make venv                           # create .venv (PYTHON= and VENV_DIR= overridable)
make install                        # pip install -e ".[dev]"
make clean                          # remove __pycache__, .pytest_cache, .mypy_cache, egg-info
make clean-venv                     # remove .venv
make clean-all                      # clean + clean-venv
```

---

## Common commands

```bash
# Tests
pytest                              # run all 68 tests
pytest tests/test_pix2pix.py -v     # single file
pytest -k "multiscale or sentinel"  # pattern match

# Lint / format / typecheck
make lint                           # ruff + black --check + isort --check
make format                         # black + isort + ruff --fix (in-place)
make typecheck                      # mypy src/ (strict mode)

# Data
make download-weights               # download VGG16 weights into weights/ (~528 MB, once)
make prepare-data                   # prepare Sentinel data (edit paths in Makefile first)
python scripts/make_dummy_data.py   # generate 50 dummy SAR/EO pairs for smoke testing

# Training
python scripts/train_pix2pix.py                                              # original pix2pix (single PatchGAN, L1 only, BCE)
python scripts/train_pix2pix.py experiment_name=run1 training.epochs=50     # named experiment
python scripts/train_pix2pixhd.py                                            # pix2pixHD (multi-scale, VGG + FM losses, hinge)
python scripts/train_pix2pixhd.py experiment_name=hd1 training.epochs=200   # named pix2pixHD run
python scripts/train.py model=dcgan training=default data=celeba             # unconditional DCGAN

# Evaluation
python scripts/evaluate.py checkpoint=outputs/.../epoch_0199.pt real_dir=data/sar_eo/test

# MLflow UI (MLflow 3.x stores in sqlite:///mlflow.db)
make mlflow                         # opens UI at http://localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000  # equivalent manual command
```

---

## Pre-commit hooks

Five hooks run automatically on every `git commit` (same gates as CI):

| Hook | Command | Blocks if… |
|---|---|---|
| `ruff lint` | `ruff check src tests scripts` | Any lint error |
| `black format check` | `black --check src tests scripts` | Any file would be reformatted |
| `isort import check` | `isort --check-only src tests scripts` | Imports not sorted |
| `mypy type check` | `mypy src` | Any type error in `src/` |
| `pytest` | `pytest --no-header -q` | Any test fails |

Run manually: `pre-commit run --all-files`  
Skip in emergency: `git commit --no-verify -m "wip: ..."`  
Fix formatting: `make format` then re-stage.

---

## CI workflows

Two GitHub Actions workflows (`.github/workflows/`):

| File | Triggers | What it runs |
|---|---|---|
| `pre-commit.yml` | push / PR to master | `pre-commit run --all-files` (lint + typecheck + tests) |
| `pytest.yml` | push / PR to master | `pytest` (full test suite) |

---

## Architecture

### Conditional pipeline (pix2pix) — main use case

**Entry point:** `scripts/train_pix2pix.py`  
**Trainer:** `src/gan_pipeline/training/pix2pix_trainer.py` — `Pix2PixTrainer`

Data flow:
1. `get_paired_dataloader` → yields `{"sar": Tensor, "eo": Tensor}` dicts
2. `UNetGenerator(sar) → fake_eo`
3. `MultiScaleDiscriminator(cat([sar, eo]))` → list of N patch maps (finest→coarsest)
4. Losses averaged across scales via `multiscale_discriminator_loss` / `multiscale_generator_loss`
5. Generator total loss: `g_adv + λ_L1·L1(fake_eo, real_eo) + λ_VGG·L_VGG + λ_FM·L_FM`

### Unconditional pipeline (DCGAN)

**Entry point:** `scripts/train.py`  
**Trainer:** `src/gan_pipeline/training/trainer.py` — `GANTrainer`  
Uses `DCGANGenerator` + `DCGANDiscriminator`. Latent vector as input, no conditioning.

### Model hierarchy

```
BaseGenerator / BaseDiscriminator   (models/base.py — ABCs)
  ├── UNetGenerator                 (models/unet.py — 8-level encoder/decoder, skip connections)
  ├── DCGANGenerator                (models/dcgan.py)
  ├── PatchGANDiscriminator         (models/patchgan.py — 70×70 receptive field)
  │     └── wrapped by MultiScaleDiscriminator  (models/multiscale_disc.py)
  └── DCGANDiscriminator            (models/dcgan.py)
```

`MultiScaleDiscriminator.forward()` returns `list[Tensor]` (one patch map per scale), not a single tensor — this is why `multiscale_*_loss` functions in `models/losses.py` exist separately from the single-scale variants.

### Loss functions (`models/losses.py`)

| Loss | Default weight | Controlled by |
|---|---|---|
| Adversarial (hinge/bce/wasserstein) | 1.0 | `training.loss_type` |
| L1 pixel | 100.0 | `training.lambda_l1` |
| VGG perceptual | 10.0 | `training.lambda_vgg` |
| Feature matching | 10.0 | `training.lambda_fm` |

Set any lambda to `0.0` to disable that loss term entirely.

### Configuration (Hydra)

```
configs/config.yaml           ← root; selects defaults
configs/model/pix2pix.yaml    ← generator/discriminator features, n_scales
configs/training/pix2pix.yaml ← lr, loss_type, lambdas, epochs, vgg_weights_path
configs/data/sar_eo.yaml      ← root path, image_size, sar_channels, dataset_format
```

CLI overrides: `python scripts/train_pix2pix.py training.loss_type=bce model.discriminator.n_scales=2`

### Data pipeline

**Training format:** side-by-side PNGs — SAR left half, EO right half (512×256 for 256×256 target). Produced by `scripts/prepare_data.py`.

`SideBySidePairedDataset` (`data/paired_dataset.py`) crops at `w//2`. Augmentation (resize→crop→hflip) is synchronized across both halves via shared `(i, j, h, w)` parameters. All tensors normalized to `[-1, 1]`.

For smoke testing without real data: `python scripts/make_dummy_data.py` generates 50 noise pairs in `data/sar_eo/`.

### VGG weights

`configs/training/pix2pix.yaml` defaults to `vgg_weights_path: weights/vgg16-397923af.pth`.  
Run `make download-weights` once after cloning to populate it (~528 MB).  
If the file is missing, `VGGPerceptualLoss` raises a clear `FileNotFoundError` with instructions.  
To fall back to auto-download: `training.vgg_weights_path=null`.

### Checkpoint format

```python
{"epoch": int, "generator": OrderedDict, "discriminator": OrderedDict,
 "opt_g": dict, "opt_d": dict,
 "metrics": {"d_loss", "g_adv", "g_l1", "g_vgg", "g_fm"}}
```

See `utils/checkpointing.py`. Outputs go to `outputs/<experiment_name>/checkpoints/`.

---

## Skills

Invoke with `/skill-name` in Claude Code. Files live in `.claude/skills/`.

| Skill | When to use |
|---|---|
| `/smoke-test` | After any model/data change — dummy data + 3-epoch full-loss training verification |
| `/train-run` | Starting a training run — prompts for all config options, builds and runs the command |
| `/eval-run` | Evaluating a checkpoint — prompts for paths, runs FID + IS |
| `/new-model` | Adding a new architecture — checklist from file creation through tests |
| `/prep-data` | Preparing Sentinel data — guides through `prepare_data.py` args |
| `/commit` | Stage and commit changes — enforces conventions, runs hooks, never adds Co-Authored-By |

---

## Key non-obvious constraints

- **`ReLU(inplace=False)` in `_dec_block` (unet.py):** inplace ReLU in the decoder corrupts encoder skip tensors that LeakyReLU backward needs — causes `RuntimeError: ... is at version 2; expected version 1`. Never change to inplace.
- **`/data/` in `.gitignore`** is root-anchored intentionally — `data/` would also exclude `src/gan_pipeline/data/`.
- **`*.pth` and `*.pt` are gitignored** — VGG weights and checkpoints are never committed. `weights/.gitkeep` tracks the directory only.
- **`MultiScaleDiscriminator` iteration:** `nn.ModuleList` elements type as `nn.Module`; use `cast(PatchGANDiscriminator, disc)` when calling `forward_with_features` — already done in `multiscale_disc.py`.
- **mypy strict mode** is enforced — all new code in `src/` must pass `mypy src/` with no errors. Use `# type: ignore[no-any-return]` for torch return types where needed.
- **MLflow** logs automatically on every training run to `mlruns/` — no setup needed.
- **Do not add `Co-Authored-By:` lines** to git commits — only the repo owner is the commit author.
