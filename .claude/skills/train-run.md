# Skill: train-run

Configure and launch a named pix2pix training run.

## Steps

1. Ask the user for (or use defaults where shown):
   - **experiment_name** — e.g. `hinge_3scale_vgg` (required, no default)
   - **loss_type** — `hinge` (default) | `bce` | `wasserstein`
   - **n_scales** — `3` (default) | `2` | `1`
   - **lambda_l1** — `100.0` (default)
   - **lambda_vgg** — `10.0` (default) | `0.0` to disable
   - **lambda_fm** — `10.0` (default) | `0.0` to disable
   - **epochs** — `200` (default)
   - **resume** — path to checkpoint, or omit

2. Build the training command:
   ```bash
   python scripts/train_pix2pix.py \
     experiment_name=<name> \
     training.loss_type=<loss_type> \
     model.discriminator.n_scales=<n_scales> \
     training.lambda_l1=<lambda_l1> \
     training.lambda_vgg=<lambda_vgg> \
     training.lambda_fm=<lambda_fm> \
     training.epochs=<epochs> \
     [resume=<checkpoint_path>]
   ```

3. Show the user the full command and confirm before running.

4. Run and monitor output. Flag if:
   - D_loss collapses to ~0 and stays there past epoch 10 (mode collapse risk)
   - Any loss becomes NaN
   - VGG loss is 0.0 when lambda_vgg > 0 (weights not loaded)

5. On completion, report final epoch losses and checkpoint location.

## Notes

- Outputs go to `outputs/<experiment_name>/`
- MLflow logs automatically to `mlruns/` — view with `mlflow ui --port 5000`
- VGG loss requires `weights/vgg16-397923af.pth` — run `make download-weights` if missing
- For GPU training, no extra flags needed; device is auto-detected
