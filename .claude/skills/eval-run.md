# Skill: eval-run

Evaluate a trained checkpoint with FID and Inception Score.

## Steps

1. Check the eval extra is installed:
   ```bash
   pip show torch-fidelity 2>/dev/null || pip install -e ".[eval]"
   ```

2. Ask the user for:
   - **checkpoint** — path to `.pt` file, e.g. `outputs/sar_eo_pix2pix/checkpoints/epoch_0199.pt`
   - **real_dir** — path to real EO images, e.g. `data/sar_eo/test`
   - **eval_samples** — `5000` (default)

3. Run evaluation:
   ```bash
   python scripts/evaluate.py \
     checkpoint=<checkpoint> \
     real_dir=<real_dir> \
     eval_samples=<eval_samples>
   ```

4. Report results:
   - **FID** — lower is better; < 50 is reasonable for SAR→EO, < 20 is strong
   - **IS mean / std** — higher is better; measures sharpness and diversity

5. If comparing multiple checkpoints, suggest running MLflow UI to compare runs:
   ```bash
   mlflow ui --port 5000
   ```

## Notes

- Requires `[eval]` extra: `pip install -e ".[eval]"` or `INSTALL_EVAL=1 bash init.sh`
- `eval_samples` should not exceed the number of images in `real_dir`
- FID on dummy/noise data is meaningless — only run on real Sentinel imagery
- For per-class evaluation (stratified by land-cover), use `manifest.csv` from `prepare_data.py`
