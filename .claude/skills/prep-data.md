# Skill: prep-data

Guide the user through preparing Sentinel-1/2 data for training.

## Steps

1. Check the geo extra is installed:
   ```bash
   pip show rasterio 2>/dev/null || pip install -e ".[geo]"
   ```

2. Ask the user:
   - **mode** — `sen12ms` (pre-cropped 256×256 patches) | `scenes` (full scenes, sliding window)
   - **s1_dir** — path to Sentinel-1 data root
   - **s2_dir** — path to Sentinel-2 data root
   - **output_dir** — destination for PNGs (default: `data/sar_eo`)
   - **sar_channels** — `1` (VV only, default) | `3` (VV/VH/VV pseudo-RGB)
   - **val_split** — `0.1` (default)
   - **test_split** — `0.1` (default)
   - **sar_already_db** — `true` for SEN12MS (already in dB), `false` for raw ESA GRD products

   If mode is `scenes`, also ask:
   - **stride** — `128` (50% overlap, default) | `256` (no overlap)
   - **min_valid_fraction** — `0.9` (default)

3. Build and run the command:

   **sen12ms mode:**
   ```bash
   python scripts/prepare_data.py \
     --mode sen12ms \
     --s1-dir <s1_dir> \
     --s2-dir <s2_dir> \
     --output-dir <output_dir> \
     --sar-channels <sar_channels> \
     --val-split <val_split> \
     --test-split <test_split> \
     [--sar-already-db]
   ```

   **scenes mode:**
   ```bash
   python scripts/prepare_data.py \
     --mode scenes \
     --s1-dir <s1_dir> \
     --s2-dir <s2_dir> \
     --output-dir <output_dir> \
     --image-size 256 \
     --stride <stride> \
     --min-valid-fraction <min_valid_fraction> \
     --sar-channels <sar_channels> \
     --val-split <val_split> \
     --test-split <test_split>
   ```

4. After completion, report:
   - Number of train/val/test pairs produced
   - Output directory structure
   - Remind user to update `configs/data/sar_eo.yaml` if `sar_channels` changed

5. Suggest running `/smoke-test` to confirm the data loads correctly before a full training run.

## Notes

- Requires `[geo]` extra: `pip install -e ".[geo]"` or `INSTALL_GEO=1 bash init.sh`
- SEN12MS S1 data is already in dB — always pass `--sar-already-db` for that dataset
- Raw ESA Copernicus GRD products are in linear power scale — do not pass `--sar-already-db`
- Side-by-side PNG format: SAR left half, EO right half, each 256×256 → total 512×256
- `configs/data/sar_eo.yaml` `sar_channels` must match `--sar-channels` used here
