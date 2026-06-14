# Skill: smoke-test

Run a fast end-to-end training smoke test to verify the full pipeline works.

## Steps

1. Generate dummy SAR/EO data:
   ```bash
   python scripts/make_dummy_data.py --train 50 --val 10 --image-size 256
   ```

2. Run 3 training epochs with all losses enabled:
   ```bash
   python scripts/train_pix2pix.py \
     training.epochs=3 \
     training.save_every=1 \
     training.sample_every=1 \
     training.log_every=10 \
     training.num_workers=0
   ```

3. Confirm all of the following in the output:
   - `Train set: 50 pairs` — dataset loaded correctly
   - `G_VGG` values present and finite — VGG loss active
   - `G_FM` values present and finite — feature matching active
   - No exceptions or NaN losses
   - Checkpoint written to `outputs/sar_eo_pix2pix/checkpoints/`
   - Sample grid written to `outputs/sar_eo_pix2pix/samples/`

## Notes

- Uses `training.num_workers=0` to avoid multiprocessing issues on CPU
- VGG loss requires `weights/vgg16-397923af.pth` — run `make download-weights` first if missing
- Dummy data is noise; losses won't converge meaningfully — that's expected
- D_loss collapsing to near zero on 50 noise images is normal; it won't happen on real data
