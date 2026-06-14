# SAR→EO Image Translation: A Production Pipeline Guide

**Scope:** End-to-end plan for building a production-grade SAR-to-optical (EO) image translation system with generative AI — datasets, preprocessing, every relevant model family with trade-offs, losses, training, evaluation, productionization, compute budgeting, risks, and a curated reading/watching list.
**Timeline assumed:** 3 months pipeline setup + 3 months training.

---

## Table of Contents

1. Orientation & strategy
2. The physics: why this is hard, why it works at all, and the hard limits
3. The six-month roadmap (your 3 + 3)
4. Data: datasets, custom data, preprocessing, pairing, splits, QC
5. The model landscape: every family, trade-offs, decision matrix
6. Loss function catalogue
7. Training playbook
8. Evaluation protocol (where most projects fail)
9. Production engineering
10. Compute budgeting
11. Risk register: failure modes and fixes
12. Reading list & what to watch
13. Appendix: the default recipe (start here if overwhelmed)

---

## 1. Orientation & strategy

The one-paragraph strategy: **build the data pipeline and evaluation harness before any fancy model; train a deterministic U-Net baseline to validate plumbing; then run two competing tracks — a Pix2PixHD-class cGAN (fast, deterministic-ish, deployable) and a conditional diffusion / foundation-model track (higher ceiling, slower inference) — and let task-based evaluation, not FID alone, pick the winner.** In 2026 the honest summary of the field is: cGANs remain the production workhorse for speed and stability of deployment; diffusion models hold the quality crown on benchmarks (FID/LPIPS) but cost 10–50× more at inference unless distilled; and EO foundation models (TerraMind) give you a strong zero-shot/fine-tune baseline almost for free, which changes the economics of a 6-month project — you should stand one up in week 2 as your bar to beat.

Three principles that will save the project:

1. **The pipeline is the product.** 70% of final image quality is determined by data quality (co-registration accuracy, temporal gap between SAR/EO acquisitions, cloud filtering, normalization), not by architecture choice. Budget accordingly.
2. **Evaluation discipline beats model novelty.** A geographic train/test split, FID + LPIPS + a downstream-task metric, and a hallucination audit are non-negotiable. Random patch splits inflate every metric and produce models that fail on new regions.
3. **Generated EO is a visualization aid, never ground truth.** This is an ill-posed inverse problem; the model *invents* plausible optical detail. Every output must carry provenance metadata marking it AI-generated, and your eval must quantify hallucination (Section 8.4). For defence/analysis workflows this is the single most important design constraint.

---

## 2. The physics: why this is hard, why it works, and the hard limits

### 2.1 Why the mapping is ill-posed

SAR measures **complex backscatter** — a function of surface roughness relative to wavelength, dielectric constant (moisture), local incidence angle, and geometry — at C-band (Sentinel-1, ~5.6 cm), X-band (TerraSAR-X, ICEYE, ~3.1 cm), or L/S-band (RISAT/EOS-04, NISAR). Optical sensors measure **solar reflectance** in visible/NIR bands — a function of pigments, mineralogy, and illumination. These physical quantities are *correlated through land cover* but not deterministically linked:

- A smooth asphalt road and calm water both appear dark in SAR; one is grey and one is blue/green in EO.
- A field's SAR signature barely changes when crops flower; its optical color changes dramatically.
- Two different roof materials with the same geometry can have identical backscatter and wildly different colors.

So SAR→EO is a **one-to-many mapping**: one SAR patch is consistent with many optical scenes. Any model must *choose* a mode. Regression losses (L1/L2) average over modes → blur. Adversarial/diffusion objectives pick a sharp plausible mode → realism, but with **hallucination risk**: the chosen mode may differ from reality in color, texture, even small object presence. This is fundamental, not an engineering bug. It defines what the system can be used for (visualization, analyst orientation, cloud-gap filling for *context*, pre-training/augmentation) and what it must never be used for (target identification, counting objects, any claim where the optical detail itself is the evidence).

### 2.2 Why it works anyway

Land cover is a strong shared latent variable. Backscatter + texture + context (a network's receptive field sees fields, road networks, urban blocks) constrain the optical answer enough that for **structure and land-cover-level appearance** the mapping is learnable: water bodies, urban fabric, forest/cropland boundaries, road networks, terrain shading translate reliably. What is *not* recoverable: exact colors of individual small objects, sub-resolution detail, anything dominated by spectral chemistry rather than structure. Multi-temporal SAR input (a short stack of 3–5 acquisitions) further constrains the mapping — temporal backscatter signatures separate crops/water/urban far better than a single date — and is one of the highest-leverage upgrades available (Section 5.7).

### 2.3 SAR-specific statistics the model must respect

- **Speckle is multiplicative** (fully developed speckle: intensity ~ exponential/Gamma distributed). Standard vision models assume additive, roughly Gaussian perturbations. **Convert to dB (log domain)** before the network: log turns multiplicative noise additive, compresses the ~50 dB dynamic range, and makes per-channel normalization meaningful. This is the homomorphic trick and it is the single most important preprocessing decision. Typical clips: VV ∈ [−25, 0] dB, VH ∈ [−32, −5] dB, then scale to [−1, 1] (or compute robust 1st/99th percentiles on *your* dataset).
- **Geometric distortions** — layover, foreshortening, radar shadow — have no optical analogue. In mountainous terrain (your Himalayan AOIs) these are severe; Radiometrically Terrain Corrected (RTC) products + a DEM input channel mitigate but cannot remove them. Expect translation quality to degrade with slope; report metrics stratified by terrain.
- **Whether to despeckle before training is a trade-off**, not a default. Aggressive filtering (Refined Lee 7×7, SAR2SAR) removes information the network could use and bakes filter artifacts into the input domain. Two defensible choices: (a) feed raw dB and let the network learn speckle statistics (works with enough data); (b) use *temporal multilooking* (mean of a short stack) which suppresses speckle without spatial blurring. Avoid heavy single-image spatial filters in the production path; if you must, keep the same filter at train and inference time forever.

---

## 3. The six-month roadmap

### Months 1–3: Pipeline (your "setup" phase)

**Month 1 — Data foundation + plumbing validation**
- Week 1: Freeze scope: AOIs, input sensor(s) & polarizations, output bands (RGB vs RGB+NIR vs 13-band), target resolution & tile size, primary use case. Write a 2-page design doc; every later decision references it.
- Weeks 1–2: Download SEN1-2 and/or SEN12MS (instant, pre-paired). In parallel start the custom-AOI pipeline: Sentinel-1 RTC (via ASF HyP3 or SNAP graphs) + Sentinel-2 L2A via Google Earth Engine / Copernicus Data Space.
- Weeks 2–3: Preprocessing v1: SAR calibration→RTC→dB→clip→normalize; S2 cloud mask (SCL/s2cloudless)→band select→normalize; tiling (256² to start) with georeferencing preserved; manifest (parquet/STAC) of every pair with metadata (date gap, cloud %, terrain slope, land-cover mix).
- Week 4: **Train U-Net + L1 baseline end-to-end.** Purpose: validate the entire pipeline (data → train → checkpoint → tiled inference → metrics), not quality. Also: pull TerraMind from Hugging Face and run its S1→S2 generation zero-shot on your val tiles — this is your free external baseline.

**Month 2 — Scale data + first real models + eval harness**
- Data QC at scale: co-registration verification (phase-correlation shift check per tile; reject >1 px), temporal-gap filter (≤ 5–10 days for vegetated areas), cloud/shadow ≤ 1% in target, near-duplicate removal, geographic deduplication.
- Geographic train/val/test split (by region/MGRS tile, never random patches). Version the dataset with DVC.
- Implement Pix2Pix then Pix2PixHD (or adapt your existing codebase) with config-driven training (Hydra/OmegaConf), experiment tracking (MLflow or W&B offline mode for air-gapped), AMP, DDP, resumable checkpoints, EMA.
- Eval harness v1: PSNR/SSIM/LPIPS per tile + FID/KID on val; results written to a leaderboard table automatically per run.

**Month 3 — Hardening + decision gate**
- Inference service v1: sliding-window tiled inference with overlap + Hann-window blending, CRS/geotransform preserved, COG output, "AI-GENERATED" provenance tags in metadata.
- Hallucination audit tooling (Section 8.4) and task-based eval (land-cover segmentation transfer, Section 8.3).
- Hyperparameter scaffolding: small-budget sweeps (lr, λ_L1, D capacity) at 256².
- Conditional-diffusion prototype at 256² (so the training phase doesn't start cold).
- **Decision gate (end of M3):** leaderboard of {U-Net, Pix2Pix, Pix2PixHD, TerraMind zero-shot, diffusion prototype} on the geographic test split. Pick ≤ 2 tracks to scale. Write the data card + first model card.

### Months 4–6: Training (your "training" phase)

**Month 4 — Scale the cGAN track**
- Pix2PixHD-class model at 512², long runs (300k–600k iters), with ablations run as cheap 256² jobs: loss weights, VV/VH/ratio channels, speckle handling, DEM channel, season conditioning.
- Parallel: fine-tune TerraMind (TerraTorch) and/or ControlNet-on-Stable-Diffusion on your data.

**Month 5 — Diffusion track + generalization**
- Full conditional diffusion training at 256→512 (or latent diffusion at 512), v-prediction + min-SNR, EMA, classifier-free guidance; then sampler acceleration (DDIM 25–50 steps) and, if inference cost matters, consistency/LCM distillation to 2–4 steps.
- Cross-region & cross-season generalization tests; terrain-stratified metrics; multi-temporal input ablation if data supports it.

**Month 6 — Selection, robustness, packaging**
- Final model selection driven by: task-based metric + hallucination audit + analyst preference study (Section 8.5) + inference cost.
- Robustness: new-sensor sanity check (e.g., EOS-04 L-band will NOT transfer from C-band training — document it), failure-case gallery, terrain/season stress tests.
- Packaging: TorchScript/ONNX (GAN) or torch.compile/TensorRT (diffusion); batch inference CLI; model card with explicit limitations and prohibited uses; reproducibility bundle (data version + config + seed + container).

**Standing weekly cadence:** every Friday, regenerate the same fixed 32-tile "probe set" with every active model and eyeball it side-by-side. Catches regressions metrics miss.

---

## 4. Data: datasets, custom data, preprocessing, pairing, splits, QC

### 4.1 Public paired datasets

| Dataset | Size | Sensors (SAR / EO) | Resolution | Pol | Best for | Caveats |
|---|---|---|---|---|---|---|
| **SEN1-2** (Schmitt et al. 2018) | 282,384 pairs, 256² | S1 GRD / S2 RGB | 10 m | VV only | Fast start, GAN pretraining | RGB-only targets; registration is geocoding-level (~pixel); seasonal subsets |
| **SEN12MS** (Schmitt et al. 2019) | 180,662 triplets, 256² | S1 / S2 all 13 bands (+ MODIS LC) | 10 m | VV+VH | Multispectral output, dual-pol input, LC-aware sampling | Some cloudy/snowy targets — filter via provided LC + your own QC |
| **SEN12MS-CR / -CR-TS** (Ebel et al. 2021/22) | ~122k pairs / time series | S1 / S2 (cloudy + cloud-free) | 10 m | VV+VH | Cloud-removal framing; **multi-temporal** experiments | Built for cloud removal — adapt loaders |
| **QXS-SAROPT** (Huang et al. 2021) | 20,000 pairs, 256² | GaoFen-3 / Google-Earth optical | 1 m | single | High-res urban/port translation | Small; scene diversity limited (ports); optical from GE mosaics (mixed dates) |
| **SAR2Opt benchmark** (Zhao et al. 2022) | ~2,000 pairs, 600² | TerraSAR-X / Google Earth | ~1 m | single | Standard benchmark to compare against literature | Small — fine-tune/eval only, not primary training |
| **SpaceNet 6** (Shermeyer et al. 2020) | 3,401 tiles, Rotterdam | Capella aerial X-band quad-pol / Maxar WV-2 | 0.5 m | quad | Very-high-res, quad-pol experiments | Single city → use for transfer studies, not generalization claims |
| **WHU-OPT-SAR** | 100 large scenes | GF-3 / GF-1 | 5 m | single | LC-labelled SAR-opt pairs | Few scenes; China only |
| **TerraMesh** (IBM/ESA 2025) | ~9M samples, global | S1 GRD+RTC / S2 L1C+L2A (+DEM, LULC, NDVI) | 10 m | VV+VH | Foundation-scale pretraining; aligned multimodal Zarr | Huge download; designed for TerraMind-style training |

**Recommendation:** SEN12MS as the primary trainer (dual-pol + 13-band targets + LC metadata for balanced sampling), SEN1-2 as a volume booster, SAR2Opt for literature-comparable numbers, plus a **custom AOI dataset** for the regions you actually care about (below). If you target high-res X-band products later, QXS/SpaceNet6 are your transfer testbeds — but do not expect C-band-trained models to transfer to X-band (different wavelength = different scattering physics).

### 4.2 Building a custom paired dataset (the part that pays off most)

1. **AOI & period selection:** your operational regions + diverse distractors (different biomes). For each AOI, enumerate S2 L2A scenes with scene cloud < 20%, then for each find the nearest-in-time S1 acquisition.
2. **Pairing constraint:** |t_SAR − t_EO| ≤ 5 days over vegetation/water-dynamic areas; ≤ 15 days acceptable for arid/urban. Temporal gap is *label noise* — the single biggest silent quality killer. Store the gap in the manifest and ablate thresholds later.
3. **SAR processing:** prefer **analysis-ready RTC**: ASF HyP3 produces on-demand Sentinel-1 RTC (gamma0, DEM-corrected) for free, or run SNAP `gpt` graphs (Apply-Orbit → Thermal-Noise-Removal → Border-Noise → Calibration to γ⁰ → [optional multilook] → Terrain-Correction with Copernicus GLO-30 DEM). SNAP graphs parallelize embarrassingly per scene — **this is the perfect job for your CPU HPC cluster** while GPUs train.
4. **Optical processing:** S2 L2A (Sen2Cor already applied), mask clouds/shadows via SCL classes {3,8,9,10} or s2cloudless probability > 0.4 with dilation; reject tiles > 1% masked. Reflectance/10000, clip per-band robust percentiles, scale to [−1,1]. Decide output bands now: **B4/B3/B2 (RGB)** for analyst visualization; add **B8 (NIR)** if downstream vegetation tasks matter; full 13-band only if you have a concrete consumer for it (it makes the mapping harder).
5. **Co-registration check:** geocoded S1-RTC vs S2 is usually within ~1 px at 10 m, but verify: per tile, compute phase-correlation shift between SAR texture and a grey-scaled optical; reject or warp tiles with |shift| > 1 px. Misregistration teaches the generator to blur edges — it's invisible in spot checks and devastating in aggregate.
6. **Tiling:** 256² for experimentation, 512² for the production model. Stride = tile (no overlap) for training; record (CRS, transform, MGRS, slope from DEM, LC histogram, dates, gap, cloud%) per tile in a parquet manifest. Keep tiles as compressed GeoTIFF/COG or pack to LMDB/WebDataset/Zarr for I/O throughput.
7. **Versioning:** DVC (or git-lfs/lakeFS) from day one; every experiment logs the dataset hash.

### 4.3 Input-channel design (an underrated lever)

- **Dual-pol stack:** [VV_dB, VH_dB, VV−VH (ratio in dB)] as 3 channels — the ratio channel adds vegetation/moisture discrimination nearly free. With VV-only data, replicate or add texture features (GLCM) — marginal.
- **DEM + slope channels:** strongly recommended for Himalayan terrain; gives the network the geometry causing layover/shadow.
- **Multi-temporal SAR:** 3–5 dates stacked (or mean+std) → big quality jump for vegetation and water; costs pairing complexity (Section 5.7).
- **Metadata conditioning:** sin/cos of day-of-year, latitude band, orbit direction — injected via FiLM/embedding. Lets you *control* season at inference (generate "summer view" from winter SAR) and resolves part of the one-to-many ambiguity explicitly.

### 4.4 Splits and leakage (do this right or all numbers are fiction)

- **Split geographically**: by MGRS tile or ≥ 50–100 km blocks; ensure no block appears in two splits. Adjacent 256² patches from one scene share content — a random patch split leaks and can inflate SSIM/FID dramatically.
- Hold out **(a) unseen regions** within trained biomes, **(b) one entire unseen biome**, and **(c) one unseen season** as three separate test tracks — report all three; they answer different generalization questions.
- Freeze the test set in month 2 and never look at per-tile test outputs until month 6 selection.

### 4.5 QC gates (automate as pipeline asserts)

Reject pair if: cloud/shadow mask > 1% • |temporal gap| > threshold • registration shift > 1 px • SAR border/no-data present • optical saturated > 0.5% • near-duplicate (perceptual hash) of an existing tile • NDVI inconsistency between optical and expected LC (catches snow/flood label-noise pairs). Log rejection reasons; the rejection histogram is itself a data-quality dashboard.


---

## 5. The model landscape: every family, trade-offs, when to use what

### 5.0 Decision matrix (summary — details below)

| Family | Quality ceiling | Training stability | Inference cost | Data need | Hallucination control | Use when |
|---|---|---|---|---|---|---|
| U-Net + L1/SSIM (regression) | Low (blurry) | Excellent | Trivial | Low | Best (it averages, doesn't invent) | Baseline; plumbing validation; when blur is acceptable |
| Pix2Pix | Medium | Good | Trivial | Medium | Medium | First adversarial result; ≤256² |
| **Pix2PixHD-class cGAN** | High | Medium (manageable with §7) | Trivial (1 fwd pass) | Medium–high | Medium | **Production default**; 512²+; tight latency |
| Attention/Transformer cGANs | High+ | Medium | Low | High | Medium | When cGAN plateaus on texture/edges |
| CycleGAN / CUT (unpaired) | Medium | Medium | Trivial | Unpaired only | Poor (geometry drift) | Only when pairing is impossible (it isn't, for S1/S2) |
| BicycleGAN / cVAE-GAN | Medium | Medium | Trivial | Medium | Medium (explicit diversity) | Need multiple diverse outputs per input from a GAN |
| **Conditional diffusion (pixel)** | **Highest (256–512²)** | Excellent (just MSE) | High (needs distillation) | High | Medium (stochastic; CFG tunable) | Quality-critical track; you have GPU-weeks |
| Latent diffusion + ControlNet | Highest at high res | Excellent | Medium–high | Medium (prior helps) | **Risky** (natural-image prior invents) | High res on a budget; accept prior-driven hallucination risk & audit hard |
| Flow matching / rectified flow | ≈ diffusion | Excellent | Medium (few-step) | High | Medium | Greenfield diffusion-track in 2026; fewer sampling steps natively |
| **TerraMind (EO foundation model)** | High (token-level, coarser texture) | Fine-tune only | Medium | **Tiny (zero-shot works)** | Medium | **Week-2 baseline**; data-scarce AOIs; multimodal roadmap |

### 5.1 Deterministic regression (build first, keep forever)

U-Net (or UNet++/Attention-U-Net) with L1 + MS-SSIM. Why it matters: it is the *conditional mean* estimator — blurry but never inventive, trivially exportable, and the reference that tells you how much of your final model's sharpness is real information vs adversarial invention. Also your data-pipeline canary: if this won't converge, no GAN will.

### 5.2 Conditional GANs — the production workhorse

**Pix2Pix** (Isola 2017): U-Net generator + 70×70 PatchGAN discriminator, loss = cGAN + λ·L1 (λ=100). *Why it works:* L1 handles low frequencies (overall structure/color), the PatchGAN models high-frequency texture as a Markov random field over local patches — a division of labor matched to the blur problem. Limits: ~256², texture repetition, instance-level artifacts.

**Pix2PixHD** (Wang 2018) — the recommended cGAN: coarse-to-fine generator (global G1 + local enhancer G2), **multi-scale discriminators** (2–3 PatchGANs at 1×, ½×, ¼×) so the model is judged at multiple receptive fields, plus **feature-matching loss** (match D's intermediate activations between real and fake — a stabilizer that acts like a learned perceptual loss) and **VGG perceptual loss**. Use LSGAN or hinge objective, not vanilla BCE. This is the architecture most SAR2EO production systems still ship.

**Upgrades worth ablating:** spectral norm on D (+G), self-attention blocks at 32²/16² feature maps (SAGAN-style) for long-range coherence (road networks!), residual/dense generator blocks, Swin/Restormer-style transformer generators (the post-2022 SAR2Opt literature's main gains came from exactly these: better generators, multi-scale Ds, perceptual/similarity losses).

**Trade-offs:** one forward pass at inference (real-time tiling, easy ONNX/TensorRT); deterministic output (auditable, cacheable); but adversarial training needs babysitting (§7), can mode-drop rare land covers (fix: LC-balanced sampling via WeightedRandomSampler — you've done this), and sharpness is partly invention — audit it.

### 5.3 Unpaired translation (CycleGAN, CUT, MUNIT)

Cycle-consistency (CycleGAN) or patch-contrastive (CUT — cheaper, often better) objectives when no pairs exist. **For S1↔S2 you can always build pairs, so treat unpaired methods as a research footnote**: they hallucinate more, drift geometry (cycle loss tolerates invertible "steganographic" cheating), and underperform paired training at equal data. Legitimate niche: adapting to a sensor for which you have SAR but no co-located optical (e.g., a new airborne system) — then consider semi-supervised: paired loss on S1/S2 + unpaired adversarial on the new sensor.

### 5.4 Conditional diffusion models — the quality ceiling

**Mechanics:** forward process noises the optical target; a U-Net (with attention) is trained to denoise, **conditioned on the SAR tile by channel-concatenation** at every step (plus optional cross-attention for metadata). Train with noise-prediction (ε) or better **v-prediction**, cosine schedule, **min-SNR-γ loss weighting** (faster convergence), EMA weights, T=1000 train steps. Inference with DDIM/DPM-Solver++ at 25–50 steps.

*Why diffusion beats GANs here:* the objective is a stable regression at every noise level (no adversarial game → no mode collapse), the model covers the full conditional distribution (better diversity over the one-to-many ambiguity), and recent S2O literature (CM-Diffusion and successors, 2024–2025) reports SOTA FID/LPIPS over GAN baselines, with explicit color-consistency mechanisms addressing the classic diffusion color-shift problem.

**Classifier-free guidance (CFG):** drop the SAR condition 10% of training steps; at inference, guide with scale s ≈ 1.5–3. Higher s → sharper, more condition-faithful structure but oversaturated colors and *more confident hallucination*. Tune on the hallucination audit, not on looks.

**Trade-offs:** training is compute-hungry (GPU-weeks, §10) and inference is 25–50 forward passes. Fixes: **distillation** — consistency models / LCM / progressive distillation compress to 1–4 steps with modest quality loss, restoring near-GAN throughput. Stochasticity: great for showing analysts *an ensemble* of plausible optical interpretations (honest uncertainty visualization — a feature for defence use), but fix the seed when determinism is required. Per-pixel std over N samples = a free **uncertainty map**; ship it.

**Variants:** *Latent diffusion* (run diffusion in a VAE latent, à la Stable Diffusion) — 8× cheaper per step at 512², but the VAE can smear fine radiometry; train/fine-tune the VAE on EO data or use TerraMind's S2 tokenizer. *ControlNet on Stable Diffusion:* bolt a trainable SAR-conditioned control branch onto frozen SD — fastest route to gorgeous 512–1024² results with only 10⁴–10⁵ pairs, **but** the natural-image prior eagerly invents photorealistic detail that was never in the SAR; for analysis use, this needs the strictest hallucination auditing. *BBDM* (Brownian-Bridge Diffusion): builds the bridge directly between SAR and EO domains instead of noise→image — elegant for I2I, competitive results. *Flow matching / rectified flow:* the 2024–2026 successor framework — straighter probability paths, few-step sampling natively; if your diffusion track starts from scratch in 2026, seriously consider flow matching over DDPM.

### 5.5 EO foundation models — the new baseline economics

**TerraMind** (IBM + ESA Φ-lab + FZJ, open-sourced 2025; tiny/small/base/large on Hugging Face; fine-tuning via TerraTorch): the first **any-to-any generative** EO foundation model, pretrained on ~9M globally distributed multimodal samples (TerraMesh) including S1 GRD/RTC and S2 L1C/L2A — meaning **S1→S2 generation works out of the box**, plus DEM/LULC/NDVI as bonus modalities and "Thinking-in-Modalities" (generate intermediate modalities to help a downstream task). Caveats: generation is token-based (FSQ-VAE tokens decoded with diffusion) → excellent semantics/structure, sometimes coarser fine texture than a bespoke pixel model; resolution tied to S2-like 10 m products. **Action:** zero-shot it in week 2 as the bar to beat; fine-tune on your AOIs in month 4; if it wins your task-based eval, you just saved a training phase. Related encoders for transfer learning (not generative): Prithvi-EO, Clay, DOFA, SatMAE — useful as perceptual-loss backbones or generator initializations.

### 5.6 Hybrids and boosters

- **Two-stage:** regression U-Net predicts the conditional mean; a GAN/diffusion *refiner* adds texture residuals. Decouples structure (trustworthy) from texture (invented) — auditable and stable.
- **Multi-task:** predict LULC segmentation alongside the optical image (shared encoder). The auxiliary task regularizes semantics and gives you a free QC signal (does predicted LC match generated colors?).
- **Self-supervised SAR pretraining:** MAE/SimCLR pretrain the generator encoder on abundant *unpaired* SAR before paired training — helps most in low-data AOIs.
- **Despeckling pretext:** SAR2SAR/MERLIN-style denoiser as encoder init — modest gains, cheap.

### 5.7 Multi-temporal input (the cheat code)

Feed a short time series of SAR (3–5 dates, stacked channels or a temporal encoder) → speckle averages out, phenology becomes visible, water/crop confusion collapses. The cloud-removal literature (SEN12MS-CR-TS; sequential diffusion S2O, 2025) consistently shows large gains. Cost: pairing logistics and a doubled data pipeline. If your operational concept allows multi-date SAR (it usually does for monitoring), schedule this as the Month-5 ablation most likely to pay off.

---

## 6. Loss function catalogue

| Loss | Formula sketch | What it buys | Watch out | Typical weight |
|---|---|---|---|---|
| L1 (pixel) | ‖y−ŷ‖₁ | Low-freq fidelity, color anchoring | Blur if alone | 50–100 (GAN), implicit in diffusion |
| Charbonnier | √((y−ŷ)²+ε²) | Smooth L1; robust to outlier pixels | — | swap-in for L1 |
| MS-SSIM | structural sim. | Local contrast/structure | Color-blind; pair with L1 | 1–5 |
| Adversarial: LSGAN / **hinge** | least-squares / hinge margins | Sharp texture; hinge = stable gradients, no saturation | Vanilla BCE saturates — avoid | 1 |
| WGAN-GP | Wasserstein + grad penalty | Very stable D signal | Slower; GP costs a backward pass | 1 (GP λ=10) |
| **Feature matching** (Pix2PixHD) | ‖D_feat(y)−D_feat(ŷ)‖₁ | Stabilizes G; learned perceptual | Needs multi-scale D | 10 |
| Perceptual (VGG / LPIPS) | feature distance in pretrained net | Texture realism, fewer artifacts | ImageNet prior ≠ EO; consider an EO-pretrained backbone (Prithvi/Clay features) | 5–10 |
| **FFT / focal-frequency loss** | distance in Fourier domain | Fights spectral bias → real high-freq detail without GAN invention | Tune band weighting | 1–10 |
| Edge (Sobel/Canny-soft) | gradient-map L1 | Road/boundary crispness | Can ring | 1–5 |
| SAM (spectral angle) | angle between band vectors | **Multispectral** outputs: spectral shape fidelity (your ENVI background applies directly) | Scale-invariant — pair with L1 | 1–5 |
| Color/histogram (CM-Diffusion-style) | match channel stats/palette memory | Kills the diffusion color-shift | — | small |
| Diffusion ε/v-MSE + min-SNR-γ | weighted denoising MSE | The whole objective; min-SNR ≈ 3–5× faster convergence | use v-pred + cosine schedule | γ=5 |
| R1 penalty (on D) | ‖∇D(real)‖² | Convergence guarantee-ish; tames D | Lazy reg every 16 steps | γ=1–10 |

**Recommended composites:** cGAN track → hinge + 100·L1 + 10·FM + 10·VGG (+ 5·FFT if edges soft, + 2·SAM if multispectral). Diffusion track → v-MSE(min-SNR) + small color-consistency term; add LPIPS only in distillation. Resist loss-soup: add one term per ablation, keep what moves the *task-based* metric.

---

## 7. Training playbook

### 7.1 GAN stabilization stack (apply in this order)

1. **Objective:** hinge or LSGAN. Never saturating BCE (you've derived why: G's gradient vanishes when D is confident).
2. **Spectral normalization** on every D conv (cheap Lipschitz control). Optionally on G.
3. **TTUR if D/G balance breaks:** start symmetric (G=D=2e-4, Adam β=(0.5, 0.999)); if D dominates (D loss → 0, G loss climbs, outputs get artifacts), go G=1e-4 / D=4e-4 *or* add R1 (γ=1–10, lazy every 16 steps) *or* reduce D capacity — change one thing at a time.
4. **EMA of G weights** (decay 0.999): evaluate/ship the EMA copy only; it's consistently better and smoother.
5. **Effective batch ≥ 16** at 512² via gradient accumulation; instance norm (or GroupNorm) in G so small per-GPU batches don't poison statistics.
6. **DiffAugment/ADA** (color+translation+cutout on both real & fake before D) when pairs < ~30–50k — prevents D memorization on small custom AOIs.
7. **LC-balanced sampling** (WeightedRandomSampler over land-cover histograms) so rare classes (water, bare rock, snow) aren't mode-dropped.

**Reading the curves:** healthy hinge-GAN: D loss oscillates around a band (≈0.3–0.7 per branch), G adversarial term noisy-flat, L1/VGG steadily ↓, FID ↓ then plateaus. Pathologies: D→0 = D winning (see #3); G adv ↓ while FID ↑ = mode collapse (raise D capacity/augment, check sampler balance); periodic loss spikes = lr too high or bad batches (inspect the offending tiles — usually QC escapes); great train-FID but bad val-FID = leakage or overfit small data (geographic split! DiffAugment!).

### 7.2 Diffusion training notes

v-prediction + cosine schedule + min-SNR-γ(5) + EMA 0.9999; lr 1e-4 AdamW(0.9,0.999) wd 0.01, warmup 5–10k; batch ≥ 64 at 256² (accumulate); train 300k–800k steps watching val-FID every 10–25k (sample with fixed seeds, DDIM-50). Condition dropout 10% for CFG. Diffusion *training* rarely diverges — your problems will be throughput and evaluation cost, so automate sampling-eval as a separate lower-priority GPU job.

### 7.3 Throughput engineering

AMP (bf16 preferred) everywhere; channels_last memory format; `torch.compile` the G/U-Net; DDP with static graph; dataloader: WebDataset/LMDB shards + pinned memory + 8–16 workers/GPU (GeoTIFF-per-tile reading will bottleneck — pack shards); pre-normalize offline so the loader only decodes+augments; profile one epoch with the PyTorch profiler before scaling — a 30-min profile typically buys back days.

### 7.4 Hyperparameters that actually matter (sweep order)

λ_L1 (50/100/200) → lr & TTUR ratio → D scales (2 vs 3) → input channels (VV+VH+ratio vs VV+VH vs +DEM) → tile size (256 vs 512) → speckle handling (raw dB vs temporal multilook) → perceptual backbone. Everything else is noise at this project's scale. Run sweeps at 256² with 1/4 data, confirm the top-2 at full scale.

---

## 8. Evaluation protocol (where most projects fail)

### 8.1 The metric battery

| Metric | Measures | Trust it for | Don't trust it for |
|---|---|---|---|
| PSNR / RMSE | pixel error vs the *one* real image | regression baselines, regressions in plumbing | sharp generative models (penalizes any plausible mode ≠ GT) |
| SSIM / MS-SSIM | local structure | structure sanity | color, semantics |
| **LPIPS** | deep perceptual distance to GT | perceptual fidelity per-pair | distribution-level realism |
| **FID** (clean-fid impl., ≥5–10k samples) | distribution realism | model ranking, convergence tracking | small val sets (<2k → use **KID**), per-image judgment |
| SAM / per-band RMSE / ERGAS | spectral fidelity (multispectral) | radiometric usability | RGB-only products |
| Edge F1 (Canny on GT vs gen) | boundary fidelity | roads/field edges | texture interiors |

Report mean ± CI **stratified by**: land-cover class, terrain-slope bins, season, temporal-gap bins, and per held-out region. Aggregate single numbers hide exactly the failure modes that matter operationally.

### 8.2 The cardinal rule

All numbers on the **geographic** test split, fixed since Month 2. Any metric computed on randomly-split patches is marketing, not measurement.

### 8.3 Task-based evaluation (the gold standard)

Train a downstream model (e.g., DeepLab/U-Net land-cover segmentation, or building extraction) **on real EO**; evaluate its accuracy when fed (a) real EO vs (b) your generated EO on the same test tiles. The mIoU/F1 gap is the most honest single statement of utility — it measures whether the translation preserves *information*, not just looks. Secondary variant: data-augmentation value (does adding generated pairs improve a SAR-side classifier?).

### 8.4 Hallucination audit (non-negotiable for defence use)

1. Run a building/ship/vehicle-scale object detector (or building-footprint model) on real vs generated EO over the same tiles; count **phantom objects** (in gen, not real) and **vanished objects** (in real, not gen) → report phantom-rate / vanish-rate per km², stratified by class and terrain.
2. Edge-map agreement and segment-boundary displacement stats (does the model move field boundaries?).
3. For diffusion: sample N=8 per input; per-pixel std = uncertainty map; flag tiles whose samples *disagree on objects* — those are exactly where the SAR underdetermines the scene.
4. Maintain a curated **failure gallery** (worst-100 by LPIPS + worst-100 by phantom count) reviewed monthly; it drives the next ablation better than any aggregate metric.

### 8.5 Human evaluation

A small forced-choice study with 3–5 image analysts: (real-vs-fake discrimination rate, preference between model variants, and "would this help you orient on the SAR scene?" Likert). 30 minutes per analyst, run in months 3 and 6. Aligns the project with the actual consumer of the imagery.

---

## 9. Production engineering

### 9.1 Repository shape

```
sar2eo/
  configs/            # hydra: data/*.yaml model/*.yaml loss/*.yaml train/*.yaml
  src/
    data/             # readers (rasterio), pairing, QC gates, shards, transforms
    models/           # generators/, discriminators/, diffusion/, registry.py
    losses/           # composable, weight-configured
    metrics/          # fid_kid.py lpips.py spectral.py hallucination.py
    engine/           # train loops (gan.py, diffusion.py), EMA, ckpt mgr
    inference/        # tiler.py (overlap+Hann blend), writer.py (COG+tags)
  scripts/            # build_dataset.py validate_pairs.py leaderboard.py
  tests/              # unit: losses shapes/grads, tiler seam test, QC gates
  envs/               # Dockerfile + Singularity def (HPC), lockfiles
```

Config-driven everything (Hydra/OmegaConf); experiment tracking with **MLflow local or W&B offline** (air-gapped friendly — sync later if ever); every run logs: git SHA, config, dataset DVC hash, seed, env hash. CI runs unit tests + a 50-step smoke train on a 100-tile fixture.

### 9.2 Reproducibility & air-gapped reality

Pin everything (uv/conda-lock); build wheels cache offline (you've done offline pip — same drill); Docker for dev, **Singularity/Apptainer** for HPC; `torch.use_deterministic_algorithms(True)` for eval runs (train can stay fast/non-det); store checkpoints + EMA + optimizer state every N steps with automatic resume — preemptible-safe.

### 9.3 Inference service

- **Tiled inference:** window 512, overlap 64–128, Hann/feather blending of overlaps (kills seams); reflect-pad scene borders; preserve CRS/transform via rasterio; write **COG** with overviews.
- **Provenance:** embed `AI_GENERATED=true`, model name+version+hash, dataset version, generation date, and (for diffusion) seed & guidance scale into GeoTIFF tags + sidecar JSON; optionally a subtle visible watermark for analyst-facing products. This is a hard requirement in an intelligence chain.
- **Determinism:** GANs are deterministic; diffusion → fix seed per scene for reproducible products, expose `n_samples` for ensemble/uncertainty mode.
- **Export:** GAN G → ONNX → TensorRT (fp16) trivially; diffusion → torch.compile or TensorRT for the U-Net + keep the sampler in Python, or ship the distilled 2–4-step student.
- **Batch pipeline:** simple queue (filesystem/SLURM array) over scenes; per-scene QC report auto-generated (metrics vs nearest cloud-free real EO when available, hallucination flags, uncertainty map thumbnail).
- **Monitoring in operation:** input-drift checks (dB histograms vs training distribution per orbit/season; alert on shift), output auto-QC (NDVI plausibility per LC, saturation %, seam detector), and a feedback channel from analysts to the failure gallery.

### 9.4 Model governance

A model card per release: intended use (visualization/orientation/gap-filling), **prohibited use (object-level intelligence, change detection on generated pixels, mensuration)**, training data summary, eval table incl. phantom/vanish rates, known failure modes (steep terrain, new sensors, snow), and the exact reproduction bundle.

---

## 10. Compute budgeting (plan, don't discover)

Rule-of-thumb planning numbers (order-of-magnitude, fp16/bf16):

| Job | Hardware | Wall time |
|---|---|---|
| SAR RTC preprocessing, ~10k S1 scenes (SNAP) | CPU cluster — embarrassingly parallel, ~10–30 min/scene/core-set | days on your 120-node CPU HPC (its perfect job) |
| U-Net baseline 256² | 1× 24 GB GPU | < 1 day |
| Pix2PixHD 512², 300–600k iters, batch 16 | 4× A100/H100 (or 8× 24 GB) | ~3–7 days |
| Conditional DDPM 256² from scratch → good FID | 8× A100 | ~1–2 weeks |
| Latent diffusion / ControlNet fine-tune 512² | 4× A100 | ~2–4 days |
| TerraMind fine-tune (TerraTorch) | 1–4× A100 | hours–2 days |
| Distillation (consistency/LCM) | 4× A100 | ~1–3 days |
| Inference: GAN 512² tiles | 1× A100 | 30–100 tiles/s |
| Inference: diffusion DDIM-50 / distilled-4 | 1× A100 | ~1–2 / ~15–40 tiles/s |

**Hard truth:** the 120-TFLOPS CPU cluster cannot train these models in your window — generative training is GPU-bound. Minimum viable: one 4× A100-80GB (or 8× RTX 4090/L40S) node for the training phase; the CPU cluster earns its keep on preprocessing, QC, metric computation (FID feature extraction parallelizes), and batch CPU inference of the exported GAN if needed. Budget ~20–30% of GPU time for ablations/sweeps, not just hero runs.

---

## 11. Risk register: failure modes → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Everywhere-soft edges despite GAN | Sub-pixel misregistration in pairs | tighten QC shift gate; per-tile phase-correlation re-warp |
| Seasons look "averaged" / wrong colors | Temporal gap label-noise | tighten gap threshold; add day-of-year conditioning |
| Great val numbers, ugly new-region output | Patch-level split leakage | geographic split; re-baseline all numbers |
| D loss → 0, G diverges | D overpowering | TTUR / R1 / shrink D / DiffAugment |
| FID rises late in training, outputs samey | Mode collapse / D memorization | EMA eval, ADA/DiffAug, balanced sampler, early-stop on FID |
| Checkerboard artifacts | Transposed-conv | resize-conv (nearest+conv) upsampling |
| Tile seams in products | naive stitching | overlap + Hann blending (test in CI with a seam unit test) |
| Diffusion color shift / oversaturation | CFG too high; no color anchor | s≤3; color-consistency loss; sample-time histogram match to L1-baseline output |
| Confident fake buildings in empty fields | Prior-driven hallucination (esp. SD/ControlNet) | lower CFG; two-stage (regression structure + refiner); phantom-rate gate blocks release |
| Mountain faces garbage | Layover/shadow underdetermined | DEM+slope channels; mask/flag high-slope outputs; report stratified metrics honestly |
| New sensor (X/L-band, airborne) fails | Wavelength/physics shift | expected — fine-tune per sensor; never claim cross-band transfer |
| Metrics disagree (FID↓ but task-mIoU↓) | Realism ≠ information | trust task-based + audit; FID is a tiebreaker only |

---

## 12. Reading list & what to watch

### Papers — foundations
Goodfellow et al. 2014, *GANs* • Mirza & Osindero 2014, *Conditional GANs* • Isola et al. 2017, *Pix2Pix* • Wang et al. 2018, *Pix2PixHD* • Zhu et al. 2017, *CycleGAN* • Park et al. 2020, *CUT* • Zhu et al. 2017, *BicycleGAN*.

### Papers — GAN training science
Arjovsky 2017 *WGAN* • Gulrajani 2017 *WGAN-GP* • Miyato 2018 *Spectral Norm* • Mescheder 2018 *Which GAN methods actually converge?* (R1) • Heusel 2017 *TTUR/FID* • Karras 2020 *ADA* • Zhao 2020 *DiffAugment* • Brock 2019 *BigGAN* (the tricks appendix).

### Papers — diffusion & successors
Ho 2020 *DDPM* • Nichol & Dhariwal 2021 *Improved DDPM* • Song 2021 *DDIM* • Dhariwal & Nichol 2021 *Diffusion beats GANs* • Ho & Salimans 2022 *Classifier-free guidance* • Rombach 2022 *Latent Diffusion* • Zhang 2023 *ControlNet* • Karras 2022 *EDM* (design-space — read before building any diffusion) • Hang 2023 *Min-SNR* • Salimans 2022 *Progressive distillation* • Song 2023 *Consistency Models* • Lipman 2023 *Flow Matching* / Liu 2023 *Rectified Flow*.

### Papers — SAR↔EO specifically
Schmitt 2018 *SEN1-2* • Schmitt 2019 *SEN12MS* • Fuentes Reyes 2019 *SAR-to-optical with cGANs* (Remote Sensing) • Zhao 2022 *SAR2Opt benchmark / comparative GAN analysis* (IEEE GRSL) • Ebel 2021/2022 *SEN12MS-CR / -CR-TS* • Meraner 2020 *DSen2-CR* (cloud removal, SAR-guided) • Li 2023 *BBDM* (CVPR) • *CM-Diffusion: SAR-to-optical diffusion with color memory* (IEEE TGRS-family, 2024) • *Brain-inspired diffusion for S2O* (Frontiers 2024 — good survey of GAN-era improvements in its related work) • *Sequential SAR-to-Optical with conditional diffusion* (Remote Sensing 2025 — the multi-temporal direction) • Huang 2021 *QXS-SAROPT* • Shermeyer 2020 *SpaceNet 6* • Jakubik et al. 2025 *TerraMind* + *TerraMesh* (arXiv; code: github.com/IBM/terramind; models on Hugging Face `ibm-esa-geospatial`).

### Papers — SAR fundamentals & despeckling
Moreira et al. 2013, *A Tutorial on SAR* (IEEE GRSM — the canonical primer) • Dalsasso 2021 *SAR2SAR* • *MERLIN* (self-supervised despeckling) • ASF HyP3 RTC docs; ESA SNAP S1 toolbox docs.

### What to watch / take as courses
- **Stanford CS236 — Deep Generative Models** (full lectures on YouTube): the theory spine — GAN game, VAEs, score/diffusion, in one coherent arc.
- **Hugging Face Diffusion Models Course** (free, hands-on) + the *Annotated Diffusion Model* blog post — implement DDPM line by line.
- **Lilian Weng's blog**: "From GAN to WGAN" and "What are Diffusion Models?" — the best concise derivations in existence; pairs with your first-principles loss work.
- **MIT 6.S191** generative lectures — fast refresher for teammates.
- **NASA ARSET SAR webinars** (free) + **ESA EO College / SAR-EDU** — SAR preprocessing done right; ideal for whoever owns the data pipeline.
- **Outlier — "Diffusion Models | Paper Explanation"** and **Yannic Kilcher's** DDPM/LDM/ControlNet videos — efficient paper digestion.
- **CVPR diffusion-model tutorial recordings (2022/2023)** — design-space and sampler intuition from the authors themselves.
- **ESA Φ-lab TerraMind materials + TerraTorch docs** — for the foundation-model track.

---

## 13. Appendix: the default recipe (if you start tomorrow)

**Data:** SEN12MS (primary) + SEN1-2 (volume) + custom AOI pairs (gap ≤ 5 d, cloud ≤ 1%, shift ≤ 1 px). Inputs: [VV, VH, VV−VH] dB clipped/normalized + DEM-slope channel. Targets: S2 L2A B4/B3/B2 (+B8). 256² dev / 512² prod. Geographic split by MGRS, frozen.
**Track A (production cGAN):** Pix2PixHD G (global+enhancer, resize-conv ups), 2× multi-scale spectral-norm PatchGAN D, hinge + 100·L1 + 10·FM + 10·VGG; Adam(0.5,0.999) lr 2e-4/2e-4 (TTUR 1e-4/4e-4 if D dominates), R1 γ=1 lazy-16, EMA 0.999, eff. batch 16, DiffAugment if <30k pairs, LC-balanced sampler, 300k iters, AMP+DDP, eval FID/LPIPS every 5k on fixed val, early-stop on FID plateau, ship EMA.
**Track B (quality):** week 2 — TerraMind zero-shot baseline; month 4 — TerraMind fine-tune; month 5 — conditional pixel-diffusion 256→512 (v-pred, cosine, min-SNR-5, CFG-dropout 10%, DDIM-50, s=2) → consistency-distill to 4 steps if it wins.
**Gatekeepers:** geographic test only; FID/LPIPS + LC-segmentation transfer gap + phantom/vanish rates; failure gallery weekly; analyst study months 3 & 6.
**Ship:** EMA weights → ONNX/TensorRT, tiled-blend inference to COG with AI-generated provenance tags, model card with prohibited uses, reproducibility bundle (config+DVC hash+seed+container).

*Generated EO is a rendering of what the SAR is consistent with — treat every pixel as hypothesis, label it as such, and the system will be both useful and safe.*
