# Reading Roadmap — Generative AI Papers

A beginner-friendly guide to the 11 papers in this directory, in the order you should read them.

---

## Before You Start

You don't need to read every equation on first pass. For each paper, do two reads:
- **Pass 1** — Abstract, Introduction, figures, and conclusion. Get the "why."
- **Pass 2** — Architecture section and results. Get the "how."

Skip proofs and appendices until you feel comfortable.

---

## Phase 1 — What is a GAN? (Week 1)

**Start here. These two papers form the complete mental model.**

### Paper 1: GAN (Goodfellow, 2014)
**File:** `01_GAN_Goodfellow_2014.pdf`

The most important paper in this list. Read it first.

- **Focus on:** Section 1 (Introduction), Section 4 (training algorithm), Figure 1
- **Skip:** Section 3 (theoretical proofs) on first read
- **Key insight:** Two networks competing — generator tries to fool discriminator, discriminator tries to catch fakes. That tension is what makes GANs work.
- **After reading:** you should be able to explain what `d_loss` and `g_loss` mean in the trainer logs

### Paper 3: DCGAN (Radford, 2015)
**File:** `03_DCGAN_Radford_2015.pdf`

Read immediately after GAN. The "here's how you actually make it work" paper.

- **Focus on:** Section 3 (Architecture guidelines), Figure 1 (the generator diagram)
- **Skip:** Section 6 (investigating the internals) on first read
- **Key insight:** The `Conv2d(4×4, stride=2)` + BatchNorm + LeakyReLU pattern used everywhere in this codebase comes from this paper.
- **After reading:** open `src/gan_pipeline/models/dcgan.py` — you'll recognise every line

---

## Phase 2 — Architecture Building Blocks (Week 2)

**Two foundational architectures that pix2pix is built on.**

### Paper 2: VGG (Simonyan, 2014)
**File:** `02_VGG_Simonyan_2014.pdf`

Short and easy. Just 8 pages of actual content.

- **Focus on:** Table 1 (the network configurations), Section 2.1
- **Skip:** Sections 4–5 (classification experiments)
- **Key insight:** Deep stacks of small `3×3` convolutions. You're reading this because `VGGPerceptualLoss` extracts features from layers `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3` of this exact network.
- **After reading:** open `src/gan_pipeline/models/losses.py` and find `VGGPerceptualLoss` — the layer names will make sense

### Paper 4: U-Net (Ronneberger, 2015)
**File:** `04_UNet_Ronneberger_2015.pdf`

Only 8 pages. Very visual — Figure 1 tells the whole story.

- **Focus on:** Figure 1 (the architecture diagram), Section 2
- **Skip:** Medical image results
- **Key insight:** The encoder compresses the image, the decoder reconstructs it, but *skip connections* copy feature maps directly from encoder to decoder. This preserves fine spatial detail that would otherwise be lost in the bottleneck.
- **After reading:** open `src/gan_pipeline/models/unet.py` — the `enc1`–`enc8` / `dec1`–`dec8` pairs are exactly Figure 1

---

## Phase 3 — Loss Functions & Training Stability (Week 3)

**Why BCE alone isn't enough, and three better alternatives.**

### Paper 5: Perceptual Losses (Johnson, 2016)
**File:** `05_PerceptualLosses_Johnson_2016.pdf`

- **Focus on:** Figure 1 (the perceptual loss idea), Section 3 (feature reconstruction loss)
- **Skip:** Style transfer results on first read
- **Key insight:** Instead of comparing pixels directly (L1/L2), compare *features* extracted from a pretrained network. The resulting images look sharper and more natural. This is the concept behind `lambda_vgg` in training.
- **After reading:** the `VGGPerceptualLoss` formula in `losses.py` will be obvious

### Paper 6: WGAN (Arjovsky, 2017)
**File:** `06_WGAN_Arjovsky_2017.pdf`

The math is heavy — don't get stuck on the proofs.

- **Focus on:** Section 1 (Introduction — very well written), Section 3 (the algorithm pseudocode)
- **Skip:** Section 2 (theoretical analysis) on first read
- **Key insight:** BCE loss gives the discriminator a vanishing gradient problem — when it's too confident, the generator gets no useful signal. Wasserstein distance fixes this by measuring *how far apart* real and fake distributions are, not just whether they overlap.
- **After reading:** you'll understand why `LossType.WASSERSTEIN` exists as an alternative

### Paper 7: WGAN-GP (Gulrajani, 2017)
**File:** `07_WGANGP_Gulrajani_2017.pdf`

Read right after WGAN. It's the "fix for the fix."

- **Focus on:** Section 1 (Introduction), Section 4 (gradient penalty algorithm box)
- **Key insight:** WGAN requires constraining the discriminator. Weight clipping (original WGAN) is hacky. Gradient penalty enforces the constraint properly by penalising gradients that deviate from norm=1.
- **After reading:** look at `multiscale_gradient_penalty` in `losses.py` — the `(‖∇D(x̂)‖₂ − 1)²` term is straight from this paper

### Paper 8: LSGAN (Mao, 2017)
**File:** `08_LSGAN_Mao_2017.pdf`

The shortest conceptual leap. 10 minutes to read the key idea.

- **Focus on:** Section 1 (Introduction), Figure 1 (the sigmoid saturation problem), Section 3
- **Key insight:** BCE saturates when the discriminator is confident — the generator's gradient vanishes. Replace BCE with MSE against target labels (real→1, fake→0). One equation change, noticeably more stable training.
- **After reading:** find `LossType.LSGAN` in `losses.py` — it's two lines

---

## Phase 4 — Conditional Image Translation (Week 4)

**The two papers this entire codebase is built on. Read last.**

### Paper 9: pix2pix (Isola, 2017)
**File:** `09_pix2pix_Isola_2017.pdf`

The most directly relevant paper. Read every section.

- **Focus on:** Section 3 (the full method), Figure 2 (U-Net vs encoder-decoder), Figure 3 (PatchGAN), Table 1
- **Key insight 1:** Condition the discriminator on the *input* image — `cat([SAR, EO])` as discriminator input. Otherwise it has no way to check that the output matches the input.
- **Key insight 2:** PatchGAN — instead of one real/fake score per image, score every `70×70` patch independently. Penalises local texture, not just global structure.
- **Key insight 3:** L1 with `λ=100` handles low-frequency structure; adversarial loss handles high-frequency texture.
- **After reading:** `scripts/train_pix2pix.py` and `pix2pix_trainer.py` will read like pseudocode from this paper

### Paper 10: Spectral Normalization (Miyato, 2018)
**File:** `10_SpectralNorm_Miyato_2018.pdf`

Short. Focus on Section 1 and Section 2 only.

- **Key insight:** Normalise each weight matrix by its largest singular value. Keeps the discriminator Lipschitz-constrained cheaply — no gradient penalty, no clipping. One line of code: `nn.utils.spectral_norm(conv)`.
- **After reading:** search `spectral_norm` in `src/gan_pipeline/models/patchgan.py`

### Paper 11: pix2pixHD (Wang, 2018)
**File:** `11_pix2pixHD_Wang_2018.pdf`

The capstone paper. Read last — by now every piece will be familiar.

- **Focus on:** Section 3 (the full method), Figure 2 (coarse-to-fine generator), Figure 3 (multi-scale discriminator)
- **Key insight 1:** Multi-scale discriminator — three independent PatchGANs at three resolutions. Coarse scale catches layout errors; fine scale catches texture errors.
- **Key insight 2:** Coarse-to-fine generator — a global network generates a low-resolution prediction; a local network refines it at full resolution.
- **Key insight 3:** Feature matching loss — make the generator produce activations inside the discriminator that match those of real images. Dense per-layer training signal beyond the single adversarial scalar.
- **After reading:** `src/gan_pipeline/models/coarse_to_fine.py`, `multiscale_disc.py`, and `pix2pix_trainer.py` are the direct implementation of this paper

---

## Summary Schedule

| Week | Papers | Goal |
|---|---|---|
| 1 | GAN → DCGAN | Understand the adversarial game and how to build it with convolutions |
| 2 | VGG → U-Net | Understand the two architectures everything else is built on |
| 3 | Perceptual → WGAN → WGAN-GP → LSGAN | Understand why BCE is insufficient and the four alternatives |
| 4 | pix2pix → SpectralNorm → pix2pixHD | Read the two papers this codebase directly implements |

After week 4, open `scripts/train_pix2pixhd.py` and trace one training step end-to-end. Every design choice — the loss weights, the discriminator input format, the coarse pass — will have a paper behind it.
