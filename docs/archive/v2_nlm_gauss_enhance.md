# Archived v2 result: sigma-adaptive NLM as the gauss_noise enhancement

v2 (submitted 2026-07-12) used sigma-adaptive Non-Local Means
(`h = 0.8·σ̂` Immerkaer, cap 30) as the matched enhancement for Gaussian
noise. v3 replaces it with **non-blind BM3D** fed the logged per-image sigma
from `snr_index.csv` (config: `enhancement.gauss_method`; `nlm` restores the
old behaviour). The full v2 tables are in git history with the 2026-07-12
submission; the NLM gauss-enhance cells are reproduced below.

## Why it was replaced — the diagnosis

NLM-enhanced gauss images scored *lower* than the raw noisy images on every
pixel-precise task at every severity. Three measurements pin the mechanism:

1. **Pixel fidelity vs task fidelity.** NLM did improve PSNR
   (25.3→29.0 / 19.8→25.5 / 16.1→23.5 dB at low/med/high, 12-image sample)
   — it succeeded *as a denoiser* while failing *as a pre-processor*.
2. **It kept noise AND lost texture.** The high-frequency energy of the NLM
   output is *above* the clean image (Laplacian-std ratio 1.14/1.36/1.50 at
   low/med/high): blotchy residual noise survives while true fine texture is
   flattened — the worst combination for texture-driven recognition. BM3D at
   the same severities: 0.90/0.75/0.55 with +4-5 dB PSNR.
3. **The damage concentrates exactly where texture matters.**
   - Detection at gauss/low: small/medium AP *drop* (0.137→0.133 /
     0.337→0.327) while large-object AP rises (0.428→0.443) — the
     smoothing trade, visible only size-stratified.
   - Keypoints: medium persons lose 1.6–2.2× more than large ones
     (e.g. high severity −0.067 vs −0.031).
   - Panoptic: SQ (boundary quality of matched segments) barely moves
     (−0.014 at high) while RQ (whether segments are recognised at all)
     collapses (−0.058), and texture-defined *stuff* classes lose
     proportionally more than *things* (low severity: stuff −20%,
     things −8.6%) — denoising erased the class evidence itself.

Also measured: Immerkaer under-estimates the true logged sigma under heavy
clipped noise (est 27.6 vs true 44.2 at high severity) — so v2's NLM was
*under*-dosed at med/high and still destroyed texture, i.e. the failure is
NLM's texture-cost-per-unit-of-denoising, not the dose. That is why v3 both
switches the filter (BM3D) and goes non-blind (logged sigma), mirroring the
motion-blur Wiener design.

## v2 NLM gauss-enhance cells (1,521-image subset)

| task | severity | clean | distorted | enhanced (NLM) | recovery |
|:--|:--|--:|--:|--:|--:|
| detection (mAP) | low | 0.352 | 0.293 | 0.288 | −0.004 |
| detection (mAP) | med | 0.352 | 0.201 | 0.213 | +0.013 |
| detection (mAP) | high | 0.352 | 0.097 | 0.154 | +0.056 |
| features (ORB) | low | 1.000 | 0.834 | 0.800 | −0.034 |
| features (ORB) | med | 1.000 | 0.731 | 0.670 | −0.061 |
| features (ORB) | high | 1.000 | 0.616 | 0.538 | −0.079 |
| keypoints (OKS) | low | 0.657 | 0.590 | 0.558 | −0.031 |
| keypoints (OKS) | med | 0.657 | 0.489 | 0.441 | −0.048 |
| keypoints (OKS) | high | 0.657 | 0.369 | 0.316 | −0.053 |
| segmentation (PQ) | low | 0.410 | 0.359 | 0.317 | −0.043 |
| segmentation (PQ) | med | 0.410 | 0.294 | 0.240 | −0.055 |
| segmentation (PQ) | high | 0.410 | 0.209 | 0.163 | −0.046 |

detection `finetuned+enh` on NLM images: low 0.244 / med 0.207 / high 0.164.
