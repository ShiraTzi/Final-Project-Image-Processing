# Archived v2 result: uniform clean+distorted fine-tuning mixture

v2 (submitted 2026-07-12) fine-tuned YOLOv8n on a 4,500-image train2017
subset with a per-image **uniform** mixture over 10 variants (clean + 9
distortion cells → 10% clean, no restored images), AdamW lr0=1e-4,
early-stopped at epoch 15/30. v3 replaces the mixture (25% clean, half of
the corrupted picks passed through the matched classical restorer, 9,000
images — knobs `finetune.clean_fraction` / `finetune.restored_fraction`) and
trains the full 30 epochs. The v2 checkpoint is preserved in git history
(`models/yolov8_finetuned.pt` @ v2 submission).

## Why it was replaced — two measured failures

1. **Clean forgetting poisoned the near-clean cells.** v2 scored 0.284 on
   clean images vs 0.352 pretrained (−0.068), and consequently *lost to
   doing nothing* on every low-severity cell. The best.pt selection split
   had the same 10% clean share, so checkpoint selection barely rewarded
   keeping clean accuracy.
2. **Deployment-domain mismatch.** The decision matrix deploys the detector
   on classically *restored* images, which v2 never trained on. On the
   median-filtered salt & pepper images the *pretrained* model (0.321–0.339)
   beat the *fine-tuned* one (0.263–0.269) at every severity; the stack only
   won a single cell (motion_blur/high).

## v2 → v3, detection mAP per cell (1,521-image subset)

| cell | distorted | ft v2 | ft v3 | stack v2 | stack v3 |
|:--|--:|--:|--:|--:|--:|
| clean | 0.352 | 0.284 | **0.310** | — | — |
| gauss low | 0.293 | 0.266 | **0.282** | 0.244 | **0.299** |
| gauss med | 0.201 | 0.233 | **0.240** | 0.207 | **0.279** |
| gauss high | 0.097 | 0.186 | 0.183 | 0.164 | **0.246** |
| s&p low | 0.284 | 0.273 | **0.292** | 0.269 | **0.299** |
| s&p med | 0.167 | 0.245 | **0.255** | 0.267 | **0.297** |
| s&p high | 0.079 | 0.203 | 0.202 | 0.263 | **0.288** |
| blur low | 0.294 | 0.259 | **0.282** | 0.275 | **0.299** |
| blur med | 0.157 | 0.202 | **0.211** | 0.240 | **0.261** |
| blur high | 0.059 | 0.114 | 0.113 | 0.174 | **0.189** |

v3 stack improved on **all 9 cells**; v3 finetuned-alone fixed the
low-severity regressions (worst −0.012, was −0.035) at the cost of ≤0.003
on the in-domain heavy cells. Note v3 stack numbers also benefit from the
v3 BM3D enhancement on the gauss cells (see
[v2_nlm_gauss_enhance.md](v2_nlm_gauss_enhance.md)); the salt & pepper and
motion-blur enhanced images are identical between v2 and v3, so those stack
columns isolate the training-mixture change.
