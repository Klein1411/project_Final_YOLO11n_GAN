# Baseline Comparison Summary

| Method | PSNR ↑ | SSIM ↑ | Inference seconds ↓ |
|---|---:|---:|---:|
| Low-light | 7.8023 | 0.1909 | 0.000000 |
| Gamma | 12.4624 | 0.6328 | 0.001398 |
| CLAHE | 9.4978 | 0.4093 | 0.001567 |
| MSRCR | 15.2938 | 0.6595 | 0.084566 |
| GAN | 18.5261 | 0.7810 | 0.394803 |

PSNR and SSIM are computed against the ground-truth images.
Higher PSNR and SSIM mean the output is closer to the ground truth.
