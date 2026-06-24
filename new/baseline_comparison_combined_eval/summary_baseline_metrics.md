# Baseline Comparison Summary

| Method | PSNR ↑ | SSIM ↑ | Inference seconds ↓ |
|---|---:|---:|---:|
| Low-light | 9.2119 | 0.3411 | 0.000000 |
| Gamma | 13.5447 | 0.6746 | 0.001431 |
| CLAHE | 10.7316 | 0.4810 | 0.000351 |
| MSRCR | 15.6260 | 0.7271 | 0.085891 |
| GAN | 17.2752 | 0.7693 | 0.065050 |

PSNR and SSIM are computed against the ground-truth images.
Higher PSNR and SSIM mean the output is closer to the ground truth.
