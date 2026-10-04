# The Promises and Pitfalls of Spectral-Normalized Neural Gaussian Processes

David Abboud, Weijie Cai, Caleb Chin, Aidan Tsang — University of Toronto

📄 **[Read the report](research_report.pdf)**

## Abstract
Uncertainty estimation is important in high-stakes settings. Spectral-normalized Neural Gaussian Processes (SNGP) provide a distance-aware framework for uncertainty quantification. We investigate the impact of incorporating distance-based metric learning objectives, such as supervised contrastive learning (SupCon) and Multi-Similarity Loss, into the SNGP framework. Our empirical results show that these objectives provide only marginal improvements in accuracy and out-of-distribution detection while degrading calibration. We postulate that enforcing overly tight class clusters harms uncertainty estimation by promoting overconfident predictions near decision boundaries. To address this, we replace the standard Random Fourier Feature (RFF) approximation with Deterministic Uncertainty Estimation (DUE) and Gaussian Mixture Models (GMMs), which mitigate vanishing uncertainty and improve calibration. Finally, we demonstrate that leveraging self-supervised pretraining with DINOv2 yields a more semantically meaningful embedding space, achieving 97.49% accuracy, ECE 0.01, and 93.62% CIFAR-100 OOD AUPR — which amounts to more than 6% improvement over the best contrastive variant.

## Key findings
- **Augmentation, not contrastive loss, drives the gains.** A SimCLR-style two-view pipeline adds +1.75% accuracy and +12.3 points on CIFAR-10-C; SupCon / Multi-Similarity losses add a little accuracy and OOD detection but worsen calibration.
- **Replacing RFF with a sparse variational GP (DUE)** consistently improves calibration and NLL.
- **SNGP on DINOv2 features** reaches 97.49% accuracy, 0.01 ECE, and 93.62% CIFAR-100 OOD AUPR (+6 points over the best contrastive variant).

| Method | CIFAR-10 Acc | ECE | NLL | CIFAR-10-C Acc | SVHN ds-AUPR | CIFAR-100 ds-AUPR |
|---|---|---|---|---|---|---|
| SNGP | 91.92 | 0.05 | 0.41 | 68.01 | 95.71 | 83.18 |
| SNGP + Aug | 93.67 | 0.02 | 0.30 | 80.31 | 97.32 | 86.78 |
| SupCon + SNGP | 94.43 | 0.04 | 0.30 | 81.22 | 96.74 | 85.06 |
| MS-SNGP | 94.14 | 0.04 | 0.26 | 80.66 | 97.67 | 88.07 |
| DUE | 95.32 | 0.03 | 0.21 | 74.41 | 93.39 | 84.36 |
| DINOv2 + SNGP | **97.49** | **0.01** | **0.10** | 82.44 | **99.34** | **94.85** |

Mean over 5 seeds; full results with std in the report.

## Repo layout
- `src/` — models (SNGP, GP layers, ResNet/WRN), training, evaluation, OOD evaluation
- `experiments/` — one script per experiment (SNGP, SupCon+SNGP, augmented, deep ensemble, DINOv2, OOD eval)
- `configs/` — YAML configs for each experiment
- `slurm_scripts/` — cluster job scripts
- `notebooks/` — UMAP/t-SNE visualisations, GMM analysis
