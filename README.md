# Euro Coin Image Classification

From a multilayer perceptron to cost-sensitive transfer learning. Classifies euro coin images into the eight denominations (1c–2€) from a dataset of only 800 training images, then addresses a second objective: minimizing the *value* of misclassification, not just the error rate.

## Results

| Model | Total params | Trainable params | Test accuracy |
|---|---|---|---|
| MLP | ≈1.71M | ≈1.71M | ≈70% |
| CNN (from scratch) | ≈345K | ≈345K | 79.4% |
| ResNet18 — Phase 1 (frozen) | ≈11.2M | ≈4.1K | 75% |
| **ResNet18 — Phase 2 (fine-tuned)** | ≈11.2M | ≈11.2M | **96%** |

## Key findings

**Three architectures, each motivated by the last one's failure.** An MLP plateaus around 70% because it discards spatial structure entirely, treating each pixel independently. A from-scratch CNN reaches 79% with 5x fewer parameters than the MLP, thanks to weight sharing and locality. A pretrained ResNet18, fine-tuned in two phases (frozen backbone first, then full unfreezing at a low learning rate), jumps to 96% — confirming that on small datasets, reusing features learned on a large dataset beats training from scratch by a wide margin.

**Rotation augmentation is model-dependent.** Aggressive 180° rotations help the from-scratch MLP and CNN (coins have no canonical orientation), but hurt the pretrained ResNet, whose ImageNet features are orientation-dependent. Dialing rotation back to ±15° was the best setting for transfer learning (77.5% vs. 74% test accuracy at ±30°).

**Cost-sensitive classification comes free, when it's needed.** Misclassifying a 2€ coin as 1c is far more costly than confusing 10c with 20c. Since cross-entropy training already drives the softmax toward an estimate of the class posterior, a Bayes-optimal risk-minimizing decision rule can be swapped in at inference time with no retraining, using cost matrix C(i,j) = |vᵢ − vⱼ|. On the confident ResNet18 this rule makes no difference — there's no uncertainty left to exploit. On the weaker MLP it lowers mean misclassification cost from 4.47 to 3.48 cents, by steering both corrections and residual errors toward coins of nearer value.

## Dataset

- `ImageFolder` structure: 100 training images and 20 test images per class (800 train / 160 test total), RGB PNGs, 64×64 px.
- Balanced classes; images vary in noise, sharpness, lighting, background, and orientation (face up/down, rotated).
- Preprocessing: resize (32×32 for MLP/CNN, 224×224 for ResNet18), tensor conversion, ImageNet normalization. Training-only augmentation: random horizontal flips and random rotation.

## Model progression

1. **MLP** — funnel architecture (3072 → 512 → 256 → 8), batch norm + ReLU + dropout (p=0.3) per block, L2 weight decay (10⁻⁴). Learning-rate tuning (10⁻³ → 10⁻⁴) fixed early loss oscillation, but a plateau near 70% pointed to an architectural ceiling, not an optimization one.
2. **CNN (from scratch)** — three Conv(3×3)→BatchNorm→ReLU→MaxPool blocks (32→64→128 channels), compact classifier head. 86% train / 79% test accuracy; remaining errors cluster among visually similar small copper coins (1c, 2c, 5c).
3. **ResNet18 (transfer learning)** — Phase 1 trains only the new head with the backbone frozen (~4K trainable params); Phase 2 unfreezes everything at a lower learning rate (10⁻⁴). Converges to 99% train / 96% test accuracy by ~epoch 40.

## Limitations

- No validation split was used; hyperparameters were tuned by observing test performance directly.
- No early stopping, despite fine-tuning converging by epoch ~40 of 50.
- Results from a single run on 800 images carry non-negligible run-to-run variance.

## Repository contents

- `euro_coin_classification_report.tex` — full write-up (dataset, model progression, cost-sensitive decision rule, limitations).

## Author

Claudio Guarrasi — Department of Electrical, Computer and Biomedical Engineering, University of Pavia.

## AI-tool usage

AI assistance (Anthropic's Claude) was used for debugging (data loading, training instabilities), discussing ML concepts (batch size, regularization, augmentation, cost-sensitive theory), and drafting/structuring the report. All design decisions, experiments, and results are the author's own.

## References

- Cirillo, Solimando & Virgili, "A deep learning approach to classify country and value of modern coins," *Neural Computing and Applications* 36, 2024.
- Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.
- He, Zhang, Ren & Sun, "Deep residual learning for image recognition," CVPR 2016.
- Elkan, "The foundations of cost-sensitive learning," IJCAI 2001.
