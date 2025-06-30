# Domain Adaptation for Binary Classification

This repository contains implementations of domain adaptation techniques for binary classification, focusing on both algebraic and deep learning approaches. The project is based on the TER (Travail d'Étude et de Recherche) subject provided by Guillaume Metzler, which explores unsupervised domain adaptation to improve model performance on target data distributions that differ from the source data.

## Project Overview

Domain adaptation addresses the challenge of training a model on a source dataset and adapting it to perform well on a target dataset with a different distribution. This repository includes two distinct implementations:

1. **Algebraic Approach**: A method based on kernel approximations and boosting, inspired by Gautheron et al. (2020).
2. **Deep Learning Approach**: A PyTorch-based implementation leveraging latent space representations.

Both approaches are evaluated and compared in the report `TER_Report.pdf`.

## Repository Structure
```
ImplementationBasedOnAlgebraicMethod
...

ImplementationBasedOnPytorch
├── README.md
├── data/
│   ├── raw/          # Données brutes
│   └── processed/    # Données transformées
├── notebooks/
│   └── DAN_pytorch.ipynb  # Notebook principal
├── src/
│   ├── models/       # Modules pour les modèles
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   └── classifier.py
│   ├── utils/        # Fonctions utilitaires
│   │   ├── __init__.py
│   │   ├── data_loading.py
│   │   ├── losses.py
│   │   ├── visualization.py
│   │   └── helpers.py
│   ├── config.py     # Configuration globale
│   └── train.py      # Script d'entraînement
└── outputs/
    └── figures/      # Visualisations
```



## Algebraic Approach

The algebraic approach is inspired by the work of Gautheron et al. (2020) and focuses on kernel-based methods and boosting techniques. Key features include:

- **Kernel Approximation**: Using random Fourier features to approximate Gaussian kernels.
- **Boosting**: Ensemble learning to improve model robustness.
- **Domain Divergence Minimization**: Techniques to align source and target distributions.

For more details, refer to the [algebraic_approach/README.md](algebraic_approach/README.md).

## PyTorch Approach

The PyTorch approach leverages deep learning for domain adaptation, focusing on latent space alignment. Key features include:

- **Neural Networks**: Architectures designed for feature extraction and classification.
- **Domain Adversarial Training**: Using adversarial losses to minimize domain divergence.
- **Latent Space Alignment**: Techniques such as Maximum Mean Discrepancy (MMD) or Domain Adversarial Neural Networks (DANN).

For more details, refer to the [pytorch_approach/README.md](pytorch_approach/README.md).

## Results

The performance of both approaches is documented in the report `TER_Report.pdf`, which includes:

- **Quantitative Metrics**: Accuracy, divergence measures, and generalization performance.
- **Qualitative Analysis**: Visualizations of feature spaces and domain alignment.
- **Comparison**: Strengths and weaknesses of each method.

## Contributors
- Hadrien BIGO-BALLAND
- Prince MEZUI ROTIMI