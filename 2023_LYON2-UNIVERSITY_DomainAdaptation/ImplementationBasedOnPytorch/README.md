├── README.md
├── data/
│   ├── MNIST_M.zip
|   └── open data MNIST
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
    ├── models/       # Modèles sauvegardés
    ├── figures/      # Visualisations
    └── results/      # Résultats d'évaluation