# Melanoma Detection using Metadata and Multimodal Deep Learning

This project explores machine learning and deep learning approaches for melanoma detection
using clinical metadata and dermoscopic images from the SLICE-3D / ISIC dataset.


## Overview / Approach

The repository contains two complementary approaches:

1. A **metadata-only machine learning pipeline** using extensive feature engineering and
   tree-based models.
2. A **multimodal deep learning pipeline** that fuses image features and metadata using
   convolutional neural networks and attention-based fusion.

Both approaches are designed with patient-level data leakage prevention and clinically
relevant evaluation metrics.


## Project Structure

melanoma-detection-slice3d/
│
├── metadata_only/ # Metadata-only ML approach
├── multimodal_image_metadata/ # Image + metadata DL approach
├── docs/ # Diagrams and documentation
├── README.md
└── requirements.txt



## Methods / Models

### Metadata-Only Approach
- Extensive feature engineering on clinical metadata
- Patient-level normalization and aggregation
- LightGBM, XGBoost, and CatBoost models
- Ensemble learning

### Multimodal Approach
- CNN backbones (ResNet, VGG, EfficientNet)
- Image preprocessing and normalization
- Metadata feature encoding using MLP
- Fusion strategies: concatenation, gated fusion, attention-based fusion


## Evaluation

Models are evaluated using:
- Partial AUC (pAUC) at high true positive rates
- Accuracy
- Precision
- Recall

pAUC is prioritized due to its relevance in clinical screening tasks.


## Notes

- Datasets are not included due to size and licensing restrictions
- Training was performed on external hardware due to resource constraints
- The codebase is modular and reproducible
