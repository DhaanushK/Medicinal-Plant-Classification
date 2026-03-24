# Medicinal Plant Classification using Vision Transformers (ViT)

## Overview
This project focuses on classifying medicinal plants using a deep learning approach based on Vision Transformers (ViT). The model is fine-tuned using a pre-trained ViT model to achieve high accuracy in image classification.

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- NumPy, Pandas
- Matplotlib, Seaborn

## Workflow
- Image preprocessing (resizing to 224x224, normalization, RGB conversion)
- Data augmentation (rotation, flipping, brightness adjustment)
- Train-test split (80:20 with stratification)
- Fine-tuning pre-trained ViT model (google/vit-base-patch16-224)
- Mixed precision training for performance optimization

## Features
- Transfer learning using Vision Transformers
- Data augmentation for overfitting control
- Exploratory Data Analysis (EDA) with visualizations
- Efficient training using GPU optimization

## Results
- Achieved strong classification performance using fine-tuned ViT
- Improved generalization through augmentation and preprocessing

## License
MIT License
