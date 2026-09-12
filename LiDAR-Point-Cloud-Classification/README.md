# 3D Point Cloud Classification: Comparing Unsupervised & Supervised Methods with Engineered Geometric Features

A comparative study of machine learning approaches for semantic segmentation of terrestrial LiDAR point clouds in urban environments.

## Overview

This project implements and evaluates multiple classification strategies for 3D point cloud data:

- **Unsupervised clustering**: K-Means, BIRCH
- **Supervised classification**: Random Forest, Support Vector Machine
- **Feature engineering**: PCA-based features + inclination angle for improved separability

The dataset consists of terrestrial LiDAR scans of an urban area with four target classes: ground, vegetation, buildings with gable roofs, and buildings with flat roofs.

## Key Results

| Method | Balanced Accuracy | F1-Score (macro) | Notes |
|--------|------------------|------------------|-------|
| K-Means (unsupervised) | — | — | Poor separability of building types |
| BIRCH (unsupervised) | — | — | Better than K-Means but still limited |
| Random Forest (Z only) | 0.82 | 0.82 | Baseline supervised |
| SVM (Z + intensity) | 0.77 | 0.75 | Lower than RF |
| **RF + PCA + inclination angle** | **0.92** | **0.92** | **Best performance** |

**Main finding**: Engineering geometric features (local PCA components + surface inclination angle) improved balanced accuracy from 0.82 to 0.92 — a 12% relative improvement.

## Methodology

### Data

- Source: Terrestrial LiDAR scan of an urban campus area
- Point cloud size: ~200,000 points after preprocessing
- Classes: Ground (label 3), Vegetation/Tree (label 2), Building Gable Roof (label 1), Building Flat Roof (label 4)
- Training samples: Manually segmented using CloudCompare

### Preprocessing

1. Outlier removal via statistical filtering
2. Normal vector estimation (KDTree, radius=0.1, max_nn=30)
3. Min-Max normalization of features

### Feature Engineering (Key Contribution)

Beyond the baseline Z-coordinate feature, this project introduces:

1. **PCA-based features**: First 3 principal components of local neighborhoods
2. **Inclination angle**: `arctan2(Z, sqrt(X² + Y²))` — captures surface orientation

These features significantly improve discrimination between flat roofs, gable roofs, and vegetation.

### Classification

**Unsupervised:**
- K-Means (n_clusters=4, max_iter=800)
- BIRCH (threshold=0.1, n_clusters=4)

**Supervised:**
- Random Forest (n_estimators=800, max_depth=3)
- SVM (RBF kernel, default parameters)

Class imbalance handled via SMOTE oversampling.

## Installation

```bash
git clone https://github.com/yourusername/lidar-point-cloud-classification.git
cd lidar-point-cloud-classification
pip install -r requirements.txt


Usage
bash
# Run unsupervised classification
python src/unsupervised_classification.py

# Run supervised classification
python src/supervised_classification.py

# Run feature engineering pipeline
python src/feature_engineering.py
Dependencies
Python 3.9+

laspy

open3d

scikit-learn

imbalanced-learn

numpy

matplotlib

See requirements.txt for exact versions.