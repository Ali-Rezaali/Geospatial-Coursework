# Methodology

## 1. Data Acquisition and Preprocessing
The dataset consists of terrestrial LiDAR scans of an urban area. Raw point clouds were preprocessed to remove outliers and noise using statistical filtering in CloudCompare. Normal vectors were estimated using a KDTree search (radius=0.1m, max_nn=30).

## 2. Feature Extraction
For baseline models, only the Z-coordinate (elevation) was used as a feature.
For the enhanced model, three additional features were engineered:
- **Inclination Angle**: `arctan2(Z, sqrt(X² + Y²))`
- **PCA Components**: First three principal components of local neighborhoods, capturing local surface curvature and orientation.

## 3. Classification Models
- **Unsupervised**: K-Means (n=4), BIRCH (threshold=0.1, n=4)
- **Supervised**: Random Forest (n_estimators=800, max_depth=3), SVM (RBF kernel)
- **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training set to balance class distributions.

## 4. Evaluation Metrics
- Balanced Accuracy (BA)
- Fisher Discriminant Ratio (FDR) for class separability
- F1-Score (macro and weighted)