# Results Analysis

## Baseline (Z-coordinate only)
- **Random Forest**: Balanced Accuracy = 0.82
- **SVM**: Balanced Accuracy = 0.77

### Confusion Matrix Insights
- Gable roofs were frequently misclassified as vegetation due to similar Z-variance.
- Flat roofs were misclassified as ground due to planar geometry.

## Enhanced Features (PCA + Inclination Angle)
- **Random Forest**: Balanced Accuracy = 0.92

### Why it Works
The inclination angle provides a clear separation between flat roofs (near 0°), gable roofs (30-45°), and vegetation (high variance). PCA components capture the local geometric complexity, which distinguishes tree canopies from building edges.

## Conclusion
Feature engineering is critical for LiDAR classification. Simple height (Z) features are insufficient for distinguishing complex urban objects. Geometric features like inclination angle and local PCA provide significant performance gains without requiring complex deep learning architectures.