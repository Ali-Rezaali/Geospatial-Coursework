# Geospatial Coursework

Selected projects from the M.Sc. program at K. N. Toosi University of Technology. Each folder is a self-contained piece of work with its own source, documentation, and outputs.

## Contents

### `Advanced-GIS/Human Resilience Index`
Composite socioeconomic indicator built from three components: education, gender parity, and log-normalized income. Weighted into a single 0 to 1 score, then mapped globally at country resolution.

- Implementation in Python (GeoPandas) and ArcGIS Pro
- Same data, same Jenks classification, both pipelines produce matching maps
- Includes cartographic write-up on classification choices, weighting rationale, and the Gestalt decisions behind the final layout
- Key output: a global choropleth showing that income alone is a poor proxy for a society's ability to absorb a crisis

### `LiDAR-Point-Cloud-Classification`
Semantic classification of an airborne ALS point cloud into four classes: gable roofs, flat roofs, vegetation, and ground.

- Compares unsupervised (K-Means, BIRCH) and supervised (Random Forest, SVM) classifiers on the same dataset
- Feature engineering step adds inclination angle and PCA components, improving overall accuracy from 0.82 to 0.92
- Documents the specific failure modes of each approach: why clustering cannot separate a pitched roof from tree canopy, why Maximum Likelihood underperforms Minimum Distance on this terrain, and where each method's assumptions break

## Structure

Each project folder contains its own source scripts, notebooks, documentation, and outputs. The `.gitignore` at the repo root excludes Python cache files, Jupyter checkpoints, and any local data directories. Large input datasets are not included; each project's README specifies what's needed and where it can be obtained.

## Environment

Most projects are Python-based. Common dependencies across the repo:

- Core: numpy, pandas, scipy, matplotlib
- Geospatial: geopandas, rasterio, shapely, pyproj
- Image processing: opencv-python, scikit-image
- Machine learning: scikit-learn, xgboost, imbalanced-learn
- LiDAR: laspy, open3d

Individual project READMEs list exact versions where they matter for reproducibility. To run anything here, `pip install` the listed packages and follow the specific README in each folder.

## License

See `LICENSE`. Work here is coursework from an academic program and is shared for reference. If you use any of it, credit is appreciated.
