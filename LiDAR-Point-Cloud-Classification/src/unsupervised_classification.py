"""
Unsupervised clustering for point cloud classification.

Implements K-Means and BIRCH clustering algorithms.

Author: Ali Rezaali
"""

import numpy as np
import laspy
from sklearn.cluster import KMeans, Birch
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import os


def run_kmeans(features: np.ndarray, n_clusters: int = 4, max_iter: int = 800) -> np.ndarray:
    """Apply K-Means clustering."""
    clustering = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=0)
    clustering.fit(features)
    return clustering.labels_


def run_birch(features: np.ndarray, threshold: float = 0.1, n_clusters: int = 4) -> np.ndarray:
    """Apply BIRCH clustering."""
    clustering = Birch(threshold=threshold, n_clusters=n_clusters)
    clustering.fit(features)
    return clustering.labels_


def export_results(las: laspy.LasData, labels: np.ndarray, output_path: str) -> None:
    """Export clustered point cloud to LAS file."""
    output = laspy.create(point_format=las.header.point_format)
    output.X = las.X
    output.Y = las.Y
    output.Z = las.Z
    output.classification = labels.astype(np.uint8)
    output.write(output_path)


def main():
    input_file = 'data/sample_points/nonground_training.las'
    output_dir = 'results'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please place your LAS files in data/sample_points/")
        return

    las = laspy.read(input_file)
    points = np.vstack([las.X, las.Y, las.Z]).transpose()
    
    # Feature extraction (Z coordinate only for baseline)
    features = points[:, 2].reshape(-1, 1)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    # Run K-Means
    kmeans_labels = run_kmeans(features)
    print(f"K-Means cluster distribution: {Counter(kmeans_labels)}")
    os.makedirs(output_dir, exist_ok=True)
    export_results(las, kmeans_labels, os.path.join(output_dir, 'kmeans_result.las'))
    
    # Run BIRCH
    birch_labels = run_birch(features)
    print(f"BIRCH cluster distribution: {Counter(birch_labels)}")
    export_results(las, birch_labels, os.path.join(output_dir, 'birch_result.las'))


if __name__ == '__main__':
    main()