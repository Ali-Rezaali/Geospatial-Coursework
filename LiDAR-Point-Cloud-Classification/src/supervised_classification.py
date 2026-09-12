"""
Supervised classification for point cloud data.

Implements Random Forest and SVM classifiers.

Author: Ali Rezaali
"""

import numpy as np
import laspy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import os


CLASS_NAMES = ["BuildingGableRoof", "Tree", "Ground", "BuildingFlatRoof"]


def load_and_label_data(main_file: str, sample_files: dict) -> tuple:
    """Load main point cloud and assign labels from sample files."""
    las = laspy.read(main_file)
    points = np.vstack([las.X, las.Y, las.Z]).transpose()
    
    labels = np.zeros((len(points), 1))
    point_set = set(map(tuple, points))
    
    for class_label, filepath in sample_files.items():
        sample_las = laspy.read(filepath)
        sample_points = np.vstack([sample_las.X, sample_las.Y, sample_las.Z]).transpose()
        sample_set = set(map(tuple, sample_points))
        
        for i, point in enumerate(points):
            if tuple(point) in sample_set:
                labels[i] = class_label
    
    return points, labels, las


def prepare_features(points: np.ndarray, use_intensity: bool = False, intensity: np.ndarray = None) -> np.ndarray:
    """Prepare feature matrix."""
    z = points[:, 2].reshape(-1, 1)
    
    if use_intensity and intensity is not None:
        features = np.hstack([z, intensity.reshape(-1, 1)])
    else:
        features = z
    
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)


def evaluate_and_export(clf, X_test, y_test, X_full, las, output_path, figure_path):
    """Evaluate classifier and export results."""
    y_pred = clf.predict(X_test)
    y_full = clf.predict(X_full)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot()
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    output = laspy.create(point_format=las.header.point_format)
    output.X = las.X
    output.Y = las.Y
    output.Z = las.Z
    output.classification = y_full.astype(np.uint8)
    output.write(output_path)


def main():
    sample_files = {
        1: 'data/sample_points/BuildingGableRoof_Train.las',
        2: 'data/sample_points/Vegetation_Train.las',
        3: 'data/sample_points/Ground_Train.las',
        4: 'data/sample_points/BuildingFlatRoof_Train.las',
    }
    
    main_file = 'data/sample_points/nonground_training.las'
    if not os.path.exists(main_file):
        print(f"Error: {main_file} not found. Please place your LAS files in data/sample_points/")
        return

    points, labels, las = load_and_label_data(main_file, sample_files)
    features = prepare_features(points)
    
    mask = labels[:, 0] != 0
    X = features[mask]
    y = labels[mask, 0]
    
    print(f"Class distribution: {Counter(y)}")
    
    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=0
    )
    
    os.makedirs('results/figures', exist_ok=True)
    
    # Train Random Forest
    rf_clf = RandomForestClassifier(n_estimators=800, max_depth=3, random_state=0)
    rf_clf.fit(X_train, y_train)
    evaluate_and_export(
        rf_clf, X_test, y_test, features, las,
        'results/supervised_rf.las',
        'results/figures/confusion_matrix_rf.png'
    )
    
    # Train SVM
    svm_clf = svm.SVC()
    svm_clf.fit(X_train, y_train)
    evaluate_and_export(
        svm_clf, X_test, y_test, features, las,
        'results/supervised_svm.las',
        'results/figures/confusion_matrix_svm.png'
    )


if __name__ == '__main__':
    main()