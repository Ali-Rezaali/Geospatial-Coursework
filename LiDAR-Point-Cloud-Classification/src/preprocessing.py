"""
Preprocessing utilities for LiDAR point cloud classification.

Author: Ali Rezaali
"""

import numpy as np
import laspy
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler


def load_las_file(filepath: str) -> tuple:
    """
    Load a LAS file and return coordinates and raw LAS object.
    
    Args:
        filepath: Path to .las file
        
    Returns:
        Tuple of (point_cloud Nx3 array, laspy object)
    """
    las = laspy.read(filepath)
    points = np.vstack([las.X, las.Y, las.Z]).transpose()
    return points, las


def compute_normals(points: np.ndarray, radius: float = 0.1, max_nn: int = 30) -> np.ndarray:
    """
    Compute normal vectors for each point using KDTree search.
    
    Args:
        points: Nx3 array of point coordinates
        radius: Search radius for neighborhood
        max_nn: Maximum number of neighbors
        
    Returns:
        Nx3 array of normal vectors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return np.asarray(pcd.normals)


def compute_inclination_angle(points: np.ndarray) -> np.ndarray:
    """
    Compute inclination angle (angle from horizontal plane).
    
    Args:
        points: Nx3 array of point coordinates
        
    Returns:
        Nx1 array of inclination angles in radians
    """
    xy_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    return np.arctan2(points[:, 2], xy_dist).reshape(-1, 1)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Apply Min-Max normalization to feature array.
    
    Args:
        features: NxM feature array
        
    Returns:
        Normalized feature array
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)