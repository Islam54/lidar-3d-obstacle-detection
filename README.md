# lidar-3d-obstacle-detection
LiDAR-based 3D obstacle detection using RANSAC ground removal and DBSCAN clustering on the nuScenes Mini dataset.


# LiDAR-Based 3D Obstacle Detection

A Python project implementing a basic 3D obstacle detection pipeline using LiDAR point clouds from the nuScenes Mini dataset (Boston Seaport, USA).

## Overview

The project uses classical point cloud processing techniques to identify potential obstacles in driving scenes.

The pipeline consists of:
- **Ground Removal:** RANSAC plane segmentation.
- **Clustering:** DBSCAN to group non-ground points.
- **3D Bounding Boxes:** Geometric representation of detected clusters.
- **Distance Estimation:** Calculation of the distance to each cluster centre.

## Technologies

- Python
- NumPy
- Open3D
- scikit-learn
- nuScenes devkit

## Dataset

This project uses the `v1.0-mini` version of the [nuScenes dataset](https://www.nuscenes.org/nuscenes).

The dataset is not included in this repository and must be downloaded separately.
