
import time
import numpy as np
import open3d as o3d

from nuscenes.nuscenes import NuScenes
from sklearn.cluster import DBSCAN




DATASET_PATH = "/Users/islamelsayed/Documents/Python/Project"




RANSAC_DISTANCE_THRESHOLD = 0.15



RANSAC_N = 3

RANSAC_ITERATIONS = 3000



DBSCAN_EPS = 0.5



DBSCAN_MIN_SAMPLES = 10



MIN_CLUSTER_POINTS = 10



NUMBER_OF_EVALUATION_SCANS = 10




SHOW_VISUALIZATIONS = True


print("Loading nuScenes Mini dataset...")

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATASET_PATH,
    verbose=True
)

print("\nDataset loaded successfully.")

print("Number of scenes:", len(nusc.scene))
print("Number of samples:", len(nusc.sample))



def load_lidar_scan(sample):

   lidar_token = sample["data"]["LIDAR_TOP"]

   lidar_path = nusc.get_sample_data_path(lidar_token)

 

    raw_points = np.fromfile(
        lidar_path,
        dtype=np.float32
    ).reshape(-1, 5)


    xyz = raw_points[:, :3]


    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)

    return xyz, pcd


def remove_ground(pcd):
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERATIONS
    )



    ground = pcd.select_by_index(inliers)

  

    obstacles = pcd.select_by_index(
        inliers,
        invert=True
    )

    return ground, obstacles, plane_model



def cluster_obstacles(obstacles):
   

    obstacle_points = np.asarray(obstacles.points)


    if len(obstacle_points) == 0:
        return obstacle_points, np.array([], dtype=int)


    dbscan = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES
    )

    labels = dbscan.fit_predict(obstacle_points)

    return obstacle_points, labels


def detect_clusters(obstacle_points, labels):
  

    detections = []

    if len(labels) == 0:
        return detections



    cluster_labels = sorted(set(labels) - {-1})

    for label in cluster_labels:

        

        cluster_points = obstacle_points[labels == label]

       

        if len(cluster_points) < MIN_CLUSTER_POINTS:
            continue

        cluster_pcd = o3d.geometry.PointCloud()

        cluster_pcd.points = o3d.utility.Vector3dVector(
            cluster_points
        )


        bbox = cluster_pcd.get_axis_aligned_bounding_box()


        bbox.color = (1.0, 0.0, 0.0)


        center = bbox.get_center()

       

        dimensions = bbox.get_extent()

        distance = np.linalg.norm(center)

        detection = {
            "label": int(label),
            "number_of_points": len(cluster_points),
            "center": center,
            "dimensions": dimensions,
            "distance": distance,
            "bounding_box": bbox,
            "point_cloud": cluster_pcd
        }

        detections.append(detection)

    return detections


def visualize_results(pcd, ground, obstacles, detections):
    


    print("\nDisplaying original LiDAR point cloud...")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="1 - Original LiDAR Point Cloud",
        width=1280,
        height=720
    )

  

    ground.paint_uniform_color([0.5, 0.5, 0.5])

    obstacles.paint_uniform_color([1.0, 0.3, 0.0])

    print("Displaying ground removal...")

    o3d.visualization.draw_geometries(
        [ground, obstacles],
        window_name="2 - Ground Removal",
        width=1280,
        height=720
    )



    geometries = []

    for detection in detections:

        geometries.append(
            detection["point_cloud"]
        )

        geometries.append(
            detection["bounding_box"]
        )

    if geometries:

        print("Displaying detected clusters and bounding boxes...")

        o3d.visualization.draw_geometries(
            geometries,
            window_name="3 - 3D Obstacle Detection",
            width=1280,
            height=720
        )

    else:
        print("No clusters available for visualization.")


def process_scan(sample, visualize=False):

    start_time = time.perf_counter()

    xyz, pcd = load_lidar_scan(sample)

    total_points = len(xyz)

    ground, obstacles, plane_model = remove_ground(pcd)

    ground_points = len(ground.points)

    obstacle_points_count = len(obstacles.points)


    obstacle_points, labels = cluster_obstacles(obstacles)


    detections = detect_clusters(
        obstacle_points,
        labels
    )

    if len(labels) > 0:
        noise_points = int(np.sum(labels == -1))
    else:
        noise_points = 0


    elapsed_time = time.perf_counter() - start_time

    print("\n" + "=" * 55)
    print("LIDAR OBSTACLE DETECTION RESULTS")
    print("=" * 55)

    print("Total LiDAR points:", total_points)

    print("Ground points:", ground_points)

    print("Non-ground points:", obstacle_points_count)

    print("Detected clusters:", len(detections))

    print("DBSCAN noise points:", noise_points)

    print("Estimated plane:", plane_model)

    print(f"Processing time: {elapsed_time:.3f} seconds")

    print("\nIndividual detections:")

    for detection in detections:

        print(
            f"Cluster {detection['label']}: "
            f"{detection['number_of_points']} points, "
            f"distance = {detection['distance']:.2f} m, "
            f"center = {np.round(detection['center'], 2)}, "
            f"dimensions = {np.round(detection['dimensions'], 2)}"
        )


    if visualize:

        visualize_results(
            pcd,
            ground,
            obstacles,
            detections
        )


    return {
        "total_points": total_points,
        "ground_points": ground_points,
        "obstacle_points": obstacle_points_count,
        "clusters": len(detections),
        "noise_points": noise_points,
        "time_seconds": elapsed_time,
        "detections": detections
    }


def evaluate_algorithm():

    number_of_scans = min(
        NUMBER_OF_EVALUATION_SCANS,
        len(nusc.sample)
    )

    results = []

    print("\n\nSTARTING EVALUATION")
    print("Number of scans:", number_of_scans)

    for i, sample in enumerate(nusc.sample[:number_of_scans]):

        print(f"\nProcessing scan {i + 1}/{number_of_scans}")

        show_this_scan = (
            SHOW_VISUALIZATIONS and i == 0
        )

        result = process_scan(
            sample,
            visualize=show_this_scan
        )

        results.append(result)


    print("\n\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    print(
        "Average total points:",
        f"{np.mean([r['total_points'] for r in results]):.1f}"
    )

    print(
        "Average ground points:",
        f"{np.mean([r['ground_points'] for r in results]):.1f}"
    )

    print(
        "Average non-ground points:",
        f"{np.mean([r['obstacle_points'] for r in results]):.1f}"
    )

    print(
        "Average detected clusters:",
        f"{np.mean([r['clusters'] for r in results]):.2f}"
    )

    print(
        "Average noise points:",
        f"{np.mean([r['noise_points'] for r in results]):.1f}"
    )

    print(
        "Average processing time:",
        f"{np.mean([r['time_seconds'] for r in results]):.3f} seconds"
    )

    print("\nEvaluation completed.")

    return results


if __name__ == "__main__":

    evaluation_results = evaluate_algorithm()
