import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)

from stereo_depth import calculate_disparity, calculate_baseline
import open3d as o3d
import torch
from ultralytics import YOLO
from pathlib import Path

# ------------------------------------------------------------------
# Colormaps
# ------------------------------------------------------------------
tab20 = cm.get_cmap("tab20")
pastel = cm.get_cmap("Pastel2", lut=50)

def get_color(z):
    return [int(255 * val) for val in tab20(z)[:3]]

def get_pastel(z):
    return [int(255 * val) for val in pastel(z)[:3]]


class bbox_3d:

    def __init__(
        self,
        model,
        camera_matrix=None,
        projection_matrix=None,
        Q=None,
        num_disparities=None,
    ):

        self.K = camera_matrix
        self.P_left = projection_matrix
        self.Q = Q
        self.num_disparities = num_disparities
        self.model = model

    # --------------------------------------------------------------
    # 1) Disparity + depth
    # --------------------------------------------------------------
    def depth_and_disp(self):
        _ = calculate_baseline()  
        disp_map, depth_map = calculate_disparity("seq_01")

        depth_0 = np.asarray(depth_map[0])
        disp_0 = np.asarray(disp_map[0])

        return depth_0, disp_0

    # --------------------------------------------------------------
    # 2) YOLO detections + depth, append (u, v, z) per box
    # --------------------------------------------------------------
    def get_depth_detections(
        self,
        left_image,
        right_image,
        method="median",
        draw_boxes=True,
    ):
        """
        Returns:
            left_image_with_boxes
            disp_map
            depth_map
            bboxes: [x1, y1, x2, y2, conf, cls, u, v, z]
        """

        depth_map, disp_map = self.depth_and_disp()

      
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

        results = self.model(left_image)
        detections = results[0]


        if draw_boxes:
            left_image = detections.plot()  

       
        boxes = detections.boxes
        if boxes is None or len(boxes) == 0:
            bboxes = np.zeros((0, 9), dtype=float) 
            return left_image, disp_map, depth_map, bboxes

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
        base_bboxes = np.concatenate([xyxy, conf, cls], axis=1)  

     
        N = base_bboxes.shape[0]
        uvz = np.zeros((N, 3), dtype=float)

        for i, (x1, y1, x2, y2, _, _) in enumerate(base_bboxes):
            x1i = int(max(0, np.floor(x1)))
            y1i = int(max(0, np.floor(y1)))
            x2i = int(min(depth_map.shape[1] - 1, np.ceil(x2)))
            y2i = int(min(depth_map.shape[0] - 1, np.ceil(y2)))

            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)

           
            box_depth = depth_map[y1i:y2i + 1, x1i:x2i + 1].flatten()
            box_depth = box_depth[box_depth > 0]  

            if len(box_depth) == 0:
                z = 0.0
            else:
                if method == "median":
                    z = np.median(box_depth)
                else:
                    z = np.mean(box_depth)

            uvz[i, :] = np.array([u, v, z])

        bboxes = np.concatenate([base_bboxes, uvz], axis=1)

        return left_image, disp_map, depth_map, bboxes

    # --------------------------------------------------------------
    # 3) disparity -> XYZ or depth to xyz
    # --------------------------------------------------------------
    def disparity_to_xyz(self, disp_map):
        if self.Q is None:
            raise ValueError("Q matrix is not set. Pass it into bbox_3d(Q=...).")
        xyz = cv2.reprojectImageTo3D(disp_map.copy(), self.Q)  
        return xyz
    
    def depth_to_pcd_o3d(depth_map, K):
        h, w = depth_map.shape

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )

        depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))

        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            intrinsic,
            depth_scale=1.0,     
            depth_trunc=80.0,   
            stride=1,
        )

        return pcd



    # --------------------------------------------------------------
    # 4) get object centers in XYZ from bboxes and xyz map
    # --------------------------------------------------------------
    @staticmethod
    def get_xyz_centers(bboxes, xyz):
        """
        bboxes: [x1, y1, x2, y2, conf, cls, u, v, z]
        xyz: (H, W, 3) array from reprojectImageTo3D
        Returns:
            object_centers_xyz: (N,3)
        """
        if bboxes.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)

        object_centers_uvz = bboxes[:, 6:] 

        object_centers_xyz = np.zeros_like(object_centers_uvz)

        for i, (u, v, z) in enumerate(object_centers_uvz):
            u_i = int(np.clip(round(u), 0, xyz.shape[1] - 1))
            v_i = int(np.clip(round(v), 0, xyz.shape[0] - 1))
            object_centers_xyz[i, :] = xyz[v_i, u_i, :] 

        return object_centers_xyz

    # --------------------------------------------------------------
    # 5) RANSAC plane removal
    # --------------------------------------------------------------
    @staticmethod
    def run_ransac(pcd, n_iters):
        """Run RANSAC on point cloud multiple times to remove large planes."""
        outlier_cloud = o3d.geometry.PointCloud()
        outlier_cloud.points = pcd.points

        for _ in range(n_iters):
            num_pts = np.asarray(outlier_cloud.points).shape[0]
            if num_pts < 3:
                print(f"[RANSAC] Skipping, only {num_pts} points left.")
                break

            plane_model, inliers = outlier_cloud.segment_plane(
                distance_threshold=0.25,
                ransac_n=3,
                num_iterations=250,
            )

            outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

        return outlier_cloud

    # --------------------------------------------------------------
    # 6) Seeded KNN clustering
    # --------------------------------------------------------------
    @staticmethod
    def get_seeded_knn_clusters(pcd, pcd_tree, centroids, eps=2.0, max_points=500):
        """
        centroids: (N,3) array of desired centers in XYZ
        """
        object_clusters = []

        for xyz_center in centroids:
            k, idx, dist = pcd_tree.search_knn_vector_3d(xyz_center, max_points)

            if k == 0:
                continue

            idx = np.asarray(idx)
            dist = np.asarray(dist)

            # keep points within eps
            idx_prune = idx[dist <= eps]
            if len(idx_prune) == 0:
                continue

            cluster = pcd.select_by_index(idx_prune)
            object_clusters.append(cluster)

        return object_clusters

    # --------------------------------------------------------------
    # 7) full 3D cluster pipeline from XYZ + centers
    # --------------------------------------------------------------
    def get_3d_clusters(self, xyz, object_centers_xyz):

        # optional cropping if you want to mimic NUM_DISPARITIES slicing
        if self.num_disparities is not None:
            xyz_use = xyz[:, self.num_disparities:, :]
        else:
            xyz_use = xyz

        #Filter invalid points
        z = xyz_use[:, :, 2]
        mask = np.isfinite(z) & (z > 0)

        points = xyz_use[mask] 

        if points.shape[0] < 3:
            print(f"[get_3d_clusters] Not enough valid 3D points: {points.shape[0]}")
            return []  

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        downpcd = pcd.voxel_down_sample(voxel_size=0.15)

        outlier_cloud = self.run_ransac(downpcd, n_iters=1)

        if np.asarray(outlier_cloud.points).shape[0] < 3:
            print("[get_3d_clusters] Outlier cloud is too small after RANSAC.")
            return []


        pcd_cloud_tree = o3d.geometry.KDTreeFlann(outlier_cloud)

        object_clusters = self.get_seeded_knn_clusters(
            outlier_cloud,
            pcd_cloud_tree,
            object_centers_xyz,
            eps=2.0,
            max_points=400,
        )

        return object_clusters

    # --------------------------------------------------------------
    # 8) XYZ -> (u,v) using projection matrix
    # --------------------------------------------------------------
    def get_left_uv_from_xyz(self, xyz):

        if self.P_left is None:
            raise ValueError(
                "P_left (projection_matrix) is not set. Pass it into bbox_3d(..., projection_matrix=P_left)."
            )

        xyzw = np.hstack((xyz, np.ones((len(xyz), 1))))  # 
       
        uvw = self.P_left @ xyzw.T

        uvw[:2, :] /= uvw[2, :]
        image_uv = np.round(uvw[:2, :]).astype(int)

        return image_uv  

    # --------------------------------------------------------------
    # 9) Draw clusters as points on image
    # --------------------------------------------------------------
    def draw_clusters_on_image(self, clusters, image):
        """draws clusters on image"""
        for cluster in clusters:
            cluster_xyz = np.asarray(cluster.points)

            if len(cluster_xyz) == 0:
                continue

            cluster_uv = self.get_left_uv_from_xyz(cluster_xyz)

            color = get_color(int(np.random.uniform(0, 20)))

            for (u, v) in cluster_uv.T:
                if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
                    cv2.circle(image, (u, v), 1, color, -1)

        return image

    # --------------------------------------------------------------
    # 10) 3D bounding boxes from clusters
    # --------------------------------------------------------------
    def get_3d_bboxes(self, clusters):
        """
        Returns:
            list of box_pts_uv arrays, each shape (2,8)
        """
        box_points_uv_list = []

        for cluster in clusters:
            box_pts_xyz = np.asarray(cluster.points)

            if len(box_pts_xyz) == 0:
                continue

            (x_min, y_min, z_min) = box_pts_xyz.min(axis=0)
            (x_max, y_max, z_max) = box_pts_xyz.max(axis=0)

            box_points = np.array(
                [
                    [x_max, y_max, z_max],
                    [x_max, y_max, z_min],
                    [x_max, y_min, z_max],
                    [x_max, y_min, z_min],
                    [x_min, y_max, z_max],
                    [x_min, y_max, z_min],
                    [x_min, y_min, z_max],
                    [x_min, y_min, z_min],
                ]
            )

            box_pts_uv = self.get_left_uv_from_xyz(box_points)
            box_points_uv_list.append(box_pts_uv)

        return box_points_uv_list

    # --------------------------------------------------------------
    # 11) Draw 3D boxes on image
    # --------------------------------------------------------------
    def draw_3d_boxes(self, image, camera_box_points):
        for i, box_pts in enumerate(camera_box_points):
            
            # box_pts: (2,8) -> transpose to ((u,v), ...)
            pts = [tuple(p) for p in box_pts.T]
            if len(pts) != 8:
                continue

            A, B, C, D, E, F, G, H = pts
            color = get_pastel(i)

        
            cv2.line(image, A, B, color, 2)
            cv2.line(image, B, D, color, 2)
            cv2.line(image, A, C, color, 2)
            cv2.line(image, D, C, color, 2)


            cv2.line(image, G, E, color, 2)
            cv2.line(image, H, F, color, 2)
            cv2.line(image, G, H, color, 2)
            cv2.line(image, E, F, color, 2)

            cv2.line(image, E, A, color, 2)
            cv2.line(image, G, C, color, 2)
            cv2.line(image, F, B, color, 2)
            cv2.line(image, H, D, color, 2)

        return image


# ------------------------------------------------------------------
# Utility: path
# ------------------------------------------------------------------
def get_path_images(seq_name: str):
    ROOT = Path(__file__).resolve().parent.parent
    data_root = ROOT / "34759_final_project_rect"

    left_dir = data_root / seq_name / "image_02" / "data"
    right_dir = data_root / seq_name / "image_03" / "data"

    left_imgs = sorted(left_dir.glob("*.png"))
    right_imgs = sorted(right_dir.glob("*.png"))

    print(f"Number of left images: {len(left_imgs)}")
    print(f"Number of right images: {len(right_imgs)}")

    return left_imgs, right_imgs


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":

    model = YOLO("yolov8n.pt")


    P_rect_02 = np.array([
        [7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01],
        [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]
    ], dtype=np.float32)

 
    P_rect_03 = np.array([
        [7.070493e+02, 0.000000e+00, 6.040814e+02, -3.341081e+02],
        [0.000000e+00, 7.070493e+02, 1.805066e+02,  2.330660e+00],
        [0.000000e+00, 0.000000e+00, 1.000000e+00,  3.201153e-03]
    ], dtype=np.float32)

    fx = P_rect_02[0, 0]
    fy = P_rect_02[1, 1]
    cx = P_rect_02[0, 2]
    cy = P_rect_02[1, 2]

    K_left = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    Tx_left  = P_rect_02[0, 3] / fx
    Tx_right = P_rect_03[0, 3] / fx
    Tx = Tx_right - Tx_left           
    baseline = abs(Tx)                

    print("fx, fy, cx, cy:", fx, fy, cx, cy)
    print("baseline [m]: ", baseline)

 
    Q = np.array([
        [1.0, 0.0, 0.0, -cx],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0,  fx],
        [0.0, 0.0, -1.0 / Tx, 0.0]     
    ], dtype=np.float32)

 
    P_left = np.array([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)

    NUM_DISPARITIES = None  # e.g. 64 if you want cropping, else None

    bbox = bbox_3d(
        model,
        camera_matrix=None,
        projection_matrix=P_left,
        Q=Q,
        num_disparities=NUM_DISPARITIES,
    )

    left_imgs, right_imgs = get_path_images("seq_01")
    index = 2

    left_image = cv2.cvtColor(cv2.imread(str(left_imgs[index])), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(str(right_imgs[index])), cv2.COLOR_BGR2RGB)

    left_image, left_disparity, depth_map, bboxes = bbox.get_depth_detections(
        left_image,
        right_image,
        method="median",
        draw_boxes=False,
    )

    # if nothing detected, just show depth
    if bboxes.shape[0] == 0:
        print("No detections.")
        plt.imshow(left_image)
        plt.show()
        raise SystemExit

    # project disparity to 3D xyz / could use another way with o3d but 
    # will have to change ---get_xyz_centers---
    xyz = bbox.disparity_to_xyz(left_disparity)

    object_centers_xyz = bbox.get_xyz_centers(bboxes, xyz)

    object_clusters_xyz = bbox.get_3d_clusters(xyz, object_centers_xyz)

    box_points_uv = bbox.get_3d_bboxes(object_clusters_xyz)
  
    left_with_3d = bbox.draw_3d_boxes(left_image.copy(), box_points_uv)

    new_image = np.zeros_like(left_image, dtype=np.uint8)
    new_image = bbox.draw_clusters_on_image(object_clusters_xyz, new_image)


    stacked = np.vstack((left_with_3d, new_image))
    plt.figure(figsize=(25, 25))
    plt.imshow(stacked)
    plt.axis("off")
    plt.show()
