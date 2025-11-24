import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 10)
from bbox._3D_bbox import bbox_3d
import open3d as o3d
import torch
from ultralytics import YOLO
from pathlib import Path

def get_path_images(seq_name: str):
    ROOT = Path(__file__).resolve().parent.parent
    data_root = ROOT / "Perception-for-Autonomous-Systems-Final-Project" / "34759_final_project_rect"
    print(ROOT)
    left_dir = data_root / seq_name / "image_02" / "data"
    right_dir = data_root / seq_name / "image_03" / "data"

    left_imgs = sorted(left_dir.glob("*.png"))
    right_imgs = sorted(right_dir.glob("*.png"))

    print(f"Number of left images: {len(left_imgs)}")
    print(f"Number of right images: {len(right_imgs)}")

    return left_imgs, right_imgs, ROOT


def main():
    # in stereo_depth --> depth and disp method change seq
    left_imgs, right_imgs, root = get_path_images("seq_01")
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


    #index = 2

    #left_image = cv2.cvtColor(cv2.imread(str(left_imgs[index])), cv2.COLOR_BGR2RGB)
    #right_image = cv2.cvtColor(cv2.imread(str(right_imgs[index])), cv2.COLOR_BGR2RGB)



    # if nothing detected, just show depth
   

    result_video = []

    for i in range(len(left_imgs)):   
        left_image = cv2.cvtColor(cv2.imread(left_imgs[i]), cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(cv2.imread(right_imgs[i]), cv2.COLOR_BGR2RGB)

        index = i
        ## 1) detect objects and get depth measurements
        left_image, left_disparity, depth_map, bboxes = bbox.get_depth_detections(
            left_image,
            right_image,
            index,
            method="median",
            draw_boxes=False,
        )

        xyz = bbox.disparity_to_xyz(left_disparity)

        object_centers_xyz = bbox.get_xyz_centers(bboxes, xyz)

        object_clusters_xyz = bbox.get_3d_clusters(xyz, object_centers_xyz)

        box_points_uv = bbox.get_3d_bboxes(object_clusters_xyz)
    
        left_with_3d = bbox.draw_3d_boxes(left_image.copy(), box_points_uv)

        new_image = np.zeros_like(left_image, dtype=np.uint8)
        new_image = bbox.draw_clusters_on_image(object_clusters_xyz, new_image)
    
        # stack frames
        stacked = np.vstack((left_with_3d, new_image))

        # add to result video
        result_video.append(stacked)


    # get width and height for video frames
    h, w, _ = stacked.shape

    out = cv2.VideoWriter('boxed_stereo_stack_2011_09_26.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))

    # or use mp4
    # out = cv2.VideoWriter('boxed_pointcloud_stack_2011_10_03.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (w,h))

    
    for i in range(len(result_video)):
        out.write(cv2.cvtColor(result_video[i], cv2.COLOR_BGR2RGB))
    out.release()

    
if __name__ == "__main__":

    main()

   