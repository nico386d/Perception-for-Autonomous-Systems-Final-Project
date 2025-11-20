import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 10)
from stereo_depth import calculate_depth_depthpro
import open3d as o3d
import torch 
from ultralytics import YOLO
from pathlib import Path



class bbox_3d:   

    def __init__(self, model, camera_matrix=None, projection_matrix=None,):

        self.K = camera_matrix 
        self.P = projection_matrix
        self.model = model

    def depth(self,seq_name,index):

        # B = calculate_baseline(), stereo only

        depth_map = calculate_depth_depthpro(seq_name, max_frames=1)

        depth_0 = depth_map[index]


        return depth_0, 
    
    
    def get_distances_pro_map(self, image, depth_map, bboxes, method='median', draw=True):
        # make sure depth_map is a 2D numpy array
        depth_map = np.asarray(depth_map)
        if depth_map.ndim == 3:
            depth_map = depth_map[..., 0]  # e.g. (H, W, 1) -> (H, W)

        H, W = depth_map.shape[:2]

        bboxes = np.asarray(bboxes)

        # extra 3 columns: x_center, y_center, depth
        bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3), dtype=float)
        bboxes_out[:, :bboxes.shape[1]] = bboxes

        for i, bbox in enumerate(bboxes):
            # YOLO format: [x1, y1, x2, y2, conf, cls]
            x1 = int(np.rint(bbox[0]))
            y1 = int(np.rint(bbox[1]))
            x2 = int(np.rint(bbox[2]))
            y2 = int(np.rint(bbox[3]))

            # ensure correct ordering
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            # clamp to image bounds
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))

            # center in image coords
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            stereo_depth = np.nan

            if x2 > x1 and y2 > y1:
                # depth_map is indexed [row (y), col (x)]
                depth_slice = depth_map[y1:y2, x1:x2]

                if depth_slice.size > 0:
                    if method == 'center':
                        cy = depth_slice.shape[0] // 2   # row index in slice
                        cx = depth_slice.shape[1] // 2   # col index in slice
                        stereo_depth = float(depth_slice[cy, cx])
                    else:
                        stereo_depth = float(np.nanmedian(depth_slice))

            if draw and not np.isnan(stereo_depth):
                cv2.putText(
                    image,
                    f"{stereo_depth:.2f} m",
                    (x_center, y_center),          # (x, y) in image coords
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            bboxes_out[i, -3:] = np.array([x_center, y_center, stereo_depth], dtype=float)

        return image, bboxes_out
    
    def get_depth_detections(self, left_image,seq_name, index, method='median', 
                         draw_boxes=True, draw_depth=True):
    
        depth_map = self.depth(seq_name,index)
        print("depth_map type:", type(depth_map), "shape:", getattr(depth_map, "shape", None))
        #filter depth map only i stereo
        #filtered_depth_map = cv2.medianBlur(depth_map, 5)

        results = model(left_image)

        detections = results[0]            

        # draw boxes on image
        if draw_boxes:
            left_image = detections.plot()


        boxes = detections.boxes  

        if boxes is None or len(boxes) == 0:
            
            return left_image, depth_map, np.zeros((0, 7))

  
        xyxy = boxes.xyxy  
        conf = boxes.conf.view(-1, 1)
        cls  = boxes.cls.view(-1, 1)
        bboxes = torch.cat([xyxy, conf, cls], dim=1)
        

        left_image, bboxes = self.get_distances_pro_map(left_image, depth_map, bboxes, method, draw_depth)

        # get_distances(left_image, depth_map, bboxes, method, draw_depth) - for stereo
                                 

        return left_image, depth_map, bboxes





def get_path_images(seq_name: str):

    ROOT = Path(__file__).resolve().parent.parent
    data_root = ROOT  / "34759_final_project_rect"

    left_dir = data_root / seq_name / "image_02" / "data"
    right_dir = data_root / seq_name / "image_03" / "data"

    left_imgs = sorted(left_dir.glob("*.png"))
    right_imgs = sorted(right_dir.glob("*.png"))

    print(f"Number of left images: {len(left_imgs)}")
    print(f"Number of right images: {len(right_imgs)}")

    return left_imgs, right_imgs



if __name__ == "__main__":
    # later on our model ofc
    model = YOLO("yolov8n.pt")
    bbox = bbox_3d(model)

    #get images
    seq_name = "seq_01"
    left_imgs, right_imgs = get_path_images(seq_name)
    index = 0
    left_image = cv2.cvtColor(cv2.imread(left_imgs[index]), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_imgs[index]), cv2.COLOR_BGR2RGB)

    #get depth_detections
    left_image, filtered_depth_map, bboxes = bbox.get_depth_detections(left_image, seq_name, index,method='median', 
                                                                                        draw_boxes=True, draw_depth=True)  
    
  
    nipy_spectral = plt.get_cmap('nipy_spectral')

    # 1) Clean NaNs/infs
    clean_depth = np.nan_to_num(filtered_depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Squeeze to 2D (H, W) if it's (H, W, 1) or similar
    clean_depth = np.squeeze(clean_depth)

    # 3) Normalize to [0, 1]
    vmin = clean_depth.min()
    vmax = clean_depth.max()
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = (clean_depth - vmin) / (vmax - vmin)

    # 4) Apply colormap → (H, W, 4), then keep only RGB
    depth_color = nipy_spectral(norm)          # (H, W, 4)
    depth_color = (255 * depth_color[..., :3]).astype(np.uint8)  # (H, W, 3)

    # 5) Make sure both images have same H, W
    if depth_color.shape[:2] != left_image.shape[:2]:
        depth_color = cv2.resize(depth_color, (left_image.shape[1], left_image.shape[0]))

    # 6) Stack vertically
    stacked = np.vstack((left_image, depth_color))

    plt.figure(figsize=(20, 10))
    plt.imshow(stacked)
    plt.axis("off")
    plt.show()

    

