import os
from glob import glob
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)

from stereo_depth import calculate_depth_depthpro
import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image
from pathlib import Path

# ------------------------------------------------------------------
# Colormaps
# ------------------------------------------------------------------
tab20 = cm.get_cmap("tab20")
pastel = cm.get_cmap("Pastel2", lut=50)


def get_color(z):
    return [int(255 * val) for val in tab20(z)[:3]]


def get_pastel(z):
    return [int(255 * val) for val in pastel(z % pastel.N)[:3]]


# ------------------------------------------------------------------
# SAM3 prompts: (class_name, text_prompt)
# ------------------------------------------------------------------
CLASS_PROMPTS = [
    ("car", "car"),
    ("pedestrian", "person walking"),
    ("cyclist", "person on a bicycle"),
]


class bbox_3d:
    """
    Uses SAM3 (Hugging Face transformers) + stereo depth to get 3D boxes.

    Key points:
      * SAM3 gives instance masks + 2D boxes for text prompts.
      * For each mask we:
          - grab all 3D points from the disparity->XYZ map
          - compute an axis-aligned 3D box
          - project the 8 corners back to the left image.
      * No KNN, no Open3D, no clustering stage.
    """

    def __init__(
        self,
        sam3_model,
        sam3_processor,
        class_prompts,
        seq_name: str,
        device: str = "cuda",
        camera_matrix=None,
        projection_matrix=None,
        Q=None,
        num_disparities=None,
    ):

        self.sam3_model = sam3_model
        self.sam3_processor = sam3_processor
        self.class_prompts = class_prompts
        self.seq_name = seq_name
        self.device = device

        self.K = camera_matrix
        self.P_left = projection_matrix
        self.Q = Q
        self.num_disparities = num_disparities

        # cache stereo for whole sequence
        self._disp_maps = None
        self._depth_maps = None

        # map class_id -> human name
        self.class_id_to_name = {i: name for i, (name, _) in enumerate(class_prompts)}

    # --------------------------------------------------------------
    # 1) Disparity + depth (stereo, cached)
    # --------------------------------------------------------------
    def _ensure_stereo(self):
        # stereo path no longer used in the DepthPro pipeline, keep for backward compatibility
        if self._disp_maps is None or self._depth_maps is None:
            raise RuntimeError("Stereo disparity not set; use depth_map_override (DepthPro).")

    def depth_and_disp(self, frame_index: int = 0):
        """
        Returns disparity + depth for a given frame index.
        """
        self._ensure_stereo()
        depth_i = np.asarray(self._depth_maps[frame_index])
        disp_i = np.asarray(self._disp_maps[frame_index])

        depth_i = np.nan_to_num(depth_i, nan=0.0, posinf=0.0, neginf=0.0)
        return depth_i, disp_i

    # --------------------------------------------------------------
    # 1b) SAM3 segmentation → instances (masks + boxes)
    # --------------------------------------------------------------
    def segment_instances_sam3(
        self,
        left_image_rgb: np.ndarray,
        score_thresh: float = 0.3,
    ):
        """
        left_image_rgb: HxWx3 RGB numpy

        Returns list of instances, each:
            {
                "mask": tensor (H,W),
                "box":  np.array(4) [x1,y1,x2,y2] (float),
                "score": float,
                "class_name": str,
                "class_id": int
            }
        """
        H, W, _ = left_image_rgb.shape
        image_pil = Image.fromarray(left_image_rgb)

        instances = []

        for class_id, (class_name, prompt_text) in enumerate(self.class_prompts):
            inputs = self.sam3_processor(
                images=image_pil,
                text=prompt_text,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam3_model(**inputs)

            # HF post-processing: returns a list of dicts
            results = self.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=score_thresh,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks = results["masks"]   # (N, H, W)
            boxes = results["boxes"]   # (N, 4)
            scores = results["scores"]

            if masks is None or masks.shape[0] == 0:
                continue

            for i in range(masks.shape[0]):
                score_i = float(scores[i])
                if score_i < score_thresh:
                    continue

                inst = {
                    "mask": masks[i],                             # tensor (H,W)
                    "box": boxes[i].detach().cpu().numpy(),       # (4,)
                    "score": score_i,
                    "class_name": class_name,
                    "class_id": class_id,
                }
                instances.append(inst)

        print(f"[segment_instances_sam3] Found {len(instances)} instances.")
        return instances

    # --------------------------------------------------------------
    # 2) SAM3 "detections" + stereo depth → bboxes [.. u v z]
    #     (and corresponding masks)
    # --------------------------------------------------------------
    def get_depth_detections(
        self,
        left_image,
        right_image=None,  # kept for API compatibility, unused when depth_map_override is provided
        method="median",
        draw_boxes=True,  # ignored, kept for API compatibility
        score_thresh: float = 0.1,
        max_depth: float = 30.0,
        frame_index: int = 0,
        depth_map_override=None,
        disp_map_override=None,
    ):
        """
        Returns:
            left_image (unchanged)
            disp_map
            depth_map
            bboxes: [x1, y1, x2, y2, conf, cls, u, v, z]
            masks:  (N, H, W) bool, aligned with bboxes
        """

        # --- stereo depth ---
        if depth_map_override is None:
            raise RuntimeError("depth_map_override is required (use DepthPro).")
        else:
            depth_map = np.asarray(depth_map_override)
            disp_map = (
                np.asarray(disp_map_override)
                if disp_map_override is not None
                else np.zeros_like(depth_map, dtype=np.float32)
            )

        # basic sanitizing and clipping
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        depth_map = np.clip(depth_map, 0.0, max_depth)
        H_d, W_d = depth_map.shape
        H_img, W_img = left_image.shape[:2]

        # make sure depth/disparity match image size
        if (H_d, W_d) != (H_img, W_img):
            depth_map = cv2.resize(
                depth_map,
                (W_img, H_img),
                interpolation=cv2.INTER_NEAREST,
            )
            disp_map = cv2.resize(
                disp_map,
                (W_img, H_img),
                interpolation=cv2.INTER_NEAREST,
            )

        # --- SAM3 segmentation instead of YOLO ---
        instances = self.segment_instances_sam3(left_image, score_thresh=score_thresh)

        if len(instances) == 0:
            print("[get_depth_detections] No SAM3 instances, returning empty bboxes.")
            return (
                left_image,
                disp_map,
                depth_map,
                np.zeros((0, 9), dtype=float),
                np.zeros((0, H_img, W_img), dtype=bool),
            )

        kept_base = []
        kept_uvz = []
        masks_out = []

        for inst in instances:
            x1, y1, x2, y2 = inst["box"]
            conf = inst["score"]
            cls_id = float(inst["class_id"])

            mask_np = inst["mask"].detach().cpu().numpy().astype(bool)
            if mask_np.shape != (H_img, W_img):
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (W_img, H_img),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            box_depth = depth_map[mask_np].flatten()
            box_depth = box_depth[(box_depth > 0.0) & np.isfinite(box_depth)]

            if len(box_depth) == 0:
                z = 0.0
            else:
                if method == "median":
                    z = float(np.median(box_depth))
                else:
                    z = float(np.mean(box_depth))

            # -----------------------------
            # filter by depth reliability
            # -----------------------------
            if (z <= 0.0) or (z > max_depth):
                # very far or invalid depth
                continue

            # center of the 2D box in pixel coords
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)

            kept_base.append([x1, y1, x2, y2, conf, cls_id])
            kept_uvz.append([u, v, z])
            masks_out.append(mask_np)

        if len(kept_base) == 0:
            print("[get_depth_detections] All SAM3 instances filtered out.")
            return (
                left_image,
                disp_map,
                depth_map,
                np.zeros((0, 9), dtype=float),
                np.zeros((0, H_img, W_img), dtype=bool),
            )

        base_bboxes = np.array(kept_base, dtype=float)
        uvz = np.array(kept_uvz, dtype=float)
        bboxes = np.concatenate([base_bboxes, uvz], axis=1)
        masks_stack = np.stack(masks_out, axis=0)

        return left_image, disp_map, depth_map, bboxes, masks_stack

    # --------------------------------------------------------------
    # 3) disparity -> XYZ
    # --------------------------------------------------------------
    def disparity_to_xyz(self, disp_map):
        if self.Q is None:
            raise ValueError("Q matrix is not set. Pass it into bbox_3d(Q=...).")
        xyz = cv2.reprojectImageTo3D(disp_map.copy(), self.Q)
        return xyz

    # --------------------------------------------------------------
    # 3b) depth (metric) -> XYZ using intrinsics
    # --------------------------------------------------------------
    def depth_to_xyz(self, depth_map):
        """
        Convert metric depth map (m) to XYZ in camera frame.
        """
        if self.K is None:
            raise ValueError(
                "camera_matrix K is not set. Pass it into bbox_3d(..., camera_matrix=K_left)."
            )

        H, W = depth_map.shape
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        Z = depth_map.astype(np.float32)
        X = (u_coords - cx) * Z / fx
        Y = (v_coords - cy) * Z / fy
        xyz = np.stack((X, Y, Z), axis=-1)
        return xyz

    # --------------------------------------------------------------
    # 4) get 3D clusters directly from masks (NO KNN)
    # --------------------------------------------------------------
    def get_3d_clusters(self, xyz, masks, min_points=30, max_depth_clip=30.0):
        """
        masks: (N, H, W) bool
        xyz:   (H, W, 3) array from reprojectImageTo3D
        Returns:
            clusters_xyz:  list of (M_i,3) arrays
            valid_indices: np.ndarray of indices into masks / bboxes
        """
        H, W, _ = xyz.shape
        clusters_xyz = []
        valid_indices = []

        for idx, mask in enumerate(masks):
            if mask.shape != (H, W):
                m_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                m_resized = mask

            pts = xyz[m_resized]  # (M,3)
            if pts.size == 0:
                continue

            z = pts[:, 2]
            valid = np.isfinite(z) & (z > 0.0) & (z < max_depth_clip)
            pts_valid = pts[valid]

            if pts_valid.shape[0] < min_points:
                continue

            clusters_xyz.append(pts_valid)
            valid_indices.append(idx)

        return clusters_xyz, np.array(valid_indices, dtype=int)

    # --------------------------------------------------------------
    # 5) XYZ -> (u,v) using projection matrix
    # --------------------------------------------------------------
    def get_left_uv_from_xyz(self, xyz):
        """
        xyz: (N,3) array
        """
        if self.P_left is None:
            raise ValueError(
                "P_left (projection_matrix) is not set. Pass it into bbox_3d(..., projection_matrix=P_left)."
            )

        xyzw = np.hstack((xyz, np.ones((len(xyz), 1))))
        uvw = self.P_left @ xyzw.T

        uvw[:2, :] /= uvw[2, :]
        image_uv = np.round(uvw[:2, :]).astype(int)

        return image_uv

    # --------------------------------------------------------------
    # 6) Draw clusters as points on image
    # --------------------------------------------------------------
    def draw_clusters_on_image(self, clusters, image):
        """draws clusters on image"""
        for cluster_xyz in clusters:
            if len(cluster_xyz) == 0:
                continue

            cluster_uv = self.get_left_uv_from_xyz(cluster_xyz)

            color = get_color(int(np.random.uniform(0, 20)))

            for (u, v) in cluster_uv.T:
                if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
                    cv2.circle(image, (u, v), 1, color, -1)

        return image

    # --------------------------------------------------------------
    # 7) 3D bounding boxes from clusters
    # --------------------------------------------------------------
    def get_3d_bboxes(self, clusters, percentile_clip: float = 5.0):
        """
        clusters: list of (M_i,3) arrays
        Returns:
            box_points_uv_list: list of (2,8) arrays
            dims_list: list of (w, h, l) for each box (meters)
        """
        box_points_uv_list = []
        dims_list = []

        for cluster_xyz in clusters:
            if len(cluster_xyz) == 0:
                continue

            # robust trimming to drop outlier points that blow up the box
            if percentile_clip > 0.0:
                lower = np.percentile(cluster_xyz, percentile_clip, axis=0)
                upper = np.percentile(cluster_xyz, 100.0 - percentile_clip, axis=0)
                mask = np.all((cluster_xyz >= lower) & (cluster_xyz <= upper), axis=1)
                cluster_use = cluster_xyz[mask] if mask.any() else cluster_xyz
            else:
                cluster_use = cluster_xyz

            (x_min, y_min, z_min) = cluster_use.min(axis=0)
            (x_max, y_max, z_max) = cluster_use.max(axis=0)

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
            dims_list.append(
                (
                    float(x_max - x_min),  # width (X span)
                    float(y_max - y_min),  # height (Y span)
                    float(z_max - z_min),  # length/depth (Z span)
                )
            )

        return box_points_uv_list, dims_list

    # --------------------------------------------------------------
    # 8) Draw 3D boxes on image (with class + confidence)
    # --------------------------------------------------------------
    def draw_3d_boxes(self, image, camera_box_points, bboxes_for_boxes=None):
        """
        camera_box_points: list of (2,8) arrays
        bboxes_for_boxes:  (M, 9) array [x1, y1, x2, y2, conf, cls, u, v, z]
                           assumed to correspond to camera_box_points order
        """
        for i, box_pts in enumerate(camera_box_points):

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

            # optional label
            if bboxes_for_boxes is not None and i < len(bboxes_for_boxes):
                _, _, _, _, conf_i, cls_i, _, _, _ = bboxes_for_boxes[i]
                cls_i = int(cls_i)
                cls_name = self.class_id_to_name.get(cls_i, str(cls_i))
                label = f"{cls_name} {conf_i:.2f}"

                u_min = min(p[0] for p in pts)
                v_min = min(p[1] for p in pts)
                text_org = (int(u_min), int(max(0, v_min - 10)))

                cv2.putText(
                    image,
                    label,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return image


# ------------------------------------------------------------------
# Utility: path
# ------------------------------------------------------------------
def get_path_images(seq_name: str):
    ROOT = Path(__file__).resolve().parent.parent
    print(ROOT)
    data_root = ROOT / "Classification" / "34759_final_project_rect"

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

    # camera params (same as before)
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

    Tx_left = P_rect_02[0, 3] / fx
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

    NUM_DISPARITIES = None

    # load images
    seq_name = "seq_02"
    left_imgs, right_imgs = get_path_images(seq_name)

    # process full sequence
    start_index = 0
    end_index = len(left_imgs)

    # load SAM3 from Hugging Face
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[main] Using device:", device)

    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

    # DepthPro metric depth for the frames we will process
    depthpro_maps = calculate_depth_depthpro(seq_name, max_frames=end_index)

    # output folder
    ROOT = Path(__file__).resolve().parent.parent
    out_dir = ROOT / "outputs_3d_bbox" / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = bbox_3d(
        sam3_model=sam3_model,
        sam3_processor=sam3_processor,
        class_prompts=CLASS_PROMPTS,
        seq_name=seq_name,
        device=device,
        camera_matrix=K_left,
        projection_matrix=P_left,
        Q=Q,
        num_disparities=NUM_DISPARITIES,
    )

    for index in range(start_index, end_index):
        print(f"\n=== Frame {index} ===")

        left_image = cv2.cvtColor(cv2.imread(str(left_imgs[index])), cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(cv2.imread(str(right_imgs[index])), cv2.COLOR_BGR2RGB)
        depth_map = depthpro_maps[index - start_index]

        (
            left_image_out,
            left_disparity,
            depth_map,
            bboxes,
            masks,
        ) = bbox.get_depth_detections(
            left_image,
            right_image,
            method="median",
            draw_boxes=False,  # ignored
            score_thresh=0.3,
            max_depth=30.0,
            frame_index=index,
            depth_map_override=depth_map,
            disp_map_override=None,
        )

        if bboxes.shape[0] == 0:
            print("No detections after filtering.")
            plt.imshow(left_image_out)
            plt.axis("off")
            plt.show()
            continue

        # depth -> XYZ (metric)
        xyz = bbox.depth_to_xyz(depth_map)

        object_clusters_xyz, valid_indices = bbox.get_3d_clusters(
            xyz, masks, max_depth_clip=30.0
        )
        if len(object_clusters_xyz) == 0:
            print("No valid 3D clusters from masks.")
            continue

        bboxes_valid = bboxes[valid_indices]
        box_points_uv, box_dims = bbox.get_3d_bboxes(
            object_clusters_xyz, percentile_clip=5.0
        )

        # No per-class size caps / volume caps; keep all surviving masks

        left_with_3d = bbox.draw_3d_boxes(
            left_image_out.copy(), box_points_uv, bboxes_for_boxes=bboxes_valid
        )

        new_image = np.zeros_like(left_image, dtype=np.uint8)
        new_image = bbox.draw_clusters_on_image(object_clusters_xyz, new_image)

        stacked = np.vstack((left_with_3d, new_image))
        out_path = out_dir / f"{index:06d}.png"
        plt.imsave(out_path, stacked)
