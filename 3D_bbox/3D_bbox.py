import cv2
import numpy as np
from stereo_depth import calculate_disparity, calculate_baseline
import open3d as o3d



class bbox_3d:   

    def __init__(self, camera_matrix=None, projection_matrix=None, class_dims=None):

        self.K = camera_matrix 
        self.P = projection_matrix 
        self.dims = class_dims 

    def depth(self):

        B = calculate_baseline()
        disp_map, depth_map = calculate_disparity("seq_01")

        depth_np = depth_map[0]    

        # Convert to float32 (required by Open3D)
        depth_np = depth_np.astype(np.float32)

        # Replace NaNs with 0 (0 = invalid depth → ignored)
        depth_np = np.nan_to_num(depth_np, nan=0.0)

        # Wrap as Open3D Image
        depth_o3d = o3d.geometry.Image(depth_np)

        return depth_o3d

    def backproject_to_point_cloud(self):


        width, height = 1224, 370

        fx = 7.070493e+02
        fy = 7.070493e+02
        cx = 6.040814e+02
        cy = 1.805066e+02

        camera = o3d.camera.PinholeCameraIntrinsic()
        camera.set_intrinsics(width, height, fx, fy, cx, cy)

        scene_pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
            self.depth(),       
            camera,
            depth_scale=1.0,    # use 1.0 if meters
            depth_trunc=100.0   # max depth distance 
        )

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsics.set_intrinsics(width, height, fx, fy, cx, cy) 

        scene_pointcould = o3d.geometry.PointCloud.create_from_depth_image(
            self.depth(), camera_intrinsics)

        # Flip it, otherwise the pointcloud will be upside down
        scene_pointcould.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        return scene_pointcould 


if __name__ == "__main__":

    bbox = bbox_3d() 
    pcd = bbox.backproject_to_point_cloud() 
    o3d.visualization.draw_geometries([pcd])
    




