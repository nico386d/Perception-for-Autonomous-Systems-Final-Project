from pathlib import Path
import cv2
import numpy as np


def calculate_baseline() :
    P_rect_02 = np.array([
        [7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01],
        [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03],
    ])

    P_rect_03 = np.array([
        [7.070493e+02, 0.000000e+00, 6.040814e+02, -3.341081e+02],
        [0.000000e+00, 7.070493e+02, 1.805066e+02, 2.330660e+00],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.201153e-03],
    ])

    f = P_rect_02[0, 0]
    Tx2 = P_rect_02[0, 3]
    Tx3 = P_rect_03[0, 3]

    return (Tx2 - Tx3) / f


def get_path_images():

    ROOT = Path(__file__).resolve().parent.parent
    data_root = ROOT / "Classification" / "34759_final_project_rect"
    print(data_root)
    return data_root

def calculate_disparity(seq_name: str):
    """
    Compute disparity and depth maps for all frames in a sequence.

    Returns:
        disparities: list of np.ndarray, one per frame (float32, pixels)
        depths:      list of np.ndarray, one per frame (float32, meters)
    """

    f = 7.070493e+02

    B = calculate_baseline()

    data_root = get_path_images()

    # Left/right image folders (rectified)
    left_dir = data_root / seq_name / "image_02" / "data"
    right_dir = data_root / seq_name / "image_03" / "data"

    left_imgs = sorted(left_dir.glob("*.png"))
    right_imgs = sorted(right_dir.glob("*.png"))
    print(left_dir)


    block_size = 9
    num_disparities = 128

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    disparities = []
    depths = []

    epsilon = 1e-6

    for idx, (l_path, r_path) in enumerate(zip(left_imgs, right_imgs)):
        left = cv2.imread(str(l_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(r_path), cv2.IMREAD_GRAYSCALE)


        disp_raw = stereo.compute(left, right).astype(np.float32)
        disp = disp_raw / 16.0

        disp[disp <= 0.0] = np.nan

        depth = f * B / (disp + epsilon)

        disparities.append(disp)
        depths.append(depth)

    return disparities, depths
def visualize_depth(
    depth: np.ndarray,
    max_depth: float = 80.0,
    window_name: str = "depth",
    save_path: Path | None = None,
    show: bool = True,
):
    """
    Visualize a depth map (in meters) using a color map.

    Args:
        depth:      2D array of shape (H, W), depth in meters (may contain NaNs).
        max_depth:  Depth (m) at which to clip for visualization. Farther = same color.
        window_name: OpenCV window name.
        save_path:  Optional path to save the colored depth image (e.g. Path("depth_000.png")).
        show:       If True, calls cv2.imshow + waitKey(0).
    """
    # Copy so we don't modify original
    depth_vis = depth.copy().astype(np.float32)

    # Replace invalid values (NaN, inf, <=0) with max_depth
    invalid_mask = ~np.isfinite(depth_vis) | (depth_vis <= 0)
    depth_vis[invalid_mask] = max_depth

    # Clip to a reasonable range [0, max_depth]
    depth_vis = np.clip(depth_vis, 0.0, max_depth)

    # Normalize to [0, 255] for display
    # Option 1: nearer = brighter
    depth_norm = depth_vis / max_depth          # [0, 1], far = 1.0
    depth_norm = 1.0 - depth_norm               # invert: near = 1.0, far = 0.0
    depth_norm = (depth_norm * 255.0).astype(np.uint8)

    # Apply a colormap (e.g. JET)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # Show
    if show:
        cv2.imshow(window_name, depth_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), depth_color)

    return depth_color

if __name__ == "__main__":


  B = calculate_baseline()
  disp_maps, depth_maps = calculate_disparity("seq_01")
  print(f"Computed {len(disp_maps)} disparity/depth maps for seq_01")
  first_depth = depth_maps[0]
  depth_color = visualize_depth(
      first_depth,
      max_depth=80.0,
      window_name="seq_01 depth frame 0",
      save_path=Path("outputs/depth_seq01_frame000.png"),
  )

