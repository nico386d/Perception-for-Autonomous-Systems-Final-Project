from pathlib import Path
import cv2
import numpy as np

from PIL import Image
import torch
from transformers import (
    pipeline,
    DepthProImageProcessorFast,
    DepthProForDepthEstimation,
)

import matplotlib.pyplot as plt  # NEW


def calculate_baseline():
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
    return data_root


def calculate_disparity(seq_name: str):
    """
    Stereo depth with OpenCV SGBM (your original implementation).
    Returns:
        disparities: list of (H, W) disparity maps
        depths:      list of (H, W) depth maps in meters
    """
    f = 7.070493e+02
    B = calculate_baseline()

    data_root = get_path_images()

    # Left/right image folders (rectified)
    left_dir = data_root / seq_name / "image_02" / "data"
    right_dir = data_root / seq_name / "image_03" / "data"

    left_imgs = sorted(left_dir.glob("*.png"))
    right_imgs = sorted(right_dir.glob("*.png"))
    print(f"Stereo left dir: {left_dir}")

    block_size = 12
    num_disparities = 128  # must be divisible by 16

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


def _to_numpy_depth(depth_tensor) -> np.ndarray:
    """
    Helper: convert a transformers depth tensor to (H, W) float32 numpy.
    Handles both torch.Tensor and already-numpy-ish inputs.
    """
    if isinstance(depth_tensor, torch.Tensor):
        depth = depth_tensor.detach().cpu().numpy()
    else:
        depth = np.array(depth_tensor)

    # squeeze leading dimension if needed (e.g. (1, H, W))
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]

    return depth.astype(np.float32)


def calculate_depth_zoedepth(
        seq_name: str,
        checkpoint: str = "Intel/zoedepth-nyu-kitti",
        max_frames: int | None = None,
):
    """
    Monocular metric depth using ZoeDepth

    Returns:
        depth_maps: list of (H, W) depth maps in meters.
    """
    data_root = get_path_images()
    left_dir = data_root / seq_name / "image_02" / "data"
    left_imgs = sorted(left_dir.glob("*.png"))
    print(f"ZoeDepth left dir: {left_dir}")

    depth_estimator = pipeline(
        task="depth-estimation",
        model=checkpoint,
    )

    depth_maps: list[np.ndarray] = []

    for idx, l_path in enumerate(left_imgs):
        image = Image.open(l_path).convert("RGB")
        outputs = depth_estimator(image)

        depth_tensor = outputs["predicted_depth"]
        depth = _to_numpy_depth(depth_tensor)
        depth_maps.append(depth)

        if max_frames is not None and (idx + 1) >= max_frames:
            break

    return depth_maps


def calculate_depth_depthanything_metric(
        seq_name: str,
        checkpoint: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        max_frames: int | None = None,
):
    """
    Monocular metric depth using Depth Anything V2 (metric, outdoor).

    Returns:
        depth_maps: list of (H, W) depth maps in meters.
    """
    data_root = get_path_images()
    left_dir = data_root / seq_name / "image_02" / "data"
    left_imgs = sorted(left_dir.glob("*.png"))
    print(f"DepthAnything V2 (metric) left dir: {left_dir}")

    depth_estimator = pipeline(
        task="depth-estimation",
        model=checkpoint,
    )

    depth_maps: list[np.ndarray] = []

    for idx, l_path in enumerate(left_imgs):
        image = Image.open(l_path).convert("RGB")
        outputs = depth_estimator(image)

        depth_tensor = outputs["predicted_depth"]
        depth = _to_numpy_depth(depth_tensor)
        depth_maps.append(depth)

        if max_frames is not None and (idx + 1) >= max_frames:
            break

    return depth_maps


def calculate_depth_depthpro(
        seq_name: str,
        checkpoint: str = "apple/DepthPro-hf",
        max_frames: int | None = None,
):
    """
    Monocular metric depth using Apple's DepthPro

    Returns:
        depth_maps: list of (H, W) depth maps in meters.
    """
    data_root = get_path_images()
    left_dir = data_root / seq_name / "image_02" / "data"
    left_imgs = sorted(left_dir.glob("*.png"))
    print(f"DepthPro left dir: {left_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_processor = DepthProImageProcessorFast.from_pretrained(checkpoint)
    model = DepthProForDepthEstimation.from_pretrained(checkpoint).to(device)

    depth_maps: list[np.ndarray] = []

    for idx, l_path in enumerate(left_imgs):
        image = Image.open(l_path).convert("RGB")

        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        post_processed = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        depth_tensor = post_processed[0]["predicted_depth"]  # metric depth (m)
        depth = _to_numpy_depth(depth_tensor)
        depth_maps.append(depth)

        if max_frames is not None and (idx + 1) >= max_frames:
            break

    return depth_maps


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
    depth_norm = depth_vis / max_depth  # [0, 1], far = 1.0
    depth_norm = 1.0 - depth_norm  # invert: near = 1.0, far = 0.0
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


# ---------- NEW: depth statistics + histogram ----------

def summarize_depth(depth: np.ndarray, name: str = "depth", max_depth: float | None = None):
    """
    Print basic statistics for a depth map to help sanity‑check it.
    """
    mask = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        mask &= depth <= max_depth

    vals = depth[mask]
    if vals.size == 0:
        print(f"[{name}] No valid depth values.")
        return

    print(f"[{name}] valid pixels: {vals.size}")
    print(f"[{name}] min={vals.min():.3f} m, max={vals.max():.3f} m, "
          f"mean={vals.mean():.3f} m, median={np.median(vals):.3f} m")
    for p in (5, 25, 50, 75, 95):
        print(f"[{name}] {p}th percentile={np.percentile(vals, p):.3f} m")


def visualize_depth_with_hist(
        depth: np.ndarray,
        max_depth: float = 80.0,
        hist_max_depth: float | None = None,
        title: str = "depth",
        save_path: Path | None = None,
        show: bool = True,
):
    """
    Show depth colormap AND histogram side by side (matplotlib).
    Uses visualize_depth() to make the color image, then plots a histogram.
    """
    if hist_max_depth is None:
        hist_max_depth = max_depth

    # Get the colored depth image without opening an OpenCV window
    depth_color = visualize_depth(
        depth,
        max_depth=max_depth,
        window_name=title,
        save_path=None,
        show=False,
    )

    # Prepare depth values for histogram
    mask = np.isfinite(depth) & (depth > 0)
    vals = depth[mask]
    if vals.size == 0:
        print(f"[{title}] No valid depth values for histogram.")
        return depth_color

    vals_hist = np.clip(vals, 0.0, hist_max_depth)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: depth colormap (convert BGR -> RGB for matplotlib)
    depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    axes[0].imshow(depth_rgb)
    axes[0].set_title("Depth colormap")
    axes[0].axis("off")

    # Right: histogram
    axes[1].hist(vals_hist.ravel(), bins=80, range=(0, hist_max_depth))
    axes[1].set_title("Depth histogram")
    axes[1].set_xlabel("Depth (m)")
    axes[1].set_ylabel("Pixel count")

    fig.suptitle(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return depth_color


# ------------------------------------------------------


if __name__ == "__main__":
    seq_name = "seq_01"
    max_d = 80.0

    # --- Stereo SGBM depth ---
    B = calculate_baseline()
    disp_maps, depth_maps = calculate_disparity(seq_name)
    print(f"Computed {len(disp_maps)} disparity/depth maps for {seq_name} with SGBM")

    first_stereo_depth = depth_maps[0]
    summarize_depth(first_stereo_depth, name="Stereo SGBM", max_depth=None)
    visualize_depth_with_hist(
        first_stereo_depth,
        max_depth=max_d,
        hist_max_depth=max_d,
        title=f"{seq_name} Stereo SGBM frame 0",
        save_path=Path("outputs/depth_sgbm_seq01_frame000_hist.png"),
        show=True,
    )

    # --- Monocular ZoeDepth ---
    zoe_depth_maps = calculate_depth_zoedepth(seq_name, max_frames=1)
    first_zoe_depth = zoe_depth_maps[0]
    summarize_depth(first_zoe_depth, name="ZoeDepth", max_depth=None)
    visualize_depth_with_hist(
        first_zoe_depth,
        max_depth=max_d,
        hist_max_depth=max_d,
        title=f"{seq_name} ZoeDepth frame 0",
        save_path=Path("outputs/depth_zoedepth_seq01_frame000_hist.png"),
        show=True,
    )

    # --- Monocular Depth Anything V2 (metric outdoor) ---
    da_depth_maps = calculate_depth_depthanything_metric(seq_name, max_frames=1)
    first_da_depth = da_depth_maps[0]
    summarize_depth(first_da_depth, name="DepthAnythingV2 Metric", max_depth=None)
    visualize_depth_with_hist(
        first_da_depth,
        max_depth=max_d,
        hist_max_depth=max_d,
        title=f"{seq_name} DepthAnythingV2 Metric frame 0",
        save_path=Path("outputs/depth_da_metric_seq01_frame000_hist.png"),
        show=True,
    )

    # --- Monocular DepthPro ---
    depthpro_maps = calculate_depth_depthpro(seq_name, max_frames=1)
    first_depthpro_depth = depthpro_maps[0]
    summarize_depth(first_depthpro_depth, name="DepthPro", max_depth=None)
    visualize_depth_with_hist(
        first_depthpro_depth,
        max_depth=max_d,
        hist_max_depth=max_d,
        title=f"{seq_name} DepthPro frame 0",
        save_path=Path("outputs/depth_depthpro_seq01_frame000_hist.png"),
        show=True,
    )
