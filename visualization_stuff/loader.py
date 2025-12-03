# loader_aabb.py
from pathlib import Path
from detection import Detection3D

def load_detections_from_aabb(aabb_path: Path,
                              min_conf: float = 0.0) -> list[Detection3D]:
    """
    Load 3D detections for one frame from boxes3d_aabb file.

    aabb_path : path to 'boxes3d_aabb' (or boxes3d_aabb.txt)
    min_conf  : optional confidence threshold
    """
    aabb_path = Path(aabb_path)
    detections: list[Detection3D] = []

    with aabb_path.open("r") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            # Expecting at least 16 tokens as in your example
            if len(parts) < 16:
                print(f"[WARN] Skipping line {line_no} in {aabb_path}, not enough fields.")
                continue

            det_id   = int(parts[0])
            cls_id   = int(parts[1])
            cls_name = parts[2]          # not used by tracker
            conf     = float(parts[3])

            if conf < min_conf:
                continue  # skip low-confidence detections if you want

            # center_u, center_v, depth_m (not used for tracking)
            # parts[4], parts[5], parts[6]

            xmin = float(parts[7])
            ymin = float(parts[8])
            zmin = float(parts[9])
            xmax = float(parts[10])
            ymax = float(parts[11])
            zmax = float(parts[12])

            w = float(parts[13])
            h = float(parts[14])
            l = float(parts[15])

            # 3D center in camera coordinates
            px = 0.5 * (xmin + xmax)
            py = 0.5 * (ymin + ymax)
            pz = 0.5 * (zmin + zmax)

            det = Detection3D(
                position=(px, py, pz),
                yaw=0.0,          # axis-aligned AABB, so yaw = 0 for now
                dimensions=(w, h, l),
                class_id=cls_id,
                confidence=conf,
                frame_id=None,
            )
            detections.append(det)

    return detections
