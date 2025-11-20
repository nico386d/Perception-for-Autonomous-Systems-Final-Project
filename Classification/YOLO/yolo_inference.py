from collections import defaultdict

from ultralytics import YOLO
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

def get_path_weights():
    ROOT = Path(__file__).resolve().parent.parent.parent
    weights_path = ROOT / "weights" / "Classification" / "best-x.pt"
    return Path(weights_path)

def get_path_images():
    ROOT = Path(__file__).resolve().parent.parent.parent
    dat_path = ROOT / "Classification" / "34759_final_project_rect"
    return Path(dat_path)

def load_kitti_labels(labels_path: Path):
    """
    Parse labels.txt into a dict: frame_idx -> list of {bbox, label}
    bbox = [x_min, y_min, x_max, y_max] in rectified image coordinates.
    """

    KITTI_TO_YOLO = {
        "Pedestrian": 3,
        "Cyclist": 5,
        "Car": 0,
    }

    annotations = defaultdict(list)

    with labels_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            frame = int(parts[0])
            obj_type = parts[2]

            if obj_type not in KITTI_TO_YOLO:
                continue

            bbox_left = float(parts[6])
            bbox_top = float(parts[7])
            bbox_right = float(parts[8])
            bbox_bottom = float(parts[9])

            cls_id = KITTI_TO_YOLO[obj_type]

            annotations[frame].append(
                {
                    "bbox": [bbox_left, bbox_top, bbox_right, bbox_bottom],
                    "label": cls_id,
                }
            )

    return annotations

def validate_sequence(model: YOLO, seq_name: str, IMAGE_ROOT : Path):

    seq_dir = IMAGE_ROOT / seq_name
    labels_path = seq_dir / "labels.txt"
    IMAGE_SUBDIR = "image_02"

    print(f"Evaluating {seq_name}")

    # Load ground-truth annotations
    gt_by_frame = load_kitti_labels(labels_path)

    # Init mAP metric
    metric = MeanAveragePrecision(iou_type="bbox")

    # Images
    img_dir = seq_dir / IMAGE_SUBDIR / "data"
    img_paths = sorted(img_dir.glob("*.png"))

    if not img_paths:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    for frame_idx, img_path in enumerate(img_paths):
        # --- YOLO prediction ---
        # returns a list; we take [0] because it's one image
        preds = model(img_path, verbose=False)[0]

        if len(preds.boxes) > 0:
            pred_boxes = preds.boxes.xyxy.cpu()
            pred_scores = preds.boxes.conf.cpu()
            pred_labels = preds.boxes.cls.to(torch.int64).cpu()
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores = torch.zeros((0,), dtype=torch.float32)
            pred_labels = torch.zeros((0,), dtype=torch.int64)

        pred = [
            {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels,
            }
        ]

        # --- Ground truth for this frame ---
        if frame_idx in gt_by_frame:
            objs = gt_by_frame[frame_idx]
            gt_boxes = torch.tensor(
                [o["bbox"] for o in objs], dtype=torch.float32
            )
            gt_labels = torch.tensor(
                [o["label"] for o in objs], dtype=torch.int64
            )
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.int64)

        target = [
            {
                "boxes": gt_boxes,
                "labels": gt_labels,
            }
        ]

        # Update metric
        metric.update(pred, target)

    # Compute metrics for this sequence
    result = metric.compute()
    print(f"[{seq_name}] mAP (IoU=0.50:0.95): {result['map'].item():.4f}")
    print(f"[{seq_name}] mAP@0.5: {result['map_50'].item():.4f}")
    print(f"[{seq_name}] mAP@0.75: {result['map_75'].item():.4f}")
    print()
modelp = YOLO(get_path_weights())


prediction_results = modelp.predict(
    "https://ultralytics.com/assets/kitti-inference-im0.png",
    save=True,
)

if __name__ == "__main__":

    WEIGHTS_PATH = get_path_weights()
    IMAGE_PATH = get_path_images()

    modelp = YOLO(WEIGHTS_PATH)

    # Evaluate validation sequences
    for seq in ["seq_01", "seq_02"]:
        validate_sequence(modelp, seq,IMAGE_PATH )

