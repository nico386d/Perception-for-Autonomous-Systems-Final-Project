# yolo_eval_sequences.py

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO

from config import DATA_ROOT, BATCH_SIZE, VAL_SEQUENCES, IMAGE_WIDTH, IMAGE_HEIGHT  # :contentReference[oaicite:0]{index=0}
from dataset import (build_sequence_detection_data, KittiFullImageDetectionDataset, class_keep)  
from train import collate_fn 


def yolo_predict_batch(model, images, imgsz):
    preds = []

    device_str = "0" if torch.cuda.is_available() else "cpu"

    for img in images:
        # img: [C,H,W] in [0,1] float -> HWC uint8
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # run YOLO; we get a list with one result
        result = model(img_np, imgsz=imgsz, device=device_str, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = result.boxes.xyxy.cpu().to(torch.float32)      # [N,4]
            scores = result.boxes.conf.cpu().to(torch.float32)     # [N]
            # YOLO classes: 0..C-1 ; our GT uses 1..C with 0=background
            labels = result.boxes.cls.cpu().to(torch.int64) + 1    # [N]

        preds.append(
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        )

    return preds


def evaluate_yolo_on_sequences(weights_path: str = "yolo11x.pt"):
    model = YOLO(weights_path)

    imgsz = max(IMAGE_WIDTH, IMAGE_HEIGHT)

    rect_root = DATA_ROOT / "34759_final_project_rect"
    print(f"Using rect_root = {rect_root}")

    print("\n" + "=" * 60)
    print(f"Evaluating YOLO model '{weights_path}' on DTU VAL_SEQUENCES")
    print("Sequences:", VAL_SEQUENCES)
    print("=" * 60)

    results_all = {}


    for seq_name in VAL_SEQUENCES:
        print(f"\n--- Sequence: {seq_name} ---")
        print(f"Loading GT from: {rect_root / seq_name}")

        seq_data = build_sequence_detection_data(rect_root, seq_name)

        dataset = KittiFullImageDetectionDataset(seq_data)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

        metric = MeanAveragePrecision(iou_type="bbox")

        for images, targets in loader:
            preds = yolo_predict_batch(model, images, imgsz=imgsz)

            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            preds_cpu = preds

            metric.update(preds_cpu, targets_cpu)

        res = metric.compute()
        results_all[seq_name] = res

        print(f"[{seq_name}] images: {len(dataset)}")
        print(f"mAP (IoU=0.50:0.95): {res['map'].item():.4f}")
        print(f"mAP@0.5:            {res['map_50'].item():.4f}")
        print(f"mAP@0.75:           {res['map_75'].item():.4f}")

    return results_all


if __name__ == "__main__":
    evaluate_yolo_on_sequences("yolo11x.pt")
