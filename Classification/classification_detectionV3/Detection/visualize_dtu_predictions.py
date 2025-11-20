import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import DATA_ROOT
from dataset import (
    build_sequence_detection_data,
    KittiFullImageDetectionDataset,
    class_keep,
)
from model import create_model, device


# ---------- helper: draw one image with GT + predictions ----------

def show_image_with_boxes(img, target, pred, score_thresh=0.5):
    """
    img: tensor [C,H,W] in [0,1]
    target: dict with "boxes" and "labels"
    pred: dict with "boxes", "labels", "scores"
    """

    # Convert image to HWC numpy
    img_np = img.permute(1, 2, 0).cpu().numpy()

    # Inverse label mapping: 1..num_classes -> class name
    id_to_class = {v + 1: k for k, v in class_keep.items()}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img_np)
    ax.set_axis_off()

    # --- Ground truth boxes (green) ---
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]

    for box, lab in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        cls_name = id_to_class.get(int(lab.item()), str(int(lab.item())))
        ax.text(
            x1,
            y1 - 4,
            f"GT: {cls_name}",
            fontsize=8,
            color="lime",
            bbox=dict(facecolor="black", alpha=0.4, linewidth=0),
        )

    # --- Predicted boxes (red) ---
    if pred is not None:
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]

        for box, lab, score in zip(boxes, labels, scores):
            if score.item() < score_thresh:
                continue

            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            cls_name = id_to_class.get(int(lab.item()), str(int(lab.item())))
            ax.text(
                x1,
                y2 + 10,
                f"Pred: {cls_name} ({score:.2f})",
                fontsize=8,
                color="red",
                bbox=dict(facecolor="black", alpha=0.4, linewidth=0),
            )

    plt.tight_layout()
    plt.show()


def main():
    # ----- 1. Load trained model -----
    model = create_model()
    state_dict = torch.load("best_detection_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ----- 2. Build DTU seq dataset -----
    raw_root = DATA_ROOT / "34759_final_project_raw"
    rect_root = DATA_ROOT / "34759_final_project_rect"

    seq_name = "seq_01"   # change to "seq_02" to inspect that sequence
    seq_data = build_sequence_detection_data(raw_root, rect_root, seq_name)

    dataset = KittiFullImageDetectionDataset(seq_data)

    print(f"{seq_name}: {len(dataset)} images")

    # Pick a few random indices
    num_examples = 5
    indices = random.sample(range(len(dataset)), k=min(num_examples, len(dataset)))

    for idx in indices:
        img, target = dataset[idx]

        with torch.no_grad():
            pred = model([img.to(device)])[0]

        show_image_with_boxes(img, target, pred, score_thresh=0.5)


if __name__ == "__main__":
    main()
