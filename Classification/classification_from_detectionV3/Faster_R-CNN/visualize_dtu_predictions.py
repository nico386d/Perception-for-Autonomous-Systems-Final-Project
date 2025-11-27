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

def show_image_with_boxes(img, target, pred, score_thresh=0.5, save_path=None):
    """
    img: tensor [C,H,W] in [0,1]
    target: dict with "boxes" and "labels" or None
    pred: dict with "boxes", "labels", "scores"
    save_path: Path or None. If given, saves figure instead of (only) showing.
    """

    # Convert image to HWC numpy
    img_np = img.permute(1, 2, 0).cpu().numpy()

    # Inverse label mapping: 1..num_classes -> class name
    id_to_class = {v + 1: k for k, v in class_keep.items()}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img_np)
    ax.set_axis_off()

    # --- Ground truth boxes (green) ---
    if target is not None and "boxes" in target and len(target["boxes"]) > 0:
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        for box, lab in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=1,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
            cls_name = id_to_class.get(int(lab.item()), str(int(lab.item())))
            ax.text(
                x1,
                y1 - 4,
                f"GT: {cls_name}",
                fontsize=6,
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
                linewidth=1,
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
                fontsize=6,
                color="red",
                bbox=dict(facecolor="black", alpha=0.4, linewidth=0),
            )

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def evenly_spaced_indices(n_items, n_samples):
    """
    Returns n_samples indices evenly spaced over [0, n_items-1].
    If n_items < n_samples, returns all indices.
    """
    if n_items <= n_samples:
        return list(range(n_items))

    # torch.linspace gives float endpoints; round to nearest int
    idxs = torch.linspace(0, n_items - 1, steps=n_samples)
    idxs = torch.round(idxs).to(torch.long).tolist()

    # remove potential duplicates from rounding, while preserving order
    seen = set()
    uniq = []
    for i in idxs:
        if i not in seen:
            uniq.append(i)
            seen.add(i)

    # If rounding caused fewer than n_samples, fill by adding missing neighbors
    while len(uniq) < n_samples:
        # simple fill: add closest missing indices
        for i in range(n_items):
            if i not in seen:
                uniq.append(i)
                seen.add(i)
            if len(uniq) == n_samples:
                break

    return uniq


def build_image_only_data(rect_root: Path, seq_name: str, camera: str = "image_02"):
    """
    Fallback loader for sequences without labels (e.g. seq_03).
    Creates an empty object list for each image.
    """
    image_folder = rect_root / seq_name / camera / "data"
    pngs = sorted(image_folder.glob("*.png"))
    if len(pngs) == 0:
        raise FileNotFoundError(f"No PNG files found in {image_folder}")

    # map: image_path -> []  (no objects)
    return {p: [] for p in pngs}


def main():
    # ----- 1. Load trained model -----
    model = create_model()

    ckpt_path = Path("best_detection_model_ResNet50.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    model_name = ckpt_path.stem  # e.g. "best_detection_model_ResNet50"

    # ----- 2. Build DTU seq dataset -----
    raw_root = DATA_ROOT / "34759_final_project_raw"
    rect_root = DATA_ROOT / "34759_final_project_rect"

    seq_name = "seq_03"   # change as needed

    # Try to use normal detection data; if labels are missing, fall back to image-only
    try:
        seq_data = build_sequence_detection_data(raw_root, rect_root, seq_name)
        has_gt = True
    except FileNotFoundError:
        seq_data = build_image_only_data(rect_root, seq_name)
        has_gt = False

    dataset = KittiFullImageDetectionDataset(seq_data)

    print(f"{seq_name}: {len(dataset)} images (has_gt={has_gt})")

    # ----- 3. Pick 10 evenly spaced frames -----
    num_examples = 10
    indices = evenly_spaced_indices(len(dataset), num_examples)

    # ----- 4. Output folder -----
    out_dir = Path("detection_viz") / model_name / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        img, target = dataset[idx]

        # If the sequence has no annotations, ignore target entirely
        if not has_gt:
            target = None

        with torch.no_grad():
            pred = model([img.to(device)])[0]

        # filename includes model, sequence, and frame index
        save_path = out_dir / f"{model_name}_{seq_name}_frame_{idx:06d}.png"

        show_image_with_boxes(
            img,
            target,
            pred,
            score_thresh=0.5,
            save_path=save_path,
        )

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
