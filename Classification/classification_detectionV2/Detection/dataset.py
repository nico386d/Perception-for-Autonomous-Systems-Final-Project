from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision as tv


class_keep = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
# We use this for the training dataset(Kitti) to filter out unwanted classes
def parse_label_file(path: Path):
    objects = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            cls = parts[0]
            if cls not in class_keep:
                continue

            trunc, occ = float(parts[1]), int(parts[2])
            if trunc > 0.7 or occ > 2:
                continue

            x1, y1, x2, y2 = map(float, parts[4:8])
            objects.append(
                {
                    "class_name": cls,
                    "class_id": class_keep[cls],
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return objects

#Here we locate and passe all the training images and labels to build the dataset for training
def build_detection_data(root_path: Path):

    image_train_folder = root_path / "data_object_image_2" / "training" / "image_2"
    label_train_folder = root_path / "data_object_label_2" / "training" / "label_2"

    image_files = sorted(image_train_folder.glob("*.png"))
    label_files = sorted(label_train_folder.glob("*.txt"))

    image_data = {}

    for img_path, label_path in zip(image_files, label_files):
        objects = parse_label_file(label_path)
        if not objects:
            continue
        image_data[img_path] = objects

    print(f"Total detection training images: {len(image_data)}")
    total_objects = sum(len(objs) for objs in image_data.values())
    print(f"Total objects across all images: {total_objects}")
    return image_data

# Here we build the dataset for validation(DTU dataset)
def build_sequence_detection_data(raw_root: Path, rect_root: Path, seq_name: str, camera: str = "image_02"):

    labels_path = raw_root / seq_name / "labels.txt"
    image_folder = raw_root / seq_name / camera / "data"

    image_data = defaultdict(list)

    with open(labels_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            frame_idx = int(parts[0])
            cls = parts[2]

            if cls not in class_keep:
                continue

            trunc = float(parts[3])
            occ = int(parts[4])
            if trunc > 0.7 or occ > 2:
                continue

            x1, y1, x2, y2 = map(float, parts[6:10])
            img_name = f"{frame_idx:010d}.png"
            image_path = image_folder / img_name

            image_data[image_path].append(
                {
                    "class_id": class_keep[cls],
                    "bbox": [x1, y1, x2, y2],
                }
            )

    print(f"Total images in {seq_name}: {len(image_data)}")
    total_objects = sum(len(objs) for objs in image_data.values())
    print(f"Total objects in {seq_name}: {total_objects}")

    return dict(image_data)


class KittiFullImageDetectionDataset(Dataset):
    def __init__(self, image_data, transform=None, target_size=(640, 640)):
        self.image_paths = list(image_data.keys())
        self.image_data = image_data
        self.transform = transform
        self.target_width, self.target_height = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        objects = self.image_data[image_path]

        # Read image as tensor [C,H,W] in [0,1]
        image = tv.io.read_image(str(image_path)).float() / 255.0
        _, orig_h, orig_w = image.shape

        scale_x = self.target_width / orig_w
        scale_y = self.target_height / orig_h

        bboxes = []
        labels = []
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            # Scale to target size
            bboxes.append(
                [
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y,
                ]
            )
            # Shift labels by +1 so they are in [1..num_classes-1]
            labels.append(obj["class_id"] + 1)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transform (resize / normalize) to image
        image_pil = tv.transforms.functional.to_pil_image(image)
        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = image

        target = {
            "boxes": bboxes,
            "labels": labels,
        }

        return image_tensor, target
