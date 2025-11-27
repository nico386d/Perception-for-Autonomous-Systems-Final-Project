from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision as tv

class_keep = {"Pedestrian": 0, "Cyclist": 1, "Car": 2} # KITTI had more classes, we keep only these


#This function the label file and returns list of objects for the KITTI annotation format
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

# here we find the path of KITTI dataset and passes it into previous  function "parse_label_file"
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


# DTU's dataset is in a different format, so we have a separate builder function
def build_sequence_detection_data(rect_root: Path, seq_name: str,camera: str = "image_02"):

    labels_path = rect_root / seq_name / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.txt missing in {labels_path}")

    image_folder = rect_root / seq_name / camera / "data"
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder missing: {image_folder}")

    pngs = sorted(image_folder.glob("*.png"))
    if len(pngs) == 0:
        raise FileNotFoundError(f"No PNG files found in {image_folder}")

    # seq_01 → "000000" → pad = 6
    # seq_02 → "0000000000" → pad = 10
    pad_width = len(pngs[0].stem)
    print(f"[{seq_name}] Using pad_width = {pad_width}")

    image_data = defaultdict(list)
    missing = 0

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

            # APPLY CORRECT ZERO-PADDING
            img_name = f"{frame_idx:0{pad_width}d}.png"
            img_path = image_folder / img_name

            if not img_path.exists():
                missing += 1
                continue

            image_data[img_path].append(
                {"class_id": class_keep[cls], "bbox": [x1, y1, x2, y2]}
            )

    print(f"Total images in {seq_name}: {len(image_data)}")
    print(f"Total objects in {seq_name}: {sum(len(x) for x in image_data.values())}")
    if missing > 0:
        print(f"Skipped {missing} labels (missing images)")

    return dict(image_data)



# Dataset class for loading full KITTI images and their detection annotations (boxes + labels) for torch.
class KittiFullImageDetectionDataset(Dataset):


    def __init__(self, image_data):
        self.image_paths = list(image_data.keys())
        self.image_data = image_data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        objects = self.image_data[image_path]

        # Read image as tensor [C,H,W] uint8, then convert to float [0,1]
        image = tv.io.read_image(str(image_path)).float() / 255.0

        bboxes = []
        labels = []

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            bboxes.append([x1, y1, x2, y2])
            labels.append(obj["class_id"] + 1)  # +1 because 0 is background

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": bboxes,
            "labels": labels,
        }

        return image, target
