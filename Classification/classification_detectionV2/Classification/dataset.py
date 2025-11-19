from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class_keep = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}

def parse_label_file(path):

    objects = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue  # Skip empty lines

            cls = parts[0]
            if cls not in class_keep:
                continue

            # truncation / occlusion filtering
            trunc, occ = float(parts[1]), int(parts[2])
            if trunc > 0.7 or occ > 2:
                continue

            # bounding box
            x1, y1, x2, y2 = map(float, parts[4:8])
            objects.append(
                {
                    "class_name": cls,
                    "class_id": class_keep[cls],
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return objects

#For training dataset of kitti
def build_samples(root_path: Path):
    image_train_folder = root_path / "data_object_image_2" / "training" / "image_2"
    label_train_folder = root_path / "data_object_label_2" / "training" / "label_2"

    image_files = sorted(image_train_folder.glob("*.png"))
    label_files = sorted(label_train_folder.glob("*.txt"))

    label_data = [parse_label_file(p) for p in label_files]

    samples = [
        {
            "image_path": img_path,
            "class_id": obj["class_id"],
            "bbox": obj["bbox"],
        }
        for img_path, objects in zip(image_files, label_data)
        for obj in objects
    ]

    print("Total number of objects:", len(samples))
    if samples:
        print("Example sample:", samples[0])

    return samples


class KittiCropClassificationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        x1, y1, x2, y2 = sample["bbox"]
        image = image.crop((x1, y1, x2, y2))
        if self.transform is not None:
            image = self.transform(image)
        label = sample["class_id"]
        return image, label

# For validation dataset of the dtu validation set
def build_sequence_samples(raw_root: Path, rect_root: Path, seq_name: str, camera: str = "image_02"):
    labels_path = raw_root / seq_name / "labels.txt"
    image_folder = raw_root / seq_name / camera / "data"

    samples = []

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
            occ   = int(parts[4])
            if trunc > 0.7 or occ > 2:
                continue

            x1, y1, x2, y2 = map(float, parts[6:10])

            img_name = f"{frame_idx:010d}.png"
            image_path = image_folder / img_name

            samples.append(
                {
                    "image_path": image_path,
                    "class_id": class_keep[cls],
                    "bbox": [x1, y1, x2, y2],
                }
            )

    print(f"Total number of objects in {seq_name}: {len(samples)}")
    return samples
