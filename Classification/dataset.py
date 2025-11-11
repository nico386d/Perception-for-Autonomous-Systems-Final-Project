from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from config import CLASS_NAME_TO_ID

def parse_kitti_label_file(label_file_path):
    objects = []
    with open(label_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if not parts or parts[0] not in CLASS_NAME_TO_ID:
                continue
            truncation = float(parts[1])
            occlusion = int(parts[2])
            if truncation > 0.7 or occlusion > 2:
                continue
            left, top, right, bottom = map(float, parts[4:8])
            class_name = parts[0]
            class_id = CLASS_NAME_TO_ID[class_name]
            objects.append((class_id, (left, top, right, bottom)))
    return objects


class KittiTrainingDataset(Dataset):
    def __init__(self, dataset_root_path, image_transform=None):
        image_directory = Path(dataset_root_path) / "data_object_image_2" / "training" / "image_2"
        label_directory = Path(dataset_root_path) / "data_object_label_2" / "training" / "label_2"

        self.samples = []
        for label_file in sorted(label_directory.glob("*.txt")):
            label_objects = parse_kitti_label_file(label_file)
            image_file = image_directory / f"{label_file.stem}.png"
            for class_id, bounding_box in label_objects:
                self.samples.append((image_file, class_id, bounding_box))

        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_id, (x1, y1, x2, y2) = self.samples[index]
        image = Image.open(image_path).convert("RGB").crop((x1, y1, x2, y2))
        if self.image_transform:
            image = self.image_transform(image)
        return image, class_id


class SequenceValidationDataset(Dataset):
    def __init__(self, csv_file_path, image_transform=None):
        dataframe = pd.read_csv(csv_file_path)
        dataframe = dataframe[dataframe["label"].str.capitalize().isin(CLASS_NAME_TO_ID)]
        self.samples = [(Path(path), CLASS_NAME_TO_ID[label.capitalize()])
                        for path, label in zip(dataframe["img_path"], dataframe["label"])]
        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_id = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        return image, class_id
