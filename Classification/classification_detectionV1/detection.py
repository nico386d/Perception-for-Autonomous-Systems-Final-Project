from pathlib import Path
import random

import kagglehub
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from sklearn.model_selection import train_test_split

from dataset import parse_label_file, class_keep
from model import device  # re-use same device as classification


full_width = 244
full_height = 244

batch_size = 8
epochs = 5
learning_rate = 1e-4
weight_decay = 1e-4
max_samples = 10000  # limit training images for speed
lambda_bbox = 1.0   # weight for bbox loss in total loss


class KittiFullImageDetectionDataset(Dataset):


    def __init__(self, samples, transform=None, target_size=(full_width, full_height)):
        self.samples = samples
        self.transform = transform
        self.target_width, self.target_height = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        x1, y1, x2, y2 = sample["bbox"]
        class_id = sample["class_id"]

        # open original image
        image = tv.io.read_image(str(image_path)).float() / 255.0  # [C,H,W]
        _, orig_h, orig_w = image.shape

        # scale bbox to target size (same resize as transform)
        scale_x = self.target_width / orig_w
        scale_y = self.target_height / orig_h
        bbox = torch.tensor(
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
            dtype=torch.float32,
        )

        # convert to PIL for torchvision transforms
        image_pil = tv.transforms.functional.to_pil_image(image)

        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = image

        label = torch.tensor(class_id, dtype=torch.long)
        return image_tensor, bbox, label


def build_detection_samples(root_path: Path):

    image_train_folder = root_path / "data_object_image_2" / "training" / "image_2"
    label_train_folder = root_path / "data_object_label_2" / "training" / "label_2"

    image_files = sorted(image_train_folder.glob("*.png"))
    label_files = sorted(label_train_folder.glob("*.txt"))

    samples = []

    for img_path, label_path in zip(image_files, label_files):
        objects = parse_label_file(label_path)
        if not objects:
            continue  # skip images with no relevant objects

        # pick the largest bbox in this image
        def area(obj):
            x1, y1, x2, y2 = obj["bbox"]
            return (x2 - x1) * (y2 - y1)

        obj = max(objects, key=area)
        samples.append(
            {
                "image_path": img_path,
                "class_id": obj["class_id"],
                "bbox": obj["bbox"],
            }
        )

    print("Total detection training images:", len(samples))
    if samples:
        print("Example detection sample:", samples[0])

    return samples


class DetectionNet(nn.Module):


    def __init__(self, num_classes: int = 3):
        super().__init__()

        backbone = tv.models.resnet50(
            weights=tv.models.ResNet50_Weights.IMAGENET1K_V2
        )
        num_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()  # remove original classification head

        self.backbone = backbone
        self.bbox_head = nn.Linear(num_feats, 4)
        self.cls_head = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        bbox = self.bbox_head(feats)
        cls_logits = self.cls_head(feats)
        return bbox, cls_logits


def box_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union


def evaluate_detection(model, data_loader, iou_thresh=0.5):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, gt_bboxes, gt_labels in data_loader:
            images = images.to(device)
            gt_bboxes = gt_bboxes.to(device)
            gt_labels = gt_labels.to(device)

            pred_bboxes, pred_logits = model(images)
            pred_labels = pred_logits.argmax(dim=1)

            for pb, pl, gb, gl in zip(pred_bboxes, pred_labels, gt_bboxes, gt_labels):
                if pl != gl:
                    total += 1
                    continue
                iou = box_iou(pb, gb)
                if iou >= iou_thresh:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def train_one_epoch_detection(model, data_loader, optimizer, bbox_loss_fn, cls_loss_fn, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, gt_bboxes, gt_labels in data_loader:
        images = images.to(device)
        gt_bboxes = gt_bboxes.to(device)
        gt_labels = gt_labels.to(device)

        optimizer.zero_grad()
        pred_bboxes, pred_logits = model(images)

        loss_bbox = bbox_loss_fn(pred_bboxes, gt_bboxes)
        loss_cls = cls_loss_fn(pred_logits, gt_labels)
        loss = lambda_bbox * loss_bbox + loss_cls

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def main():
    # 1) Download KITTI data via kagglehub (same as classification)
    root_path = Path(kagglehub.dataset_download("klemenko/kitti-dataset")).resolve()
    print("Dataset downloaded to:", root_path)

    # 2) Build detection samples (full images)
    samples = build_detection_samples(root_path)

    random.seed(42)
    random.shuffle(samples)
    samples = samples[:max_samples]

    labels = [s["class_id"] for s in samples]

    # train / val split on full images
    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # 3) Transforms for full images
    transform = tv.transforms.Compose([
        tv.transforms.Resize((full_height, full_width)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
    ])

    # 4) Datasets & loaders
    train_dataset = KittiFullImageDetectionDataset(train_samples, transform=transform)
    val_dataset   = KittiFullImageDetectionDataset(val_samples,   transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # 5) Model, optimizer, losses
    num_classes = len(class_keep)
    model = DetectionNet(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    bbox_loss_fn = nn.SmoothL1Loss()
    cls_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # 6) Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_detection(
            model, train_loader, optimizer, bbox_loss_fn, cls_loss_fn, epoch
        )
        val_acc = evaluate_detection(model, val_loader, iou_thresh=0.5)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_det_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"\nBest validation detection accuracy (IoU>=0.5): {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
