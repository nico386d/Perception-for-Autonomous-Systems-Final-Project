from pathlib import Path
import random

import kagglehub
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv

from dataset import (build_samples, KittiCropClassificationDataset, build_sequence_samples)
from model import model
from train import train_one_epoch, evaluate_classifier



# Hyperparameters
full_width  = 244
full_height = 244
batch_size = 16
epochs = 5
learning_rate = 1e-4
weight_decay = 1e-4
max_samples = 1500

def main():
    root_path = Path(kagglehub.dataset_download("klemenko/kitti-dataset")).resolve()
    print("Dataset downloaded to:", root_path)

    samples = build_samples(root_path)

    random.seed(42)
    random.shuffle(samples)
    train_samples = samples[:max_samples]  
    project_root = Path(__file__).resolve().parents[1]
    raw_root  = project_root.parent  / "Data" / "34759_final_project_raw"
    rect_root = project_root.parent  / "Data" / "34759_final_project_rect"

    val_seq1 = build_sequence_samples(raw_root, rect_root, "seq_01", camera="image_02")
    val_seq2 = build_sequence_samples(raw_root, rect_root, "seq_02", camera="image_02")
    val_samples = val_seq1 + val_seq2

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")



    # Transforms
    transform = tv.transforms.Compose([
        tv.transforms.Resize((full_height, full_width)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
    ])

    # Datasets / loaders
    train_dataset = KittiCropClassificationDataset(train_samples, transform)
    val_dataset   = KittiCropClassificationDataset(val_samples,   transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)


    # Optimizer + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    
    best_validation_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_function, epoch)
        val_acc = evaluate_classifier(model, val_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f}"
        )


        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")


if __name__ == "__main__":
    main()
