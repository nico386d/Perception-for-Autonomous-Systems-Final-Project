import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv

from config import (CROP_WIDTH, CROP_HEIGHT, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,)
from dataset import (KittiCropClassificationDataset)
from model import create_model, device
from train import train_one_epoch, evaluate_classifier


class ClassificationTrainer:

    def __init__(self, train_samples, val_samples, transform=None, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, weight_decay: float = WEIGHT_DECAY):
        self.model = create_model()

        if transform is None:
            transform = tv.transforms.Compose(
                [
                    tv.transforms.Resize((CROP_HEIGHT, CROP_WIDTH)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )

        self.train_dataset = KittiCropClassificationDataset(train_samples, transform)
        self.val_dataset = KittiCropClassificationDataset(val_samples, transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, epochs: int = EPOCHS, save_path: str = "best_classification_model.pt"):
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion, epoch)
            val_acc = evaluate_classifier(self.model, self.val_loader)
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.3f} | "
                f"val_acc={val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)

        print(f"Best validation accuracy: {best_val_acc:.3f}")

    def evaluate_val(self):
        
        acc = evaluate_classifier(self.model, self.val_loader)
        print(f"Validation accuracy: {acc:.3f}")
        return acc
