import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv

from config import (IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE as DEFAULT_BATCH_SIZE)
from dataset import KittiFullImageDetectionDataset
from model import create_model, device
from train import (collate_fn, train_one_epoch_detection, evaluate_detection_map,)

class DetectionTrainer:
    def __init__(
        self,
        train_data,
        val_data,
        transform=None,
        target_size=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.model = create_model()

        # Default image size if not provided
        if target_size is None:
            target_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        self.target_size = target_size
        self.batch_size = batch_size

        # Default transform if not provided
        if transform is None:
            transform = tv.transforms.Compose(
                [
                    tv.transforms.Resize((self.target_size[1], self.target_size[0])),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )

        self.transform = transform

        self.train_dataset = KittiFullImageDetectionDataset(
            train_data, transform=self.transform, target_size=self.target_size
        )
        self.val_dataset = KittiFullImageDetectionDataset(
            val_data, transform=self.transform, target_size=self.target_size
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def fit(self, epochs: int = 5, save_path: str = "best_detection_model.pt"):
        best_val_map = 0.0
        self.best_model_path = save_path

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch_detection(
                self.model,
                self.train_loader,
                self.optimizer,
                epoch,
            )

            val_results = evaluate_detection_map(self.model, self.val_loader)
            val_map = val_results["map"].item()
            val_map50 = val_results["map_50"].item()
            val_map75 = val_results["map_75"].item()

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_mAP={val_map:.4f} | "
                f"val_mAP@0.5={val_map50:.4f} | "
                f"val_mAP@0.75={val_map75:.4f}"
            )

            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(self.model.state_dict(), self.best_model_path)

        print(f"Best validation mAP: {best_val_map:.4f}")

    def load_best(self, path: str | None = None):
        if path is None:
            path = getattr(self, "best_model_path", "best_detection_model.pt")

        state_dict = torch.load(path, map_location=device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded best detection model from: {path}")

    def evaluate_loader(self, loader):
        self.model.eval()
        return evaluate_detection_map(self.model, loader)

    def evaluate_sequences(self, seq_data_dict, batch_size: int | None = None):
        if batch_size is None:
            batch_size = self.batch_size

        print("\n" + "=" * 60)
        print("Evaluating on DTU sequences")
        print("=" * 60)

        results_all = {}

        for seq_name, data in seq_data_dict.items():
            ds = KittiFullImageDetectionDataset(
                data,
                transform=self.transform,
                target_size=self.target_size,
            )
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

            results = self.evaluate_loader(loader)
            results_all[seq_name] = results

            print(f"\nSequence {seq_name}:")
            print(f"  mAP (IoU=0.50:0.95): {results['map'].item():.4f}")
            print(f"  mAP@0.5:            {results['map_50'].item():.4f}")
            print(f"  mAP@0.75:           {results['map_75'].item():.4f}")

        return results_all
