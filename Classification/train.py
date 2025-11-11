import torch
from torch import nn
from torch.utils.data import DataLoader
from config import *
from utils import set_random_seed, create_training_transforms, create_validation_transforms
from dataset import KittiTrainingDataset, SequenceValidationDataset
from model import build_resnet50_classifier
from trainer import ModelTrainer

def main():
    set_random_seed(RANDOM_SEED)

    training_dataset = KittiTrainingDataset(KITTI_ROOT_PATH, create_training_transforms(IMAGE_SIZE))
    validation_dataset = SequenceValidationDataset(VALIDATION_CSV_PATH, create_validation_transforms(IMAGE_SIZE))

    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUMBER_OF_WORKERS, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUMBER_OF_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50_classifier(number_of_classes=len(CLASSES), freeze_until_layer=FREEZE_UNTIL).to(device)

    if USE_CLASS_WEIGHTS:
        class_counts = torch.zeros(len(CLASSES))
        for _, class_id, _ in training_dataset.samples:
            class_counts[class_id] += 1
        class_weights = (class_counts.sum() / (class_counts + 1e-6))
        class_weights = (class_weights / class_weights.mean()).to(device)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    trainer = ModelTrainer(model, optimizer, scheduler, device, OUTPUT_DIRECTORY, USE_MIXED_PRECISION)
    trainer.fit(training_loader, validation_loader, loss_function, EPOCHS)

    print("\nBest model saved to:", trainer.output_path)

if __name__ == "__main__":
    main()
