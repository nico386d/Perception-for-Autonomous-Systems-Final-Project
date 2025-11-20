from pathlib import Path
import random

import kagglehub

from config import (MAX_TRAIN_SAMPLES, MAX_VALIDATION_SAMPLES, RANDOM_SEED, RAW_ROOT, RECT_ROOT)
from dataset import build_samples, build_sequence_samples
from trainer import ClassificationTrainer


def main():
    # Download KITTI dataset from Kaggle
    root_path = Path(kagglehub.dataset_download("klemenko/kitti-dataset")).resolve()
    print("Dataset downloaded to:", root_path)

    # train samples
    samples = build_samples(root_path)
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    train_samples = samples[:MAX_TRAIN_SAMPLES]

    # validation samples
    val_seq1 = build_sequence_samples(RAW_ROOT, RECT_ROOT, "seq_01", camera="image_02")
    val_seq2 = build_sequence_samples(RAW_ROOT, RECT_ROOT, "seq_02", camera="image_02")
    val_samples = val_seq1 + val_seq2

    # e.g. limit validation to 400 random DTU samples
    random.seed(RANDOM_SEED)
    random.shuffle(val_samples)
    val_samples = val_samples[:MAX_VALIDATION_SAMPLES]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    trainer = ClassificationTrainer(train_samples, val_samples)
    trainer.fit()


if __name__ == "__main__":
    main()
