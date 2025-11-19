from pathlib import Path
import random
import kagglehub

from config import (MAX_TRAIN_IMAGES, DATA_ROOT, VAL_SEQUENCES, EPOCHS, BATCH_SIZE)
from dataset import (build_detection_data, build_sequence_detection_data)
from trainer import DetectionTrainer


def main():
    # first we donwload the KITTI dataset from Kaggle
    root_path = Path(kagglehub.dataset_download("klemenko/kitti-dataset")).resolve()
    print("KITTI dataset downloaded to:", root_path)

    # Then we filter and build the training and val set from kitti
    image_data = build_detection_data(root_path)
    image_paths = list(image_data.keys())

    # We shuffle and split the dataset into train and val sets
    random.seed(42)
    random.shuffle(image_paths)

    if MAX_TRAIN_IMAGES is not None:
        image_paths = image_paths[:MAX_TRAIN_IMAGES]

    # Simple 80/20 split
    split_idx = int(0.8 * len(image_paths))
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]

    train_data = {p: image_data[p] for p in train_paths}
    val_data = {p: image_data[p] for p in val_paths}

    print(f"Train images: {len(train_data)}")
    print(f"Val images:   {len(val_data)}")

    # Next we build the validation sequences from DTU dataset
    raw_root = DATA_ROOT / "34759_final_project_raw"
    rect_root = DATA_ROOT / "34759_final_project_rect"

    seq_data = {}
    for seq in VAL_SEQUENCES:
        seq_data[seq] = build_sequence_detection_data(raw_root, rect_root, seq)

    # Finally we create the trainer and start training
    trainer = DetectionTrainer(
        train_data=train_data,
        val_data=val_data,
        batch_size=BATCH_SIZE,
    )

    trainer.fit(epochs=EPOCHS, save_path="best_detection_model.pt")

    trainer.load_best("best_detection_model.pt")
    trainer.evaluate_sequences(seq_data, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
