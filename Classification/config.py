from pathlib import Path

CLASSES = ["Pedestrian", "Cyclist", "Car"]
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASSES)}

KITTI_ROOT_PATH = Path("data/KITTI_ROOT")
VALIDATION_CSV_PATH = Path("data/val_crops/labels.csv")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUMBER_OF_WORKERS = 4
USE_MIXED_PRECISION = True
FREEZE_UNTIL = ""
USE_CLASS_WEIGHTS = True
OUTPUT_DIRECTORY = Path("outputs/ckpts")
RANDOM_SEED = 42
