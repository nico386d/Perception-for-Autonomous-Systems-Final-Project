from pathlib import Path

# Target resolution *scale* for detection model.
# GeneralizedRCNNTransform will resize the shorter side to MIN_SIZE
# and keep the longer side <= MAX_SIZE while preserving aspect ratio.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

# Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Limit how many KITTI images to use for training (set to None to use all)
MAX_TRAIN_IMAGES = 20

# (kept for possible custom losses later)
LAMBDA_BBOX = 1.0
LAMBDA_CLS = 1.0

# Paths
# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]   
DATA_ROOT = PROJECT_ROOT.parent / "Data"            

# DTU sequences to evaluate on after training
VAL_SEQUENCES = ["seq_01", "seq_02"]
