from pathlib import Path

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_TRAIN_IMAGES = None #we set to None to use all images, was used to limit for quick testing
LAMBDA_BBOX = 1.0 
LAMBDA_CLS = 1.0 

PROJECT_ROOT = Path(__file__).resolve().parents[1]   
DATA_ROOT = PROJECT_ROOT.parent / "Data"            
VAL_SEQUENCES = ["seq_01", "seq_02"]
