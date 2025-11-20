import torch
from torch import nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from config import IMAGE_WIDTH, IMAGE_HEIGHT
from dataset import class_keep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

# We add 1 for the background class (label 0)
NUM_CLASSES = len(class_keep) + 1  # background + 3 object classes


def create_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Create a Faster R-CNN model with a ResNet50 backbone, pretrained on COCO,
    and replace the prediction head to match our number of classes.
    We also attach a GeneralizedRCNNTransform so images and boxes are resized
    consistently inside the model.
    """

    # Option 1: ResNet-50 FPN backbone
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Option 2: MobileNet FPN backbone (uncomment if you want to try it)
    # weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    # model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace the classification head to match our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Override the internal transform so it resizes + normalizes consistently
    min_size = min(IMAGE_WIDTH, IMAGE_HEIGHT)
    max_size = max(IMAGE_WIDTH, IMAGE_HEIGHT)

    # Same mean/std as COCO / ImageNet
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    model.transform = GeneralizedRCNNTransform(
        min_size=min_size,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std,
    )

    model = model.to(device)
    return model
