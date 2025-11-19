import torch
from torch import nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import class_keep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

# We add 1 for the background class (label 0)
NUM_CLASSES = len(class_keep) + 1  # background + 3 object classes


def create_model(num_classes: int = NUM_CLASSES) -> nn.Module:

    # ResNet50 option
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # mobilenet option
    #weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    #model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace the head to match our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)
    return model


#weights = models.ResNet50_Weights.IMAGENET1K_V2
#model = models.resnet50(weights=weights)
#model.fc = nn.Linear(model.fc.in_features, num_classes)

#weights = models.VGG16_Weights.IMAGENET1K_V1
#model = models.vgg16(weights=weights)
#in_features = model.classifier[6].in_features
#model.classifier[6] = nn.Linear(in_features, num_classes)

#weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
#model = models.mobilenet_v2(weights=weights)
#model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

#weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
#model = models.efficientnet_b0(weights=weights)
#model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

#weights = models.DenseNet121_Weights.IMAGENET1K_V1
#model = models.densenet121(weights=weights)
#model.classifier = nn.Linear(model.classifier.in_features, num_classes)