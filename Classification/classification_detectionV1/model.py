import torch
from torch import nn
import torchvision.models as models

from dataset import class_keep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("CUDA device name: ", torch.cuda.get_device_name(0))

num_classes = len(class_keep)

weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


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