import torchvision.models as models
from torch import nn

def build_resnet50_classifier(number_of_classes=3, freeze_until_layer=""):
    pretrained_weights = models.ResNet50_Weights.IMAGENET1K_V2
    network = models.resnet50(weights=pretrained_weights)
    network.fc = nn.Linear(network.fc.in_features, number_of_classes)

    if freeze_until_layer in ("layer2", "layer3"):
        frozen_layers = ["conv1", "bn1", "layer1"]
        if freeze_until_layer == "layer3":
            frozen_layers += ["layer2"]
        for name, parameter in network.named_parameters():
            if any(name.startswith(layer) for layer in frozen_layers):
                parameter.requires_grad = False

    return network
