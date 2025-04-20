import torch.nn as nn
from torchvision import models

def create_model(model_name, num_classes=2, pretrained=True):
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "densenet201":
        model = models.densenet201(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "inception_v3":
        model = models.inception_v3(weights="IMAGENET1K_V1" if pretrained else None, aux_logits=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model
