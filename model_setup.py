import torch
from torchvision import models
from torchvision.models import ResNet50_Weights

def setup_model(device, num_classes=100):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    return model
