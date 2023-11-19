import torch
from model import CustomCNN

def setup_model(device, num_classes=100, use_gpus=1):
    model = CustomCNN()
    model.to(device)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    if num_gpus > 1 and use_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(use_gpus)))

    return model

