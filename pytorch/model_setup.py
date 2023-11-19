# model_setup.py
import torch
import torchvision.models as models
from torchvision.models import ResNet152_Weights
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def setup_model(rank, world_size, num_classes=100):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    weights = ResNet152_Weights.DEFAULT
    model = models.resnet152(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    model = DistributedDataParallel(model, device_ids=[rank])

    return model


# import torch
# from model import CustomCNN

# def setup_model(device, num_classes=100, use_gpus=1):
#     model = CustomCNN()
#     model.to(device)

#     num_gpus = torch.cuda.device_count()
#     print(f"Number of GPUs available: {num_gpus}")

#     if num_gpus > 1 and use_gpus > 1:
#         model = torch.nn.DistributedDataParallel(model, device_ids=list(range(use_gpus)))

#     return model