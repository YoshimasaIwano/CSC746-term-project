import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
# import numpy as np

# def get_train_loader(batch_size=64, data_size_factor=1):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     # Load the full CIFAR-100 dataset
#     full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

#     # Calculate the size of the subset using the data_size_factor
#     subset_size = len(full_trainset) // data_size_factor

#     # Generate random indices for the subset
#     subset_indices = torch.randperm(len(full_trainset))[:subset_size].tolist()

#     # Create a subset of the full dataset using the selected indices
#     trainset = Subset(full_trainset, subset_indices)

#     # Create DataLoader for the subset
#     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
#     return trainloader

from torchvision.datasets import ImageNet

def get_train_loader(batch_size=64, data_size_factor=1):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_trainset = ImageNet(root='./data', split='train', transform=transform)
    subset_size = len(full_trainset) // data_size_factor
    subset_indices = torch.randperm(len(full_trainset))[:subset_size].tolist()
    trainset = Subset(full_trainset, subset_indices)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    return trainloader
