# data_preparation.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, DistributedSampler

def get_train_loader(batch_size=128, data_size_factor=10, rank=0, world_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    subset_size = len(full_trainset) // data_size_factor
    subset_indices = torch.randperm(len(full_trainset))[:subset_size].tolist()
    trainset = Subset(full_trainset, subset_indices)

    # Use DistributedSampler for distributing the dataset
    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    
    return trainloader
