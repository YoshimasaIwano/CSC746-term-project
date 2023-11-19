# main.py
import argparse
import os
import torch.distributed as dist
from data_preparation import get_train_loader
from model_setup import setup_model
from train import train_model

def main(batch_size, data_size_factor, rank, world_size):
    print(f"Batch size: {batch_size}, Rank: {rank}, World Size: {world_size}")
    print(f"Data size factor: {data_size_factor}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    trainloader = get_train_loader(batch_size=batch_size, data_size_factor=data_size_factor, rank=rank, world_size=world_size)
    model = setup_model(rank, world_size)
    train_model(model, trainloader, rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--data_size_factor', type=int, default=10, help='Factor to scale data size')
    
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    main(args.batch_size, args.data_size_factor, rank, world_size)


