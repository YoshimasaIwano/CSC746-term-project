import argparse
import tensorflow as tf
from data_preparation import get_train_loader
from model_setup import setup_model
from train import train_model

def main(batch_size, use_gpus, data_size_factor):
    print(f"Batch size: {batch_size}")
    print(f"Using {use_gpus} GPUs")
    print(f"Data size factor: {data_size_factor}")

    trainloader = get_train_loader(batch_size=batch_size, data_size_factor=data_size_factor)
    model = setup_model(use_gpus=use_gpus)
    train_model(model, trainloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--use_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--data_size_factor', type=int, default=10, help='Factor to scale data size')
    
    args = parser.parse_args()
    main(args.batch_size, args.use_gpus, args.data_size_factor)
