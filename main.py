import argparse
from device_setup import get_device
from data_preparation import get_train_loader
from model_setup import setup_model
from train import train_model

def main(batch_size, use_gpus, data_size_factor):
    print(f"Batch size: {batch_size}")
    print(f"Using {use_gpus} GPUs")
    print(f"Data size factor: {data_size_factor}")
    device = get_device()
    trainloader = get_train_loader(batch_size=batch_size, data_size_factor=data_size_factor)
    model = setup_model(device, use_gpus=use_gpus)
    train_model(model, trainloader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--use_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--data_size_factor', type=float, default=1.0, help='Factor to scale data size')
    
    args = parser.parse_args()
    main(args.batch_size, args.use_gpus, args.data_size_factor)

