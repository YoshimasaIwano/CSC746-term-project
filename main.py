import argparse
from device_setup import get_device
from data_preparation import get_train_loader
from model_setup import setup_model
from train import train_model

def main(batch_size, use_gpus):
    print(f"Batch size: {batch_size}")
    print(f"Using {use_gpus} GPUs")
    device = get_device()
    trainloader = get_train_loader(batch_size=batch_size)
    model = setup_model(device, use_gpus=use_gpus)
    train_model(model, trainloader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--use_gpus', type=int, default=1, help='Number of GPUs to use')
    
    args = parser.parse_args()
    main(args.batch_size, args.use_gpus)

