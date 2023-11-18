from device_setup import get_device
from data_preparation import get_train_loader
from model_setup import setup_model
from train import train_model

def main():
    batch_size = 64
    use_gpus = 2
    print(f"Batch size: {batch_size}")
    print(f"Using {use_gpus} GPUs")
    device = get_device()
    trainloader = get_train_loader(batch_size=batch_size)
    model = setup_model(device, use_gpus=use_gpus)
    train_model(model, trainloader, device)

if __name__ == "__main__":
    main()
