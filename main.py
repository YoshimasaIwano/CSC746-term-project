from device_setup import get_device
from data_preparation import get_train_loader
from model_setup import setup_model
from train import train_model

def main():
    device = get_device()
    trainloader = get_train_loader(batch_size=64, num_workers=2)
    model = setup_model(device)
    train_model(model, trainloader, device)

if __name__ == "__main__":
    main()