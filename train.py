import torch.optim as optim
import torch
from torch.profiler import profile, ProfilerActivity

def train_model(model, trainloader, device, epochs=1):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Starting training...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, group_by_input_shape=True) as prof:
        for epoch in range(epochs): 
            # print(f"Starting epoch {epoch + 1}/{epochs}")
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # print(f"  Training batch {i + 1}")
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # if i % 2000 == 1999: 
                #     # print(f'  [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                #     running_loss = 0.0

            # print(f"Completed epoch {epoch + 1}/{epochs}")

    print("Training complete. Printing profiler results...")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))


