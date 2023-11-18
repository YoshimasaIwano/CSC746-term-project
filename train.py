import torch.optim as optim
import torch
from torch.profiler import profile, ProfilerActivity

def train_model(model, trainloader, device, epochs=1):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Starting training...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        for epoch in range(epochs): 
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

    print("Training complete. Printing profiler results...")
    # prof.export_chrome_trace("trace.json")  # Export the trace to a file

    # Summarize and print total times
    total_cuda_time = sum([k.cuda_time_total for k in prof.key_averages()])
    total_cpu_time = sum([k.cpu_time_total for k in prof.key_averages()])
    print(f"Total CUDA time: {total_cuda_time / 1e6} s")
    print(f"Total CPU time: {total_cpu_time / 1e6} s")

    # Print the table of the most significant operations
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))


