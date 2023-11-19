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

    # Get aggregated profiling data once
    aggregated_profiler_data = prof.key_averages()

    # Summarize and print total times
    total_cuda_time = sum([k.cuda_time_total for k in aggregated_profiler_data])
    total_cpu_time = sum([k.cpu_time_total for k in aggregated_profiler_data])
    total_operations = sum([k.count for k in aggregated_profiler_data])
    print(f"Total CUDA time: {total_cuda_time}")
    print(f"Total CPU time: {total_cpu_time}")
    print(f"Total operations: {total_operations}")

    # Print the table of the most significant operations
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))

    # Initialize a dictionary to store max values for each key
    max_values = {key: 0 for key in [
        "cpu_memory_usage", "cuda_memory_usage", 
    ]}

    for key in max_values:
        print(f"Max of {key}: {aggregated_profiler_data.table(sort_by=key, row_limit=1)}")
    

# GPU memory: 40 GB (https://docs.nersc.gov/systems/perlmutter/architecture/)
# CPU memory: 256 GB (https://docs.nersc.gov/systems/perlmutter/architecture/)