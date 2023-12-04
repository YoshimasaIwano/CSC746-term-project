#!/bin/bash

# Loop over each batch size and GPU count combination
for data_size_factor in 10 2 1; do
    for batch_size in 128 512 1024 2048; do
        for gpu_count in 1 2 4; do
            echo "Running with batch size $batch_size, $gpu_count GPUs, and data size factor $data_size_factor"

            # Use the PyTorch distributed launcher
            export OMP_NUM_THREADS=$gpu_count
            torchrun --nproc_per_node=$gpu_count main.py --batch_size $batch_size --data_size_factor $data_size_factor
        done
    done
done
