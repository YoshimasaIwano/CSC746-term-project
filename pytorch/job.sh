#!/bin/bash

# Loop over each batch size and GPU count combination
for batch_size in 64 128 512 1024
do
    for gpu_count in 1 2 4
    do
        for data_size_factor in 10
        do
            echo "Running with batch size $batch_size, $gpu_count GPUs, and data size factor $data_size_factor"
            python main.py --batch_size $batch_size --use_gpus $gpu_count --data_size_factor $data_size_factor
        done
    done
done
