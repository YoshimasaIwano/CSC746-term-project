#!/bin/bash

# Loop over each batch size and GPU count combination
for batch_size in 64 128 512 1024
do
    for gpu_count in 1 2 4
    do
        echo "Running with batch size $batch_size and $gpu_count GPUs"
        python main.py --batch_size $batch_size --use_gpus $gpu_count
    done
done

