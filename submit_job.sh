#!/bin/bash

#SBATCH --job-name=pytorch_gpu_job     # Job name
#SBATCH --partition=regular            # Partition name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --gpus=4                       # Number of GPUs
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --output=job_output_%j.txt     # Standard output and error log (%j expands to jobId)
#SBATCH --account=yoshi03   # Replace with your account name

# Load necessary modules
module load pytorch/2.0.1  # Replace with the specific version available

# Execute the Python script
srun python main.py
