#!/bin/bash
#SBATCH --job-name=gpu_info
#SBATCH --partition=cola02
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/salvadordeharoo/IDL/practica1/slurm_outputs/info_gpu_%j.out

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
