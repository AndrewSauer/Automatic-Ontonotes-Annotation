#!/bin/bash
#SBATCH -c 8
#SBATCH -p gpu,dgx,dgxs
#SBATCH -G 1
#SBATCH --mem 100000
#SBATCH --time 1-00:00:00
CUDA_DIR=/usr/local/eecsapps/cuda/cuda-10.0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/eecsapps/cuda/cuda-10.0
python3 onf2tensor.py
