#!/bin/env bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time 1-00:00:00

python ${1}/stanford2json.py ${1}/${2}/
