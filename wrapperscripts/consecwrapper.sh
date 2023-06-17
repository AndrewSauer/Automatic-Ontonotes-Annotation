#!/bin/env bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time 1-00:00:00

cd ${1}/consec-main
python ${1}/consec-main/src/scripts/model/raganato_evaluate.py model.model_checkpoint=${1}/consec-main/experiments/released-ckpts/consec_wngt_best.ckpt test_raganato_path=${1}/${2}/
cd ..
