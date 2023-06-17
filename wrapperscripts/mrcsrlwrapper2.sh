#!/bin/env bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time 1-00:00:00

python ${1}/mrc-srl/module/RolePrediction/predict.py \
--dataset_tag conll2012 \
--dataset_path ${1}/${2}/ \
--checkpoint_path ${1}/mrc-srl/scripts/checkpoints/conll2012/role_prediction/2022_09_03_20_04_39/checkpoint_7.cpt \
--max_tokens 2048 \
--alpha 5 \
--save \
--amp
