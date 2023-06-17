#!/bin/env bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time 1-00:00:00

python ${1}/mrc-srl/module/PredicateDisambiguation/predict.py \
--frames_path ${1}/mrc-srl/data/conll2012/frames.json \
--dataset_path ${1}/${2}/ \
--checkpoint_path ${1}/mrc-srl/scripts/checkpoints/conll2012/disambiguation/2022_09_02_16_14_20/checkpoint_4.cpt \
--max_tokens 2048 \
--save \
--amp
