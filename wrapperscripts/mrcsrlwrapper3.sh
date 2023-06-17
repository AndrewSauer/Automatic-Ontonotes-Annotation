#!/bin/env bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time 1-00:00:00

python ${1}/mrc-srl/module/ArgumentLabeling/ckpt_eval.py \
--data_path ${1}/${2}/ \
--checkpoint_path ${1}/mrc-srl/scripts/checkpoints/conll2012/arg_labeling/2022_09_04_11_40_28/checkpoint_11.cpt \
--gold_level 1 \
--arg_query_type 2 \
--argm_query_type 1 \
--max_tokens 1024 \
--amp
