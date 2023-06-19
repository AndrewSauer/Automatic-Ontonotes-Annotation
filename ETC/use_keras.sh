#!/bin/bash
#SBATCH -c 8
#SBATCH -p dgx,dgxs
#SBATCH -G 1
#SBATCH --mem 100000
#SBATCH --time 1-00:00:00
CUDA_DIR=/usr/local/eecsapps/cuda/cuda-10.0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/eecsapps/cuda/cuda-10.0
python3 use_keras.py \
--TOKENIZER="BERT" \
--ANNOTATE=True \
--FULL=True \
--TRUEFULL=True \
--IGN_PATH="./ign_mask.pickle" \
--GLOBAL_SIZE=80 \
--NUM_ENTITIES=50 \
--NUM_LABELS=96 \
--BATCH_SIZE=1 \
--ONE_LABEL=-1 \
--NUM_HIDDEN_LAYERS=12 \
--LEARNING_RATE=0.00002 \
--NUM_EPOCHS=15 \
--ZERO_WEIGHT=0.25 \
--DO_MASK=True \
--LOSS_FUNC="AFL" \
--NUM_FINAL_LAYERS=5 \
--GAMMA=1.0 \
--PRETRAIN_FILE="etcmodel/pretrained/etc_base/model.ckpt" \
--PRETRAIN=True \
--OPTIMIZER="Adam" \
--CONTINUE=False \
--RUN_NUMBER=-1
