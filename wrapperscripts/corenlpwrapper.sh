#!/bin/env bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time 1-00:00:00

${1}/stanford-corenlp-4.5.0/corenlp.sh -fileList ${1}/${2}/${2}filelist -outputDirectory ${1}/${2}
