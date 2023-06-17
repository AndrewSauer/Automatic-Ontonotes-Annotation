#!/bin/env bash
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time 1-00:00:00
while true
do
    python ${1}/stanfordtoxml.py ${1}/${2}/
    python ${1}/stanford2json.py ${1}/${2}/
    python ${1}/output2onf.py ${1}/${2}/
    python ${1}/garbagecollect.py ${1}/${2}/
done
