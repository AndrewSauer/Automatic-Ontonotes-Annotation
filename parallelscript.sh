#!/bin/env bash

#This script should take in a directory (single name only, no path), and then setup all the models on separate threads, telling them to process data as it becomes available in the pipeline
#All .txt files in the directory will be processed, and .onf files placed in the same directory.
base=~/hpc-share/personal/Thesis

python3 preprocess.py $base/${1}/

#do a garbagecollect pass first, so that the filelist only contains .txt files without corresponding .onf
python3 garbagecollect.py $base/${1}/

#copy frames.json to output directory
cp $base/mrc-srl/data/conll2012/frames.json $base/${1}

#list of .txt file paths in the given directory
find $base/${1} -maxdepth 1 | grep '\.txt$' > $base/${1}/${1}filelist

#start up corenlp in the background
sbatch --nice=0 $base/wrapperscripts/corenlpwrapper.sh $base ${1}

#start up conversion tools, this includes xml, json, and onf conversion
sbatch --nice=1 $base/wrapperscripts/convertwrapper.sh $base ${1}

#start up consec model in the background
sbatch --nice=2 $base/wrapperscripts/consecwrapper.sh $base ${1}

#start up mrc-srl in the background
sbatch --nice=3 $base/wrapperscripts/mrcsrlwrapper1.sh $base ${1}
sbatch --nice=4 $base/wrapperscripts/mrcsrlwrapper2.sh $base ${1}
sbatch --nice=5 $base/wrapperscripts/mrcsrlwrapper3.sh $base ${1}
