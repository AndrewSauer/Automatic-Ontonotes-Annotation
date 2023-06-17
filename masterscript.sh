#!/bin/env bash

#DEFUNCT: use parallelscript.sh instead
#usage: ./masterscript.sh <inputfile without extension(must be at base)>
base=~/hpc-share/personal/Thesis
#edit this to your base file, along with the line in consec-main/src/scripts/model/predict.py
$base/stanford-corenlp-4.5.0/corenlp.sh -file $base/${1}.txt #Run the stanford annotator on the input file
python $base/consec-main/stanfordtoxml.py $base/${1}.txt.out $base/consec-main/data/stanford/${1}.data.xml ${1}
echo lmao > $base/consec-main/data/stanford/${1}.gold.key.txt
cd consec-main
python $base/consec-main/src/scripts/model/raganato_evaluate.py model.model_checkpoint=$base/consec-main/experiments/released-ckpts/consec_wngt_best.ckpt test_raganato_path=$base/consec-main/data/stanford/${1}
cd ..
rm $base/consec-main/data/stanford/${1}.data.xml
rm $base/consec-main/data/stanford/${1}.gold.key.txt
mv $base/${1}.txt.out $base/master-output
mv $base/consec-main/data/stanford/${1}.wsd.out $base/master-output

#tenative code for srl
python $base/stanford2json.py $base/master-output/${1}.txt.out $base/master-output/${1}.json

#this part could be removed if wordnet senses are matched with propbank frames
#disambiguate predicate lemmas
python $base/mrc-srl/module/PredicateDisambiguation/predict.py \
--frames_path $base/mrc-srl/data/conll2012/frames.json \
--dataset_path $base/master-output/${1}.json \
--output_path $base/master-output/${1}.psense.json \
--checkpoint_path $base/mrc-srl/scripts/checkpoints/conll2012/disambiguation/2022_09_02_16_14_20/checkpoint_4.cpt \
--max_tokens 2048 \
--save \
--amp

#predict roles of arguments
python $base/mrc-srl/module/RolePrediction/predict.py \
--dataset_tag conll2012 \
--dataset_path $base/master-output/${1}.psense.json \
--output_path $base/master-output/${1}.psense.plabel.json \
--checkpoint_path $base/mrc-srl/scripts/checkpoints/conll2012/role_prediction/2022_09_03_20_04_39/checkpoint_7.cpt \
--max_tokens 2048 \
--alpha 5 \
--save \
--amp

#argument labelling
python $base/mrc-srl/module/ArgumentLabeling/ckpt_eval.py \
--data_path $base/master-output/${1}.psense.plabel.json \
--checkpoint_path $base/mrc-srl/scripts/checkpoints/conll2012/arg_labeling/2022_09_04_11_40_28/checkpoint_11.cpt \
--output_path $base/master-output/${1}.list \
--gold_level 1 \
--arg_query_type 2 \
--argm_query_type 1 \
--max_tokens 1024 \
--amp

rm $base/master-output/${1}.json
rm $base/master-output/${1}.psense.json
python $base/output2onf.py $base/master-output/${1}.txt.out $base/master-output/${1}.wsd.out $base/master-output/${1}.psense.plabel.json $base/master-output/${1}.list $base/master-output/${1}.onf
rm $base/master-output/${1}.txt.out
rm $base/master-output/${1}.wsd.out
rm $base/master-output/${1}.psense.plabel.json
rm $base/master-output/${1}.list
