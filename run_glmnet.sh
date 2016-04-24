#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_exp.sh <dataset>"
    exit 1
fi

green=`tput setaf 2`
reset=`tput sgr0`

dataset=$1

TOPK=20

MIN_SUP=10
MAX_DEPTH=6

RANDOM_FEATURES=4
RANDOM_POSITIONS=8


echo ${green}===Compiling===${reset}

make > /dev/null

mkdir -p tmp
mkdir -p tmp/${dataset}

echo ${green}===DPClass Started===${reset}

echo ${green}===Current step: Producing Candidate Discriminative Patterns===${reset}
./bin/produce_candidate_patterns data/${dataset}/train.csv TRAIN tmp/${dataset}/tree_model tmp/${dataset}/rule_set ${MIN_SUP} ${MAX_DEPTH} ${RANDOM_FEATURES} ${RANDOM_POSITIONS}

echo ${green}===Current step: Selecting Top-K Discriminative Patterns By GLMNET===${reset}
cd glmnet_matlab
../bin/select_patterns_by_GLMNET ../data/${dataset}/train.csv ../tmp/${dataset}/rule_set ${TOPK} ../tmp/${dataset}/glmnet_top_${TOPK}_rules_set
cd ..

echo ${green}===Current step: Rebuilding Feature Tables===${reset}
./bin/rebuild_features data/${dataset}/train.csv tmp/${dataset}/glmnet_top_${TOPK}_rules_set tmp/${dataset}/train.glmnet_top_${TOPK}.csv ${TOPK}
./bin/rebuild_features data/${dataset}/test.csv tmp/${dataset}/glmnet_top_${TOPK}_rules_set tmp/${dataset}/test.glmnet_top_${TOPK}.csv ${TOPK}

echo ${green}===Current step: Predicting and Evaluating Test Data===${reset}
cd glmnet_matlab
../bin/predict_by_glmnet ../tmp/${dataset}/train.glmnet_top_${TOPK}.csv ../tmp/${dataset}/test.glmnet_top_${TOPK}.csv ../tmp/${dataset}/pred.glmnet_top_${TOPK}.csv
cd ..

echo ${green}===Done===${reset}