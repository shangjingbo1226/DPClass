#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_forward.sh <dataset> <l2-norm-coefficient>"
	echo "Example: ./run_forward.sh adult 0.5"
    exit 1
fi

green=`tput setaf 2`
reset=`tput sgr0`

dataset=$1

TOPK=20
NTHREADS=20

MIN_SUP=10
MAX_DEPTH=6

RANDOM_FEATURES=4
RANDOM_POSITIONS=8

SMALL_ROUNDS=40
LARGE_ROUNDS=100
LAMBDA=$2

echo ${green}===Compiling===${reset}

make > /dev/null

mkdir -p tmp
mkdir -p tmp/${dataset}

echo ${green}===DPClass Started===${reset}

echo ${green}===Current step: Producing Candidate Discriminative Patterns===${reset}
./bin/produce_candidate_patterns data/${dataset}/train.csv TRAIN tmp/${dataset}/tree_model tmp/${dataset}/rule_set ${MIN_SUP} ${MAX_DEPTH} ${RANDOM_FEATURES} ${RANDOM_POSITIONS}

echo ${green}===Current step: Selecting Top-K Discriminative Patterns By Forward Selection===${reset}
./bin/merge_patterns data/${dataset}/train.csv tmp/${dataset}/rule_set ${TOPK} tmp/${dataset}/top_${TOPK}_rules_set ${NTHREADS} ${SMALL_ROUNDS} ${LARGE_ROUNDS} ${LAMBDA}

echo ${green}===Current step: Rebuilding Feature Tables===${reset}
./bin/rebuild_features data/${dataset}/train.csv tmp/${dataset}/top_${TOPK}_rules_set tmp/${dataset}/train.top_${TOPK}.csv ${TOPK}
./bin/rebuild_features data/${dataset}/test.csv tmp/${dataset}/top_${TOPK}_rules_set tmp/${dataset}/test.top_${TOPK}.csv ${TOPK}

echo ${green}===Current step: Predicting and Evaluating Test Data===${reset}
./bin/predict_by_svm tmp/${dataset}/train.top_${TOPK}.csv tmp/${dataset}/test.top_${TOPK}.csv ${NTHREADS} ${LARGE_ROUNDS} ${LAMBDA}


echo ${green}===Done===${reset}