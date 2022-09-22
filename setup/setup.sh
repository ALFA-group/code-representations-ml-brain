#!/usr/bin/env bash
##################################
# $1 is base_dir
##################################
base_dir=${1:-$(dirname $(pwd))}
bash _setup_inputs.sh $base_dir
bash _setup_code_seq2seq.sh $base_dir True False False False False False
bash _setup_code_transformer.sh $base_dir
