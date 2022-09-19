#!/usr/bin/env bash

set -e

# install code_seq2seq package
pip install -e .

# conda install -yc anaconda wget
# conda install -yc conda-forge tar
ROOT_PATH=$1
DOWNLOAD_MODEL=$2
DOWNLOAD_DATA=$3
SPLIT_DATA=$4
TOKENIZE_DATA=$5
RUN_MODEL=$6
DEBUG_TRAINING=$7

TRAIN_FILES_NAME="train_files.txt"
TEST_FILES_NAME="test_files.txt"
TRAIN_FILES_TOK_NAME="train_files_tok.tsv"
TEST_FILES_TOK_NAME="test_files_tok.tsv"
MODEL_NAME="code_seq2seq_py8kcodenet"
SAVED_MODEL_NAME="$MODEL_NAME.torch"
SAVED_VOCAB_NAME="vocab_$MODEL_NAME.pkl"

NAME=code_seq2seq
CACHE_DIR=$ROOT_PATH/braincode/.cache
DATASET_DIR=$CACHE_DIR/datasets/$NAME
MODEL_DIR=$CACHE_DIR/models/$NAME
BINARY_DIR=$CACHE_DIR/bin/$NAME
LOG_DIR=$CACHE_DIR/logs/$NAME
ENV_DIR=$HOME/.config/$NAME

mkdir -p $DATASET_DIR
mkdir -p $MODEL_DIR
mkdir -p $BINARY_DIR
mkdir -p $LOG_DIR
mkdir -p $ENV_DIR

TAR_NAME=codenet-python

if [ $DOWNLOAD_MODEL == "True" ]; then
    echo "dowload saved model"
    cd $MODEL_DIR
    wget -O $SAVED_MODEL_NAME https://www.dropbox.com/s/15luknaba42bmax/code_seq2seq_py8kcodenet.torch?dl=0
    wget -O $SAVED_VOCAB_NAME https://www.dropbox.com/s/f315rhn71hqhjcp/vocab_code_seq2seq_py8kcodenet.pkl?dl=0
else
    if [ $DOWNLOAD_DATA == "True" ]; then
        cd $DATASET_DIR
        wget -O $TAR_NAME.tar.gz https://www.dropbox.com/s/zi9i8ictstsh9bx/codenet-python.tar.gz?dl=0
        mkdir $TAR_NAME
        tar -xvzf $TAR_NAME.tar.gz -C $TAR_NAME --strip-components 1
        rm $TAR_NAME.tar.gz
    fi

    if [ $SPLIT_DATA == "True" ]; then
        python -m code_seq2seq.split_data $DATASET_DIR/$TAR_NAME $DATASET_DIR/$TRAIN_FILES_NAME $DATASET_DIR/$TEST_FILES_NAME .py
    fi

    if [ $TOKENIZE_DATA == "True" ]; then
        python -m code_seq2seq.tokenize $DEBUG_TRAINING $DATASET_DIR/$TRAIN_FILES_NAME $DATASET_DIR/$TEST_FILES_NAME $DATASET_DIR/$TRAIN_FILES_TOK_NAME $DATASET_DIR/$TEST_FILES_TOK_NAME
    fi

    if [ $RUN_MODEL == "True" ]; then
        python -m code_seq2seq.train \
        --train_path $DATASET_DIR/$TRAIN_FILES_TOK_NAME \
        --dev_path $DATASET_DIR/$TEST_FILES_TOK_NAME \
        --expt_dir $MODEL_DIR \
        --save_model_as $SAVED_MODEL_NAME \
        --save_vocab_as $SAVED_VOCAB_NAME
    fi

fi

cd $ENV_DIR
echo "
export CODE_SEQ2SEQ_DATA_PATH=$DATASET_DIR
export CODE_SEQ2SEQ_MODELS_PATH=$MODEL_DIR
export CODE_SEQ2SEQ_LOGS_PATH=$LOG_DIR
export CODE_SEQ2SEQ_SAVED_MODEL_NAME=$SAVED_MODEL_NAME
" > $ENV_DIR/.env
