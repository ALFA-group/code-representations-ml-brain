#!/usr/bin/env bash

set -e

#conda install -yc anaconda wget
conda install -yc conda-forge tar

NAME=code_transformer
CACHE_DIR=$1/braincode/.cache
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

cd $DATASET_DIR
TAR_NAME=python.tar.gz
wget -O $TAR_NAME https://www.dropbox.com/s/ukve7lu6t9d6kfu/python.tar.gz?dl=0
tar -xvzf $TAR_NAME
rm $TAR_NAME

cd $MODEL_DIR
TAR_NAME=csn-single-language-models.tar.gz
wget -O $TAR_NAME https://www.dropbox.com/s/s7yjr5yr8hxyfvj/csn-single-language-models.tar.gz?dl=0
tar -xvzf $TAR_NAME
rm $TAR_NAME
rm -r great_code_summarization/
rm -r ct_code_summarization/CT-[1234678]
rm -r xl_net_code_summarization/XL-[234]

cd $BINARY_DIR
TAR_NAME=semantic.tar.gz
wget -O $TAR_NAME https://www.dropbox.com/s/vxpcjs2myi8yych/semantic.tar.gz?dl=0
tar -xvzf $TAR_NAME
rm $TAR_NAME

cd $ENV_DIR
echo "
export CODE_TRANSFORMER_DATA_PATH=$DATASET_DIR
export CODE_TRANSFORMER_BINARY_PATH=$BINARY_DIR
export CODE_TRANSFORMER_MODELS_PATH=$MODEL_DIR
export CODE_TRANSFORMER_LOGS_PATH=$LOG_DIR

export CODE_TRANSFORMER_CSN_RAW_DATA_PATH=$DATASET_DIR/raw/csn
export CODE_TRANSFORMER_CODE2SEQ_RAW_DATA_PATH=$DATASET_DIR/raw/code2seq
export CODE_TRANSFORMER_CODE2SEQ_EXTRACTED_METHODS_DATA_PATH=$DATASET_DIR/raw/code2seq-methods

export CODE_TRANSFORMER_DATA_PATH_STAGE_1=$DATASET_DIR/stage1
export CODE_TRANSFORMER_DATA_PATH_STAGE_2=$DATASET_DIR/stage2

export CODE_TRANSFORMER_JAVA_EXECUTABLE=java
export CODE_TRANSFORMER_JAVA_PARSER_EXECUTABLE=$BINARY_DIR/java-parser-1.0-SNAPSHOT.jar
export CODE_TRANSFORMER_JAVA_METHOD_EXTRACTOR_EXECUTABLE=$BINARY_DIR/JavaMethodExtractor-1.0.0-SNAPSHOT.jar
export CODE_TRANSFORMER_SEMANTIC_EXECUTABLE=$BINARY_DIR/semantic
" > $ENV_DIR/.env
