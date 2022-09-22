#!/usr/bin/env bash

set -e

conda install -yc anaconda wget
conda install -yc conda-forge unzip

HOME_DIR=$1/braincode/
cd $HOME_DIR

ZIP_NAME=inputs.zip
wget -O $ZIP_NAME https://www.dropbox.com/s/hxnxdhygk9b05rj/inputs.zip?dl=0
unzip $ZIP_NAME -d inputs
rm $ZIP_NAME

# Process and populate benchmark metrics on input files
python -m braincode.utils $HOME_DIR 2
