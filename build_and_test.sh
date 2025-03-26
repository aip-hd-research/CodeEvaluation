#!/bin/bash -i

OUTPUT_FOLDER="${1:-$PWD}"
ENV_NAME="CodeEvaluation_Build_And_Test"

conda create -y -n $ENV_NAME python=3.10 conda-build conda-verify
conda activate $ENV_NAME
mkdir -p $OUTPUT_FOLDER
conda-build . --keep-old-work -c conda-forge | tee $OUTPUT_FOLDER/test.log
CONDA_TEST_SRC_DIR=$(grep -oP '^export SRC_DIR=\K.*test_tmp$' $OUTPUT_FOLDER/test.log)
CONDA_PACKAGE=$(grep -oP '^TEST START: \K.*$' $OUTPUT_FOLDER/test.log)
cp $CONDA_TEST_SRC_DIR/environment.test.yaml $OUTPUT_FOLDER
cp $CONDA_PACKAGE $OUTPUT_FOLDER
conda deactivate
conda env remove -y -n $ENV_NAME
