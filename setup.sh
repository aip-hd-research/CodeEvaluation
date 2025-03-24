#!/bin/bash -i

ENV_NAME="${1:-"CodeEvaluation"}"
conda env create -y -n $ENV_NAME -f environment.dev.yaml
conda activate $ENV_NAME
chmod +x build_and_test.sh
./build_and_test.sh
conda env update -n $ENV_NAME -f environment.test.yaml
pre-commit install
conda deactivate
