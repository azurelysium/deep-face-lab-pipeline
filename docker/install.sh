#!/bin/bash -x

conda activate deepfacelab

git clone --depth 1 https://github.com/azurelysium/DeepFaceLab_Linux.git
cd DeepFaceLab_Linux
git clone --depth 1 https://github.com/iperov/DeepFaceLab.git
python -m pip install -r ./DeepFaceLab/requirements-cuda.txt
