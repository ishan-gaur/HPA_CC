#!/usr/bin/env bash
ENV_NAME="hpa-cc-copy"

# Install the actual HPA_CC package
cd src && conda run -n $ENV_NAME pip install -e . && cd ..
conda run -n $ENV_NAME pip install --upgrade --upgrade-strategy only-if-needed install microfilm
conda run -n $ENV_NAME pip install --upgrade --upgrade-strategy only-if-needed install opencv-python

# Installing the HPA-Cell-Segmentation package
cd src/HPA-Cell-Segmentation
conda run -n $ENV_NAME pip install 'git+https://github.com/haoxusci/pytorch_zoo@master#egg=pytorch_zoo'
conda run -n $ENV_NAME pip install -e . --upgrade --upgrade-strategy only-if-needed
cd ../..

# JAX Tools
conda run -n $ENV_NAME pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda run -n $ENV_NAME pip install ott-jax

# Misc
conda run -n $ENV_NAME pip install wandb