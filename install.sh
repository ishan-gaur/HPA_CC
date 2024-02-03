#!/usr/bin/env bash
cd src && conda run -n hpa-cc pip install -e .
conda run -n hpa-cc pip install --upgrade --upgrade-strategy only-if-needed install microfilm
conda run -n hpa-cc pip install --upgrade --upgrade-strategy only-if-needed install opencv-python
conda run -n hpa-cc pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda run -n hpa-cc pip install ott-jax

