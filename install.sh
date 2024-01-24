#!/usr/bin/env bash
cd src && conda run -n hpa-cc pip install -e .
conda run -n hpa-cc pip install --upgrade --upgrade-strategy only-if-needed install microfilm
conda run -n hpa-cc pip install --upgrade --upgrade-strategy only-if-needed install opencv-python

