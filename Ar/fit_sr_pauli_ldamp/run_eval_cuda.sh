#!/bin/bash

# activate virtual environment
source /path_to_venv/bin/activate

# train
python3 eval-pcace-cuda.py best_model.pth ../data_Ar.xyz

# deactivate virtual environment
deactivate

