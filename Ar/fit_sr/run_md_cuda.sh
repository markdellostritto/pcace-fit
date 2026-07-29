#!/bin/bash

# activate virtual environment
source /path_to_venv/bin/activate

# train
python3 md-pcace-cuda.py best_model.pth 30.0

# deactivate virtual environment
deactivate

