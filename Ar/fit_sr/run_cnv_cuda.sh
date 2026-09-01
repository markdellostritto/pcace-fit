#!/bin/bash

# activate virtual environment
source /path_to_venv/bin/activate

# convert
python3 make_mliap.py best_model.pth

# deactivate virtual environment
deactivate


