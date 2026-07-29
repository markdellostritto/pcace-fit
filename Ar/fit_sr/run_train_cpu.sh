#!/bin/bash

# activate virtual environment
source ~/path_to_venv/bin/activate

# train
python3 fit-pcace-cpu.py ../data_Ar.xyz > run-train-0.out

# deactivate virtual environment
deactivate

