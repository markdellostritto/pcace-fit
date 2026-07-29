#!/bin/bash
grep 'INFO: train_e/atom_rmse' run-train-*.out | cut -d ' ' -f 5 > train_e_rmse.dat
grep 'INFO: val_e/atom_rmse' run-train-*.out | cut -d ' ' -f 5 > val_e_rmse.dat
grep 'INFO: train_f_rmse' run-train-*.out | cut -d ' ' -f 5 > train_f_rmse.dat
grep 'INFO: val_f_rmse' run-train-*.out | cut -d ' ' -f 5 > val_f_rmse.dat
grep 'Learning rate' run-train-*.out | grep 'INFO' | cut -d ' ' -f 9 > lr.dat
