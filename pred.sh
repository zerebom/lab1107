#!/bin/bash
YAML=$1
GPU=$2

python3 ./src/set_hdf5.py -yml $YAML 
python3 ./src/Keras/pred3D_wrap.py -yml $YAML -g $GPU
python3 ./src/evaluate/preprocess_evaluate.py -yml $YAML
python3 ./src/evaluate/evaluate_tsukuba_med.py -yml $YAML
python3 ./src/evaluate/evaluate_plot.py -yml $YAML
