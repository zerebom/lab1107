#!/bin/bash
python3 Keras/pred3D_wrap.py -yml $0
python3 evaluate/preprocess_evaluate.py -yml $0
python3 evaluate/evaluate_tsukuba_med.py -yml $0