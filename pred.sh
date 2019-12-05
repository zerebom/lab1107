#!/bin/bash
python3 preprocess_evaluate.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml
python3 evaluate_tsukuba_med.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml
python3 evaluate_plot.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml

python3 ./src/Keras/pred3D_wrap.py -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/normal64_clip/mini_setting.yml -g 0
python3 ./src/evaluate/preprocess_evaluate.py -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/normal64_clip/mini_setting.yml
python3 ./src/evaluate/evaluate_tsukuba_med.py -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/normal64_clip/mini_setting.yml
python3 ./src/evaluate/evaluate_plot.py -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/normal64_clip/mini_setting.yml
