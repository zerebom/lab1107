#!/bin/bash

# pred.sh /home/kakeya/Desktop/higuchi/20191107/experiment/hist_equal05/setting.yml
#python3 pred3D_wrap.py -yml $0

cd Keras
python3 pred3D_wrap.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/single_channel_HE05/setting.yml
cd ..
sudo python3 evaluate/preprocess_evaluate.py -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/normal64/mini_setting.yml
sudo python3 evaluate/evaluate_tsukuba_med.py -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/normal64/mini_setting.yml
sudo python3 evaluate/evaluate_plot.py -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/normal64/mini_setting.yml
