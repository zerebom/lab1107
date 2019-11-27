#!/bin/bash
python3 preprocess_evaluate.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml
python3 evaluate_tsukuba_med.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml
python3 evaluate_plot.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml

python3 preprocess_evaluate.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/KLD/setting.yml
python3 evaluate_tsukuba_med.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/KLD/setting.yml
python3 evaluate_plot.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/KLD/setting.yml
