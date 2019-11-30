#!/bin/bash
python3 preprocess_evaluate.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml
python3 evaluate_tsukuba_med.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml
python3 evaluate_plot.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/normal/setting.yml

python3 ./src/Keras/pred3D_wrap.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/KLD_inkid/setting.yml -g 1
python3 ./src/evaluate/preprocess_evaluate.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/KLD_inkid/setting.yml
python3 ./src/evaluate/evaluate_tsukuba_med.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/KLD_inkid/setting.yml
python3 ./src/evaluate/evaluate_plot.py -yml /home/kakeya/Desktop/higuchi/20191107/experiment/KLD_inkid/setting.yml
