import argparse
import os
import datetime
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append('/home/kakeya/Desktop/higuchi/20191107/src/Keras')
import numpy as np
import tensorflow as tf
from pathlib import Path
from callbacks import CometLogImageUploader, CustomizedLearningRateScheduler
from AdaBound import AdaBoundOptimizer
from dataset.dataset import Loader, Generator,SingleGenerator
from model.unet_3d import UNet3D
import yaml
from Loss.loss_funcs import categorical_crossentropy, bg_recall, bg_precision, bg_dice, \
    hcc_recall, hcc_precision, hcc_dice, \
    cyst_recall, cyst_precision, cyst_dice, \
    angioma_recall, angioma_precision, angioma_dice

# main.pyてきなやつ
# wpがあれば、ここからモデルの重みをロードする。
'''
python3 /home/kakeya/Desktop/higuchi/20191021/Keras/src/run_unet_3d_med.py -wp /home/kakeya/Desktop/higuchi/20191021/Keras/weights-e027_unet_liver_tumor_and_cyst_3cls.hdf5



python3 run_unet_3d_med.py -ex tutorial -g 1 -wp /home/kakeya/Desktop/higuchi/20191107/experiment/tutorial/2019-11-07_23-30/weights-e016_unet_liver_tumor_and_cyst_3cls.hdf5
python3 run_unet_3d_med.py -ex single_channel -g 0 -yml /home/kakeya/Desktop/higuchi/20191107/experiment/single_channel/setting.yml
python3 run_unet_3d_med.py -ex standard05 -g 0 -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/standard05/mini_setting.yml


'''
def ParseArgs():
    # from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gpu_number', default=1, type=int)
    parser.add_argument('-wp','--weight_path', type=str)
    parser.add_argument('-ex','--experiment', type=str)
    parser.add_argument('-yml','--setting_yml_path',type=str,default='/home/kakeya/Desktop/higuchi/20191107/experiment/sigle_channel/setting.yml')
    args = parser.parse_args()
    return args

def ConfigGpu(gpu_number):
    if gpu_number >= 0:
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(gpu_number)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        return config
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def ConstructCallback(model,WEIGHT_SAVE_DIR):
    now = datetime.datetime.now()

    import pathlib
    log_dir = pathlib.Path(f'{WEIGHT_SAVE_DIR}/{now.strftime("%Y-%m-%d_%H-%M")}')
    os.makedirs(str(log_dir), exist_ok=True)
    model_filename = log_dir / 'model.h5'
    weight_filename = log_dir / 'weights-e{epoch:03d}.hdf5'
    results_filename = log_dir / 'epoch_results.csv'

    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=str(
        weight_filename), save_best_only=False, save_weights_only=True))
    callbacks.append(tf.keras.callbacks.CSVLogger(str(results_filename)))
    # callbacks.append(CustomizedLearningRateScheduler(patience=10, decay=0.8))

    return callbacks


'''
1. setting GPU
2. load dataset tng,val
3. make Generator 
4. fit Generator
vscodeから実行するとおかしくなっちゃう
'''


def main(args):
    config = ConfigGpu(args.gpu_number)
    with open(args.setting_yml_path) as file:
        yml = yaml.load(file)
        ROOT_DIR = yml['DIR']['ROOT']
        DATA_DIR = yml['DIR']['DATA']
        WEIGHT_SAVE_DIR=f'{ROOT_DIR}/experiment/{args.experiment}'
        train_cid=yml['CID']['TRAIN']
        val_cid=yml['CID']['VAL']
        train_patch=yml['PATCH_DIR']['TRAIN']
        val_patch=yml['PATCH_DIR']['VAL']
        patch_shape=yml['PATCH_SHAPE']
        BATCH_SIZE = yml['BATCH_SIZE']
        BATCH_GENERATOR =eval(yml['GENERATOR']) if 'GENERATOR' in yml else Generator

    # return dataframe require:patch_npy
    loader = Loader(DATA_DIR, patch_dir_name=train_patch)
    train_dataset = loader.load_train(train_cid)

    loader = Loader(DATA_DIR, patch_dir_name=val_patch)
    valid_dataset = loader.load_valid(val_cid)

    def weight_method(X, Y, nclasses):
        _Y = np.argmax(Y, axis=4)
        counts = np.array([(_Y == label).sum() for label in range(nclasses)])
        counts = np.cbrt(np.where(counts != 0, counts, 1))
        Y = Y * 1e0 / counts
        return Y
    print(BATCH_GENERATOR,type(BATCH_GENERATOR))
    train_generator = BATCH_GENERATOR(train_dataset, batch_size=BATCH_SIZE, nclasses=4, enable_random_crop=True,
                                crop_size=(48, 48, 16), threshold=float('inf'), weight_method=weight_method)
    valid_generator = BATCH_GENERATOR(valid_dataset, batch_size=BATCH_SIZE, nclasses=4, enable_random_crop=False,
                                crop_size=(48, 48, 16), threshold=float('inf'), weight_method=weight_method)

    with tf.Session(config=config) as sess:
        #(self, input_shape, nclasses, use_bn=True, use_dropout=True)
        #num of channe2l is 2(SE2,SE3)
        model = UNet3D(patch_shape, 4)
        if args.weight_path!=None:
            if Path(args.weight_path).is_file():
                path=Path(args.weight_path)
                initial_epoch=int(path.name.split('-')[1][1:4])
                print('_'*30)
                print(f'load model weight from {args.weight_path}')
                model.load_weights(os.path.join(args.weight_path))
        else:
            print('_'*30)
            print('no load model weight')
            initial_epoch=0

        callbacks = ConstructCallback(model,WEIGHT_SAVE_DIR)
        # image_logger...? experiment can't find.
        # callbacks.append(CometLogImageUploader(experiment, train_generator, upload_steps=250))

        model.compile(loss={'segment': categorical_crossentropy},
                      loss_weights={'segment': 1.},
                      optimizer=AdaBoundOptimizer(learning_rate=1e-3, final_lr=1e-1),
                      metrics={'segment': [bg_dice, hcc_dice, cyst_dice,angioma_dice]})

        model.fit_generator(train_generator, steps_per_epoch=len(train_generator),initial_epoch=initial_epoch,
                            validation_data=valid_generator, validation_steps=len(valid_generator),
                            callbacks=callbacks, workers=6, max_queue_size=12, use_multiprocessing=True,
                            epochs=50, shuffle=False)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
