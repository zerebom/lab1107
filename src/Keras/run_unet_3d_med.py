import argparse
import os
import datetime
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append('/home/kakeya/Desktop/higuchi/20191107/src/Keras')
from model.unet_3d import UNet3D
import numpy as np
import tensorflow as tf
from pathlib import Path
from callbacks import CometLogImageUploader, CustomizedLearningRateScheduler
from AdaBound import AdaBoundOptimizer
from dataset.dataset import Loader, Generator, SingleGenerator
import yaml
from Loss.loss_funcs import categorical_crossentropy, bg_recall, bg_precision, bg_dice, \
    hcc_recall, hcc_precision, hcc_dice, \
    cyst_recall, cyst_precision, cyst_dice, \
    angioma_recall, angioma_precision, angioma_dice
from utils import send_line_notification

# main.pyてきなやつ
# wpがあれば、ここからモデルの重みをロードする。
'''
python3 /home/kakeya/Desktop/higuchi/20191021/Keras/src/run_unet_3d_med.py -wp /home/kakeya/Desktop/higuchi/20191021/Keras/weights-e027_unet_liver_tumor_and_cyst_3cls.hdf5



python3 run_unet_3d_med.py -ex tutorial -g 1 -wp /home/kakeya/Desktop/higuchi/20191107/experiment/tutorial/2019-11-07_23-30/weights-e016_unet_liver_tumor_and_cyst_3cls.hdf5
python3 run_unet_3d_med.py -ex single_channel -g 0 -yml /home/kakeya/Desktop/higuchi/20191107/experiment/single_channel/setting.yml
python3 run_unet_3d_med.py  -g 0 -yml /home/higuchi/Desktop/higuchi/lab1107/experiment/standard05/mini_setting.yml
c/home/kakeya/Desktop/higuchi/20191107/experiment/lr01_epoch50_100/setting.yml

'''


def ParseArgs():
    # from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_number', default=1, type=int)
    parser.add_argument('-wp', '--weight_path', type=str)
    parser.add_argument('-ex', '--experiment', type=str)
    parser.add_argument('-yml', '--setting_yml_path', type=str,
                        default='/home/kakeya/Desktop/higuchi/20191107/experiment/sigle_channel/setting.yml')
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


def ConstructCallback(model, WEIGHT_SAVE_DIR):
    now = datetime.datetime.now()

    import pathlib
    log_dir = pathlib.Path(f'{WEIGHT_SAVE_DIR}/{now.strftime("%Y-%m-%d_%H-%M")}')
    os.makedirs(str(log_dir), exist_ok=True)
    weight_filename = log_dir / 'weights-e{epoch:03d}.hdf5'
    results_filename = log_dir / 'epoch_results.csv'

    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=str(
        weight_filename), save_best_only=True, save_weights_only=True))
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

def split_train_val(fold=0):
    train=[['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '016', '017', '018', '019', '025', '028', '029', '030', '031', '032', '039'], ['014', '015', '021', '022', '023', '024', '026', '033', '034', '035', '036', '037', '038', '040', '041', '044', '045', '046', '047', '051', '055', '057', '064', '065'], ['053', '058', '060', '061', '062', '063', '066', '067', '068', '069', '071', '072', '073', '074', '076', '077', '078', '079', '083', '086', '093', '094'], ['082', '088', '090', '095', '096', '097', '098', '101', '102', '103', '104', '105', '107', '109', '112', '113', '117', '118', '122', '123', '125', '130']]
    valid=train.pop(fold)
    #flatten
    train=sum(train,[])
    return train,valid

def main(args):
    config = ConfigGpu(args.gpu_number)
    with open(args.setting_yml_path) as file:
        yml = yaml.load(file)
        ROOT_DIR = yml['DIR']['ROOT']
        DATA_DIR = yml['DIR']['DATA']
        WEIGHT_SAVE_DIR = yml['DIR']['OWN']
        WEIGHT_PATH=yml['PRED_WEIGHT'] if 'PRED_WEIGHT' in yml else None
        train_cid = yml['CID']['TRAIN'] if not 'FOLD' in yml else split_train_val(yml['FOLD'])[0]
        val_cid = yml['CID']['VAL'] if not 'FOLD' in yml else split_train_val(yml['FOLD'])[1]
        train_patch = yml['PATCH_DIR']['TRAIN']
        val_patch = yml['PATCH_DIR']['VAL']
        patch_shape = yml['PATCH_SHAPE']
        epochs =yml['EPOCH'] if 'EPOCH' in yml else 50
        lr = yml['LR'] if 'LR' in yml else 1e-3
        final_lr =yml['FINAL_LR'] if 'FINAL_LR' in yml else 1e-1
        BATCH_SIZE = yml['BATCH_SIZE']
        BATCH_GENERATOR = eval(yml['GENERATOR']) if 'GENERATOR' in yml else Generator

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
    print(BATCH_GENERATOR, type(BATCH_GENERATOR))
    train_generator = BATCH_GENERATOR(train_dataset, batch_size=BATCH_SIZE, nclasses=4, enable_random_crop=True,
                                      crop_size=(48, 48, 16), threshold=float('inf'), weight_method=weight_method)
    valid_generator = BATCH_GENERATOR(valid_dataset, batch_size=BATCH_SIZE, nclasses=4, enable_random_crop=False,
                                      crop_size=(48, 48, 16), threshold=float('inf'), weight_method=weight_method)

    with tf.Session(config=config) as sess:
        # (self, input_shape, nclasses, use_bn=True, use_dropout=True)
        # num of channe2l is 2(SE2,SE3)
        model = UNet3D(patch_shape, 4)
        if WEIGHT_PATH != None:
            if Path(WEIGHT_PATH).is_file():
                path = Path(WEIGHT_PATH)
                initial_epoch = int(path.name.split('-')[1][1:4])
                print('_' * 30)
                print(f'load model weight from {WEIGHT_PATH}')
                model.load_weights(os.path.join(WEIGHT_PATH))
        else:
            print('_' * 30)
            print('no load model weight')
            initial_epoch = 0

        callbacks = ConstructCallback(model, WEIGHT_SAVE_DIR)
        # image_logger...? experiment can't find.
        # callbacks.append(CometLogImageUploader(experiment, train_generator, upload_steps=250))

        model.compile(loss={'segment': categorical_crossentropy},
                      loss_weights={'segment': 1.},
                      optimizer=AdaBoundOptimizer(learning_rate=lr, final_lr=final_lr),
                      metrics={'segment': [bg_dice, hcc_dice, cyst_dice, angioma_dice]})

        model.fit_generator(train_generator, steps_per_epoch=len(train_generator), initial_epoch=initial_epoch,
                            validation_data=valid_generator, validation_steps=len(valid_generator),
                            callbacks=callbacks, workers=6, max_queue_size=12, use_multiprocessing=True,
                            epochs=epochs, shuffle=False)

        send_line_notification('finish:',args.setting_yml_path)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
