import argparse
import os
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from pathlib import Path
from callbacks import CometLogImageUploader, CustomizedLearningRateScheduler

from AdaBound import AdaBoundOptimizer
from dataset import Loader, Generator
from unet_3d import UNet3D
from loss_funcs import categorical_crossentropy, bg_recall, bg_precision, bg_dice, \
    hcc_recall, hcc_precision, hcc_dice, \
    cyst_recall, cyst_precision, cyst_dice, \
    angioma_recall, angioma_precision, angioma_dice

# main.pyてきなやつ
'''
python3 /home/kakeya/Desktop/higuchi/20191021/Keras/src/run_unet_3d_med.py -wp /home/kakeya/Desktop/higuchi/20191021/Keras/weights-e027_unet_liver_tumor_and_cyst_3cls.hdf5
python3 run_unet_3d_med.py  -wp /home/kakeya/Desktop/higuchi/20191021/Keras/src/logs/2019-10-24_18-28/weights-e023_unet_liver_tumor_and_cyst_3cls.hdf5
'''
WEIGHT_SAVE_DIR='/home/kakeya/Desktop/higuchi/20191021/Keras/src/weights/'

def ParseArgs():
    # from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_number', default=1, type=int)
    parser.add_argument('-wp','--weight_path', type=str)

    args = parser.parse_args()
    return args

# GPU on off setting


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

# make save_dir & append callbacks


def ConstructCallback(model):
    now = datetime.datetime.now()

    import pathlib
    log_dir = pathlib.Path(f'{WEIGHT_SAVE_DIR}/{now.strftime("%Y-%m-%d_%H-%M")}')
    os.makedirs(str(log_dir), exist_ok=True)
    model_filename = log_dir / 'model.h5'
    weight_filename = log_dir / 'weights-e{epoch:03d}_unet_liver_tumor_and_cyst_3cls.hdf5'
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
    
    DATA_DIR = '/home/kakeya/Desktop/higuchi/data'
    # DATA_DIR = '/media/higuchi/Windows/NN_data/kits19/data'
    # DATA_DIR = '/home/kakeya/Desktop/higuchi/DNP/data'
    BATCH_SIZE = 16

    loader = Loader(DATA_DIR, patch_dir_name='tumor_60x60x20')
    # return dataframe require:patch_npy
    train_dataset = loader.load_train([
    '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '016', '017', '018', '019', '025', '028', '029', '030', '031', '032', '039','014', '015', '021', '022', '023', '024', '026', '033', '034', '035', '036', '037', '038', '040', '041', '044', '045', '046', '047', '051', '055', '057', '064', '065','053', '058', '060', '061', '062', '063', '066', '067', '068', '069', '071', '072', '073', '074', '076', '077', '078', '079', '083', '086', '093', '094'])



    # train_dataset = loader.load_train([
    # '002','004','005','006','009','010','011','012','013','014',
    # '015','017','018','019','030','031','037','039','040','045',
    # '047','053','055','061','062','063','064','065','066','067',
    # '068','069','073','074','076','078','079','082','093','094',
    # '095','097','098','101','102','103','104','113','117','118'
    # ])
    loader = Loader(DATA_DIR, patch_dir_name='tumor_48x48x16')
    valid_dataset = loader.load_valid([
        '082', '088', '090', '095', '096', '097', '098', '101', '102', '103', '104', '105', '107', '109', '112', '113', '117', '118', '122', '123', '125', '130'
    ])
    # ?

    def weight_method(X, Y, nclasses):
        _Y = np.argmax(Y, axis=4)
        counts = np.array([(_Y == label).sum() for label in range(nclasses)])
        counts = np.cbrt(np.where(counts != 0, counts, 1))
        Y = Y * 1e0 / counts
        return Y
    train_generator = Generator(train_dataset, batch_size=BATCH_SIZE, nclasses=4, enable_random_crop=True,
                                crop_size=(48, 48, 16), threshold=float('inf'), weight_method=weight_method)
    valid_generator = Generator(valid_dataset, batch_size=BATCH_SIZE, nclasses=4, enable_random_crop=False,
                                crop_size=(48, 48, 16), threshold=float('inf'), weight_method=weight_method)

    with tf.Session(config=config) as sess:
        #(self, input_shape, nclasses, use_bn=True, use_dropout=True)
        #num of channe2l is 2(SE2,SE3)
        model = UNet3D((48, 48, 16, 2), 4)
        if Path(args.weight_path).is_file():
            print('_'*30)
            print(f'load model weight from {args.weight_path}')
            model.load_weights(os.path.join(args.weight_path))
        else:
            print('_'*30)
            print('no load model weight')

        callbacks = ConstructCallback(model)
        # image_logger...? experiment can't find.
        # callbacks.append(CometLogImageUploader(experiment, train_generator, upload_steps=250))

        model.compile(loss={'segment': categorical_crossentropy},
                      loss_weights={'segment': 1.},
                      optimizer=AdaBoundOptimizer(learning_rate=1e-3, final_lr=1e-1),
                      metrics={'segment': [bg_dice, hcc_dice, cyst_dice,angioma_dice]})
        # [bg_recall, bg_precision, bg_dice,
        #  hcc_recall, hcc_precision, hcc_dice,
        #  cyst_recall, cyst_precision, cyst_dice]
        model.fit_generator(train_generator, steps_per_epoch=len(train_generator),
                            validation_data=valid_generator, validation_steps=len(valid_generator),
                            callbacks=callbacks, workers=6, max_queue_size=12, use_multiprocessing=True,
                            epochs=50, shuffle=False)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
