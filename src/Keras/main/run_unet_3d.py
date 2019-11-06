かimport argparse
import os
import datetime

import numpy as np
import tensorflow as tf

from callbacks import CometLogImageUploader, CustomizedLearningRateScheduler

from AdaBound import AdaBoundOptimizer
from dataset import Loader, Generator
from unet_3d import UNet3D
from loss_funcs import categorical_crossentropy, bg_recall, bg_precision, bg_dice, \
    hcc_recall, hcc_precision, hcc_dice, \
    cyst_recall, cyst_precision, cyst_dice, \
    angioma_recall, angioma_precision, angioma_dice

# main.pyてきなやつ


def ParseArgs():
    # from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_number', default=1, type=int)
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
    log_dir = pathlib.Path(f'~/Desktop/Lab/DNP/logs/{now.strftime("%Y-%m-%d_%H-%M")}')
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
    # DATA_DIR = '/home/higuchi/Desktop/kits19/data'
    # DATA_DIR = '/media/higuchi/Windows/NN_data/kits19/data'
    DATA_DIR = '/home/higuchi/ssd/kits19/data'
    BATCH_SIZE = 16

    loader = Loader(DATA_DIR, patch_dir_name='tumor_60x60x20')
    # return dataframe require:patch_npy
    train_dataset = loader.load_train([
        '000', '001', '003', '007', '009', '010', '011', '012',
        '014', '016', '017', '018', '020', '021', '022', '023',
        '024', '025', '026', '027', '028', '029', '030', '031', '032',
        '033', '035', '036', '038', '039', '040', '041', '042',
        '043', '044', '045', '046', '047', '048', '049', '050', '051',
        '052', '053', '054', '056', '057', '058', '062', '063', '064',
        '065', '066', '067', '068', '069', '070', '071', '072', '073',
        '074', '075', '076', '077', '078', '079', '080', '081', '082',
        '084', '085', '086', '087', '088', '089', '090', '092', '093',
        '094', '095', '096', '097', '098', '099', '100', '102', '103',
        '104', '106', '107', '108', '109', '110', '111', '112', '113',
        '116', '117', '118', '119', '121', '122', '125', '126', '128',
        '129', '130', '133', '134', '136', '137', '139', '140', '141',
        '143', '144', '146', '147', '150', '153', '154', '155',
        '157', '158', '159', '160', '161', '162', '163', '164', '165',
        '166', '167', '168', '169', '170', '172', '175', '176', '177',
        '178', '179', '180', '181', '183', '184', '186', '187', '188',
        '189', '190', '192', '193', '194', '195', '196', '197', '198',
        '200', '201', '202', '204', '205', '206', '207', '208', '209',
        # deleted
        # '005', '015', '037', '151'
    ])
    loader = Loader(DATA_DIR, patch_dir_name='tumor_48x48x16')
    valid_dataset = loader.load_valid([
        # small * mix
        '004', '142', '132', '171', '006', '174', '138', '059', '034',
        '131', '123', '083', '203', '061', '148', '105', '115',
        # small * mix * cyst
        '060', '182', '156', '152',
        # small * black
        '101', '127', '173', '013', '019', '091', '185',
        # small * black * cyst
        '124', '002',  # 1 -> 2
        # big * mix
        '199', '055', '149', '008',
        # big * mix * cyst
        '145',  # 0 -> 1
        # big * black
        '114', '135',
        # hard * mix
        '191', '120'  # 1 -> 2
    ])
    # ?

    def weight_method(X, Y, nclasses):
        _Y = np.argmax(Y, axis=4)
        counts = np.array([(_Y == label).sum() for label in range(nclasses)])
        counts = np.cbrt(np.where(counts != 0, counts, 1))
        Y = Y * 1e0 / counts
        return Y
    train_generator = Generator(train_dataset, batch_size=BATCH_SIZE, nclasses=3, enable_random_crop=True,
                                crop_size=(48, 48, 16), threshold=float('inf'), weight_method=weight_method)
    valid_generator = Generator(valid_dataset, batch_size=BATCH_SIZE, nclasses=3, enable_random_crop=False,
                                crop_size=(48, 48, 16), threshold=float('inf'), weight_method=weight_method)

    with tf.Session(config=config) as sess:
        model = UNet3D((48, 48, 16, 1), 3)

        callbacks = ConstructCallback(model)
        # image_logger...? experiment can't find.
        # callbacks.append(CometLogImageUploader(experiment, train_generator, upload_steps=250))

        model.compile(loss={'segment': categorical_crossentropy},
                      loss_weights={'segment': 1.},
                      optimizer=AdaBoundOptimizer(learning_rate=1e-3, final_lr=1e-1),
                      metrics={'segment': [bg_dice, hcc_dice, cyst_dice]})
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
