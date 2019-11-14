import sys
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os
from itertools import product
import argparse
import scipy.ndimage
import pathlib
sys.path.append('/home/kakeya/Desktop/higuchi/20191107/src/Keras')
from model.unet_3d import UNet3D
import yaml
import numpy.ma as ma

args = None

'''
1114修正。
作ったバッチ（前処理後）を用いて推論するようにする。
'''
# python pred3D.py "D:\okada_script\MHA_0802\00125\SE3\patient.mha" "D:\Code\unet3d\data\3dunet_2class.yml" "./log/latestweights.hdf5" --mask="D:\okada_script\MHA_0802\00125\SE3\肝臓.mha"

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("predfile_list", help="Prediction file (*.mha)", nargs='+')
    parser.add_argument('-mask',"--maskfile", help="Mask file (*.nii.gz)")
    parser.add_argument("--tumorfile")
    parser.add_argument("--cystfile")
    parser.add_argument("--predfile", default='hogehuga')
    parser.add_argument("--stepscale", help="Step scale for patch tiling.", default=1.0, type=float)
    parser.add_argument("--save_dir", default='./result')
    parser.add_argument("--outfilename", default='pred_label.mha')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=1, type=int)
    parser.add_argument('-c', '--class_num', type=int)
    parser.add_argument('-yml', '--setting_yml_path', type=str)

    args = parser.parse_args()
    return args


def ValidateArgs(args):
    if not pathlib.Path(args.modelweightfile).exists():
        print(f'Model weight({args.modelweightfile}) is not found.')
        return False
    for predfile in args.predfile_list:
        if not pathlib.Path(predfile).exists():
            print(f'Patient CT data({predfile}) is not found.')
            return False

    return True


def extract_max_island(array):
    image = sitk.GetImageFromArray(array)
    cc = sitk.ConnectedComponent(image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, image)
    max_label, max_size = 0, 0
    for label in stats.GetLabels():
        label_size = stats.GetPhysicalSize(label)
        if max_size < label_size:
            max_label, max_size = label, label_size

    array[sitk.GetArrayFromImage(cc) != max_label] = 0

    return array


def scoring_dice(pred, true):
    return (float)(2. * np.count_nonzero(np.logical_and(pred, true))) / (np.count_nonzero(pred) + np.count_nonzero(true))


def scoring_dice2(pred, true):
    return (float)(np.count_nonzero(np.logical_and(pred, true))) / (np.count_nonzero(true))


def norm(x, w=350, l=40):
    x = np.clip(x, l - w // 2, l + w // 2)
    x = x - (l - w // 2)
    x = x / w
    return x


def SaveVolume(path, volume_array, ref_image):
    volume = sitk.GetImageFromArray(volume_array)
    volume.SetOrigin(ref_image.GetOrigin())
    volume.SetSpacing(ref_image.GetSpacing())
    volume.SetDirection(ref_image.GetDirection())
    sitk.WriteImage(volume, str(path), True)


def histgram_equalization(image_array, mask_array, vmin=-750, vmax=750, alpha=0.5):

    image_array = np.clip(image_array, vmin, vmax)
    image_array += vmax
    mask_image_array = ma.masked_where(mask_array == 0, image_array)

    ctRange = vmax - vmin + 1
    HIST = np.array([0.0] * ctRange)
    roi_hist, _ = np.histogram(mask_image_array[~mask_image_array.mask], ctRange, [0, ctRange])

    # 1に正規化する
    HIST = roi_hist / roi_hist.sum()
    # 一様分布を混ぜる
    HIST = HIST * alpha + (1 - alpha) / 1500
    # 累積和を求める
    cdf = HIST.cumsum()
    # 度数が0のところは処理しないというマスクを作成する
    mask_cdf = np.ma.masked_equal(cdf, 0)
    standared_mask_cdf = (mask_cdf - mask_cdf.min()) / (mask_cdf.max() - mask_cdf.min())
    standared_mask_cdf = 1500 * standared_mask_cdf
    standared_mask_cdf = np.ma.filled(standared_mask_cdf, 0).astype('int64')


    image_array = image_array.astype(int)
    new_image_array = standared_mask_cdf[image_array] - 750
    return new_image_array


def standardization(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    new_x = (x - xmean) / xstd
    return new_x


def main(_):
    with open(args.setting_yml_path) as file:
        yml = yaml.load(file)
        ROOT_DIR = yml['DIR']['ROOT']
        patch_shape = yml['PATCH_SHAPE']
        LOCAL_HE= yml['LOCAL_HE'] if 'LOCAL_HE' in yml else False
        STANDARD = yml['STANDARD'] if 'STANDARD' in yml else False



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        print('loading 3D U-net model ...', end='', flush=True)
        if not pathlib.Path(args.predfile).exists():
            model = UNet3D(patch_shape, args.class_num)
        else:
            model = UNet3D(patch_shape, args.class_num)
        print('loading weights...', end='', flush=True)
        # error
        model.load_weights(args.modelweightfile)
        print('done')

    print(f'input_shape = {model.input_shape}')
    print(f'output_shape = {model.output_shape}')

    # ここからパッチの用意。順番をちゃんともとの形に戻せるようにしないと行けない
    # get patch size
    patch_size = np.array(model.input_shape[1:4])
    # patch_size = patch_size[::-1]

    image_list = [sitk.ReadImage(predfile) for predfile in args.predfile_list]
    tmp_array = sitk.GetArrayFromImage(image_list[0])
    if not pathlib.Path(args.predfile).exists():
        image_array = np.zeros(tmp_array.shape + (len(args.predfile_list),), dtype=tmp_array.dtype)
    else:
        image_array = np.zeros(tmp_array.shape + (len(args.predfile_list) + 1,), dtype=tmp_array.dtype)
        image_array[..., -1] = (sitk.GetArrayFromImage(sitk.ReadImage(args.predfile)) > 0).astype(tmp_array.dtype)

    for i, image in enumerate(image_list):
        print(sitk.GetArrayFromImage(image).shape)
        tmp_im=sitk.GetArrayFromImage(image)
        if LOCAL_HE:
            mask_array = sitk.GetArrayFromImage(sitk.ReadImage(args.maskfile))
            image_array[..., i] = histgram_equalization(tmp_im, mask_array, alpha=0.5)
        else:
            image_array[..., i] = tmp_im
    if STANDARD==True:
        image_array=standardization(image_array)

    if patch_shape[-1]==1:
        image_array=image_array[...,0]
        image_array=image_array[...,np.newaxis]

    shape = image_array.shape[:3]

    step = (patch_size / args.stepscale).astype(np.int8)
    print('step = {}'.format(step))

    label = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    # 正解ラベルの入れ物
    labelarr = sitk.GetArrayFromImage(label)
    print('labelarr shape: {}'.format(labelarr.shape))

    # オーバーラップの回数を記憶しているarr
    counterarr = sitk.GetArrayFromImage(sitk.Image(image.GetSize(), sitk.sitkVectorUInt8, model.output_shape[-1]))
    # predictの生起確率を記憶するarr
    paarry = sitk.GetArrayFromImage(sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, model.output_shape[-1]))

    if not args.maskfile or not pathlib.Path(args.maskfile).exists():
        mask_array = np.ones(labelarr.shape)
    else:
        # maskを大きくしてから学習。
        mask_image = sitk.ReadImage(args.maskfile)
        dilate = sitk.BinaryDilateImageFilter()
        dilate.SetKernelRadius(3)
        dilated = dilate.Execute(mask_image)
        mask_array = sitk.GetArrayFromImage(dilated)

    from tqdm import trange
    for iz in trange(0, shape[2], step[2]):
        iz = iz if (iz + patch_size <= shape)[2] else (shape - patch_size)[2]
        for iy in range(0, shape[1], step[1]):
            iy = iy if (iy + patch_size <= shape)[1] else (shape - patch_size)[1]
            for ix in range(0, shape[0], step[0]):
                ix = ix if (ix + patch_size <= shape)[0] else (shape - patch_size)[0]
                patchimagearray = image_array[ix:ix + patch_size[0], iy:iy + patch_size[1], iz:iz + patch_size[2], :]
                patchimagearray = patchimagearray[np.newaxis, ...]  # .transpose((0,2,3,1,4))

                is_in_liver = mask_array[ix:ix + patch_size[0], iy:iy + patch_size[1], iz:iz + patch_size[2]].sum() != 0
                if patchimagearray.shape[-1] == len(args.predfile_list) + 1:
                    is_in_liver = is_in_liver and patchimagearray[..., -1].sum() != 0

                pavec = model.predict_on_batch(patchimagearray)[0] if is_in_liver else np.zeros(model.output_shape[1:])

                paarry[ix:ix + patch_size[0], iy:iy + patch_size[1], iz:iz + patch_size[2], :] += pavec
                counterarr[ix:ix + patch_size[0], iy:iy + patch_size[1], iz:iz + patch_size[2], :] += \
                    np.ones(pavec.shape, dtype=np.uint8) if is_in_liver else np.zeros(pavec.shape, dtype=np.uint8)

    print('Re resampling pred mask ...', end='', flush=True)
    counterarr[counterarr == 0] = 1
    paarry = paarry / counterarr
    print('done')
    print('labelarr : {}'.format(labelarr.shape))

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    labelarr = paarry
    labelarr = np.argmax(paarry, axis=-1).astype(dtype=np.int8)
    # mask_array = extract_max_island(mask_array + labelarr)
    labelarr[mask_array == 0] = 0
    SaveVolume(save_dir / f'{args.outfilename}', labelarr, image)


if __name__ == '__main__':
    args = ParseArgs()
    if ValidateArgs(args):
        tf.app.run(main=main, argv=[sys.argv[0]])
