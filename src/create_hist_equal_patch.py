import SimpleITK as sitk
import numpy as np
import pathlib
from tqdm import tqdm
import numpy.ma as ma
import glob
# /home/higuchi/Desktop/kits19/data/case_00000/imaging.nii.gz
# /home/higuchi/Desktop/kits19/data/case_00000/segmentation.nii.gz

'''
間違えて途中まで作っちゃったとき↓
sudo rm -rf ./*/tumor*standard*
'''
'''
for i in `seq -w 000 160`; do
cd /home/higuchi/Desktop/higuchi/data/00${i}
pwd
sudo python3 /home/higuchi/Desktop/higuchi/lab1107/src/create_hist_equal_patch.py SE2.nii.gz SE3.nii.gz kidney.nii.gz CCRCC.nii.gz cyst.nii.gz --size 60 60 20 -st -sf standard05
sudo python3 /home/higuchi/Desktop/higuchi/lab1107/src/create_hist_equal_patch.py SE2.nii.gz SE3.nii.gz kidney.nii.gz CCRCC.nii.gz cyst.nii.gz --size 48 48 16 -st -sf standard05
'''


'''
for i in `seq -w 000 160`; do
cd /home/kakeya/Desktop/higuchi/data/00${i}
sudo python3 /home/kakeya/Desktop/higuchi/20191107/src/create_hist_equal_patch.py SE2.nii.gz SE3.nii.gz kidney.nii.gz CCRCC.nii.gz cyst.nii.gz --suffix standard05 --size 60 60 20
sudo python3 /home/kakeya/Desktop/higuchi/20191107/src/create_hist_equal_patch.py SE2.nii.gz SE3.nii.gz kidney.nii.gz CCRCC.nii.gz cyst.nii.gz --suffix standard05 --size 48 48 16
pwd

done
大体30分くらいで処理が終わる。
'''

import argparse


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_volume_list', nargs=2)
    parser.add_argument('label_volume_list', nargs=3)
    # nargs...受け取る引数の数。?なら0 or 1こ
    parser.add_argument('--size', nargs=3, type=int)
    parser.add_argument('-st', '--standardization', action='store_true')
    parser.add_argument("--onehot", help="Whether or not to Onehot Vector is Save data",
                        default=False, action='store_false')
    parser.add_argument('-sf', "--suffix", type=str, default='hist_equal_05')

    args = parser.parse_args()
    return args


def ValidateArgs(args):
    dirs = glob.glob('./*')
    if f"./tumor_{'x'.join(map(str, args.size))}_{args.suffix}" in dirs:

        print('already exist patch dir')
        return False

    for image_volume in args.image_volume_list:
        if not pathlib.Path(image_volume).is_file():
            print(f'Image data({image_volume}) is not file.')
            return False
    if not pathlib.Path('kidney.nii.gz'):
        print('no kidney voxel')
        return False

    for i, label_volume in enumerate(args.label_volume_list):
        if not pathlib.Path(label_volume).is_file():
            args.label_volume_list[i] = None
            print(f'Label data({label_volume}) is not file.')

    return True

# -----------------new preprocess function ---------------------------


def standardization(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    new_x = (x - xmean) / xstd
    return new_x


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
# -----------------new preprocess function ---------------------------


import math


def getListCropPoint(read_range, pad_range):
    # read_range：パッチの始点の範囲
    # pad_range：始点の基準間隔
    equal = math.ceil(read_range / pad_range)
    crop_point = np.round(np.linspace(0, read_range, equal + 1)).astype(int)

    return crop_point


def getOnehotVector(imagearry):
    # TODO:動的に変換できるようにする
    tmparry = np.zeros([imagearry.shape[0], imagearry.shape[1], imagearry.shape[2], 2])
    tmparry[:, :, :, 1] = imagearry == 1
    tmparry[:, :, :, 0] = imagearry != 1
    return tmparry


def saveMHA(array, image, save_path):
    save_image = sitk.GetImageFromArray(array)
    save_image.SetOrigin(image.GetOrigin())
    save_image.SetSpacing(image.GetSpacing())
    save_image.SetDirection(image.GetDirection())
    sitk.WriteImage(save_image, save_path, True)


def saveNPY(array, save_path, float=False):
    if float:
        array = array.astype(np.float16)
        np.save(save_path, array)
    else:
        np.save(save_path, array)


def main(args):
    # make dir
    save_directory = pathlib.Path(f"./tumor_{'x'.join(map(str, args.size))}_{args.suffix}")
    if save_directory.is_dir:
        assert EnvironmentError('savedir already contains patches.')

    save_directory.mkdir(exist_ok=True)

    # read image and make list
    image_list = [sitk.ReadImage(image_volume) for image_volume in args.image_volume_list]
    label_list = [sitk.ReadImage(label_volume)
                  if label_volume is not None else None for label_volume in args.label_volume_list]
    # make tmp arr for get array shape
    tmp_array = sitk.GetArrayFromImage(image_list[0])
    # タプルを足して4次元にしている...?
    image_array = np.zeros(tmp_array.shape + (len(image_list),), dtype=tmp_array.dtype)
    kid_aray = sitk.GetArrayFromImage(sitk.ReadImage('kidney.nii.gz'))

    for i, image in enumerate(image_list):
        # add channel
        SE_array = sitk.GetArrayFromImage(image)
        image_array[..., i] = histgram_equalization(SE_array, kid_aray, vmin=-750, vmax=750, alpha=0.5)
        if args.standardization:
            image_array = standardization(image_array)

    label_array = np.zeros(tmp_array.shape, dtype=np.int16)
    for i, label in enumerate(label_list):
        if label is not None:
            # add label by bianry system for overlap more double label
            label_array += sitk.GetArrayFromImage(label) * 2 ** i
    # label_array[label_array > 0] = 1

    patch_index = 0

    # パッチの読み取り範囲をOriginとサイズから、最小のボックスになるように調整する
    read_range = image.GetSize()

    # パディングした全領域を出力パッチサイズでラスタスキャンする
    # 出力パッチサイズで割り切れずはみ出してしまう領域は、均等にずらして確保する
    # TODO:Augmentation内でクロップするほうがよさそう？
    z_crop_point = getListCropPoint(read_range[0] - args.size[2], args.size[2] // 2)
    y_crop_point = getListCropPoint(read_range[1] - args.size[1], args.size[1] // 5)
    x_crop_point = getListCropPoint(read_range[2] - args.size[0], args.size[0] // 5)

    def make_patch(x, y, z, patch_index):
        #　ラベルはネットワークの出力パッチサイズの大きさでクロップする
        crop_label = label_array[x:x + args.size[0], y:y + args.size[1], z:z + args.size[2]]

        # バッチ内に腫瘍領域が存在しない
        # パッチ内に除外領域が含まれる
        # 腫瘍領域がパッチボクセルの 80% 以上
        if (crop_label != 0).sum() <= np.prod(args.size) * 0.01:
                # or crop_exclude.sum() != 0 \
                # or (crop_label!=0).sum() >= np.prod(args.size) * 0.8:
            return
        if crop_label.shape[2] != args.size[2]:
            # src:http://nonbiri-tereka.hatenablog.com/entry/2014/06/22/171504
            # デバッグツール
            import ipdb
            ipdb.set_trace()

        if args.onehot:
            crop_label = getOnehotVector(crop_label)
            save_seg_path = save_directory / f'patch_onehot_{patch_index}.npy'
        else:
            save_seg_path = save_directory / f'patch_no_onehot_{patch_index}.npy'
        saveNPY(crop_label, str(save_seg_path))

        #　ボリュームはネットワークの入力パッチサイズの大きさでクロップする
        crop_vol = image_array[x:x + args.size[0],
                               y:y + args.size[1],
                               z:z + args.size[2], :]

        save_vol_path = save_directory / f'patch_image_{patch_index}.npy'
        saveNPY(crop_vol, str(save_vol_path), True)

    patch_method = make_patch

    for z in tqdm(z_crop_point):
        for y in y_crop_point:
            for x in x_crop_point:
                patch_method(x, y, z, patch_index)
                patch_index += 1


if __name__ == '__main__':
    args = ParseArgs()
    if ValidateArgs(args):
        main(args)
