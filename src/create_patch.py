import SimpleITK as sitk
import numpy as np
import pathlib
from tqdm import tqdm

#/home/higuchi/Desktop/kits19/data/case_00000/imaging.nii.gz
#/home/higuchi/Desktop/kits19/data/case_00000/segmentation.nii.gz


'''
for i in `seq -w 000 160`; do
cd /home/kakeya/Desktop/higuchi/DNP/data/00${i}
sudo python3 /home/kakeya/Desktop/higuchi/DNP/src/create_patch.py SE2.nii.gz SE3.nii.gz kidney.nii.gz CCRCC.nii.gz cyst.nii.gz --size 60 60 20 -st --su
sudo python3 /home/kakeya/Desktop/higuchi/DNP/src/create_patch.py SE2.nii.gz SE3.nii.gz kidney.nii.gz CCRCC.nii.gz cyst.nii.gz --size 48 48 16 -st 
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
    parser.add_argument('--exclude_area', nargs='?')
    parser.add_argument('--predict_volume', default='', nargs='?')
    parser.add_argument('--size', nargs=3, type=int)
    # parser.add_argument("--input_size", help="Input Network Patch Size", default=[144, 144, 1], nargs='+', type=int)
    # parser.add_argument("--output_size", help="Output Network Patch Size", default=[36, 36, 1], nargs='+', type=int)
    parser.add_argument("--onehot", help="Whether or not to Onehot Vector is Save data", default=False, action='store_false')
    args = parser.parse_args()
    return args

def ValidateArgs(args):
    for image_volume in args.image_volume_list:
        if not pathlib.Path(image_volume).is_file():
            print(f'Image data({image_volume}) is not file.')
            return False
    for i, label_volume in enumerate(args.label_volume_list):
        if not pathlib.Path(label_volume).is_file():
            args.label_volume_list[i] = None
            print(f'Label data({label_volume}) is not file.')
            return False

    return True



import math
def getListCropPoint(read_range, pad_range):
    # read_range：パッチの始点の範囲
    # pad_range：始点の基準間隔
    equal = math.ceil(read_range/pad_range)
    crop_point = np.round(np.linspace(0, read_range, equal+1)).astype(int)

    return crop_point

def getOnehotVector(imagearry):    
    # TODO:動的に変換できるようにする
    tmparry = np.zeros([imagearry.shape[0], imagearry.shape[1], imagearry.shape[2], 2])
    tmparry[:,:,:,1] = imagearry==1
    tmparry[:,:,:,0] = imagearry!=1
    return tmparry

def saveMHA(array, image, save_path):
    save_image = sitk.GetImageFromArray(array)
    save_image.SetOrigin(image.GetOrigin())
    save_image.SetSpacing(image.GetSpacing())
    save_image.SetDirection(image.GetDirection())
    sitk.WriteImage(save_image, save_path, True)

def saveNPY(array, save_path):
    np.save(save_path, array)

def main(args):
    #make dir
    save_directory = pathlib.Path(f"./tumor_{'x'.join(map(str, args.size))}")
    if len(list(save_directory.glob('*')))>0:
        assert EnvironmentError('savedir already contains patches.')

    save_directory.mkdir(exist_ok=True)
    
    #read image and make list
    image_list = [sitk.ReadImage(image_volume) for image_volume in args.image_volume_list]
    label_list = [sitk.ReadImage(label_volume) if label_volume is not None else None for label_volume in args.label_volume_list]
    #make tmp arr for get array shape
    tmp_array = sitk.GetArrayFromImage(image_list[0])
    #タプルを足して4次元にしている...?
    image_array = np.zeros(tmp_array.shape + (len(image_list),), dtype=tmp_array.dtype)
    for i, image in enumerate(image_list):
      # add channel
      image_array[..., i] = sitk.GetArrayFromImage(image)

    label_array = np.zeros(tmp_array.shape, dtype=np.int16)
    for i, label in enumerate(label_list):
      if label is not None:
          # add label by bianry system for overlap more double label
            label_array += sitk.GetArrayFromImage(label) * 2 ** i
    # label_array[label_array > 0] = 1

    #exclude area
    if args.exclude_area and pathlib.Path(args.exclude_area).is_file():
        exclude_array = sitk.GetArrayFromImage(sitk.ReadImage(args.exclude_area))
    else:
        exclude_array = np.zeros(label_array.shape, dtype=label_array.dtype)
    # no use
    if pathlib.Path(args.predict_volume).is_file():
        predict_array = sitk.GetArrayFromImage(sitk.ReadImage(args.predict_volume))
        predict_array = (predict_array > 0).astype(label_array.dtype)
    else:
        predict_array = np.zeros(label_array.shape, dtype=label_array.dtype)

    patch_index = 0

    # パッチの読み取り範囲をOriginとサイズから、最小のボックスになるように調整する
    # lower = np.fmax([0,0,0], label.TransformPhysicalPointToIndex(image.TransformIndexToPhysicalPoint((0,0,0))))
    # upper = np.fmin(label.GetSize(), label.TransformPhysicalPointToIndex(image.TransformIndexToPhysicalPoint(image.GetSize())))
    read_range = image.GetSize()

    # パディングした全領域を出力パッチサイズでラスタスキャンする
    # 出力パッチサイズで割り切れずはみ出してしまう領域は、均等にずらして確保する
    # TODO:Augmentation内でクロップするほうがよさそう？
    # z_crop_point = getListCropPoint(read_range[0]-args.size[2], args.size[2]//5)
    # args[2] is overlap stride
    z_crop_point = getListCropPoint(read_range[0]-args.size[2], args.size[2]//2)
    y_crop_point = getListCropPoint(read_range[1]-args.size[1], args.size[1]//5)
    x_crop_point = getListCropPoint(read_range[2]-args.size[0], args.size[0]//5)

    def make_patch(x, y, z, patch_index):
        #　ラベルはネットワークの出力パッチサイズの大きさでクロップする
        crop_label   =   label_array[x:x+args.size[0], y:y+args.size[1], z:z+args.size[2]]
        crop_exclude = exclude_array[x:x+args.size[0], y:y+args.size[1], z:z+args.size[2]]

        # バッチ内に腫瘍領域が存在しない
        # パッチ内に除外領域が含まれる
        # 腫瘍領域がパッチボクセルの 80% 以上
        if (crop_label!=0).sum() <= np.prod(args.size) * 0.01:
                # or crop_exclude.sum() != 0 \
                # or (crop_label!=0).sum() >= np.prod(args.size) * 0.8:
            return
        if crop_label.shape[2] != args.size[2]:
            #src:http://nonbiri-tereka.hatenablog.com/entry/2014/06/22/171504 
            # デバッグツール
            import ipdb; ipdb.set_trace()

        if args.onehot:
            crop_label = getOnehotVector(crop_label)
            save_seg_path = save_directory / f'patch_onehot_{patch_index}.npy'
        else:
            save_seg_path = save_directory / f'patch_no_onehot_{patch_index}.npy'
        saveNPY(crop_label, str(save_seg_path))

        #　ボリュームはネットワークの入力パッチサイズの大きさでクロップする
        crop_vol = image_array[x:x+args.size[0],           
                               y:y+args.size[1],           
                               z:z+args.size[2], :]        

        save_vol_path = save_directory / f'patch_image_{patch_index}.npy'
        saveNPY(crop_vol, str(save_vol_path))              
                                                           
    def make_patch_with_pred(x, y, z, patch_index):
        #　ラベルはネットワークの出力パッチサイズの大きさでクロップする
        crop_label   =   label_array[z:z+args.size[2], y:y+args.size[1], x:x+args.size[0]]
        crop_exclude = exclude_array[z:z+args.size[2], y:y+args.size[1], x:x+args.size[0]]
        crop_predict = predict_array[z:z+args.size[2], y:y+args.size[1], x:x+args.size[0]]

        # パッチ内に検出領域が存在しない
        # パッチ内に除外領域が含まれる
        if crop_predict.sum() == 0 \
                or crop_exclude.sum() != 0:
            return

        if args.onehot:
            crop_label = getOnehotVector(crop_label)
            save_seg_path = save_directory / f'patch_onehot_{patch_index}.npy'
        else:
            save_seg_path = save_directory / f'patch_no_onehot_{patch_index}.npy'
        saveNPY(crop_label, str(save_seg_path))

        #　ボリュームはネットワークの入力パッチサイズの大きさでクロップする
        crop_vol = image_array[z:z+args.size[2],
                               y:y+args.size[1],
                               x:x+args.size[0]]
        crop_vol_tmp = np.zeros(crop_vol.shape[:-1] + (crop_vol.shape[-1]+1,), dtype=crop_vol.dtype)
        crop_vol_tmp[..., :crop_vol.shape[-1]] = crop_vol
        crop_vol_tmp[..., :-1] = crop_predict[..., np.newaxis]

        save_vol_path = save_directory / f'patch_image_{patch_index}.npy'
        saveNPY(crop_vol_tmp, str(save_vol_path))

    patch_method = make_patch if predict_array.sum() == 0 else make_patch_with_pred

    for z in tqdm(z_crop_point):
        for y in y_crop_point:
            for x in x_crop_point:
                patch_method(x, y, z, patch_index)
                patch_index += 1


if __name__== '__main__':
    args = ParseArgs()
    if ValidateArgs(args):
        main(args)
