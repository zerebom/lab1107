import sys
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os
from itertools import product
import argparse
import scipy.ndimage
import pathlib

from unet_3d import UNet3D

args = None


# python pred3D.py "D:\okada_script\MHA_0802\00125\SE3\patient.mha" "D:\Code\unet3d\data\3dunet_2class.yml" "./log/latestweights.hdf5" --mask="D:\okada_script\MHA_0802\00125\SE3\肝臓.mha"
def ParseArgs():
    parser = argparse.ArgumentParser()  
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("predfile_list", help="Prediction file (*.mha)", nargs='+')
    parser.add_argument("--maskfile", help="Mask file (*.mha)")
    parser.add_argument("--tumorfile")
    parser.add_argument("--cystfile")
    parser.add_argument("--predfile", default='hogehuga')
    parser.add_argument("--stepscale", help="Step scale for patch tiling.", default=1.0, type=float)
    parser.add_argument("--save_dir", default='./result')
    parser.add_argument("--outfilename", default='pred_label.mha')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument('-c', '--class_num', type=int)

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
    # if args.maskfile and not pathlib.Path(args.maskfile).exists():
    #     print(f'Mask for dice calc file({args.maskfile}) is not found.')
    #     return False
    # if args.tumorfile and not pathlib.Path(args.maskfile).exists():
    #     print(f'Tumor file({args.tumorfile}) is not found.')
    #     return False

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
    x = np.clip(x, l - w//2, l + w//2)
    x = x - (l - w//2)
    x = x / w
    return x

def SaveVolume(path, volume_array, ref_image):
    volume = sitk.GetImageFromArray(volume_array)
    volume.SetOrigin(ref_image.GetOrigin())
    volume.SetSpacing(ref_image.GetSpacing())
    volume.SetDirection(ref_image.GetDirection())
    sitk.WriteImage(volume, str(path), True)

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        print('loading 3D U-net model ...', end='', flush=True)
        if not pathlib.Path(args.predfile).exists():
            model = UNet3D((48, 48, 16, 2), args.class_num)
        else:
            model = UNet3D((48, 48, 16, 2), args.class_num)
        print('loading weights...', end='', flush=True) 
        #error         
        model.load_weights(args.modelweightfile) 
        print('done')

    print(f'input_shape = {model.input_shape}')
    print(f'output_shape = {model.output_shape}')

    # get patch size
    patch_size = np.array(model.input_shape[1:4])
    # patch_size = patch_size[::-1]

    image_list = [sitk.ReadImage(predfile) for predfile in args.predfile_list]
    tmp_array = sitk.GetArrayFromImage(image_list[0])
    if not pathlib.Path(args.predfile).exists():
        image_array = np.zeros(tmp_array.shape + (len(args.predfile_list),), dtype=tmp_array.dtype)
    else:
        image_array = np.zeros(tmp_array.shape + (len(args.predfile_list)+1,), dtype=tmp_array.dtype)
        image_array[..., -1] = (sitk.GetArrayFromImage(sitk.ReadImage(args.predfile)) > 0).astype(tmp_array.dtype)
    for i, image in enumerate(image_list):
        image_array[..., i] = sitk.GetArrayFromImage(image)

    # print('loading input image {}...'.format(args.predfile), end='', flush=True)
    # image = sitk.ReadImage(args.predfile)
    # image_array = sitk.GetArrayFromImage(image)
    # # spacing = image.GetSpacing()[::-1]
    # shape = image.GetSize()
    shape = image_array.shape[:3]

    step = (patch_size / args.stepscale).astype(np.int8)
    print('step = {}'.format(step))

    label = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    labelarr = sitk.GetArrayFromImage(label)
    print('labelarr shape: {}'.format(labelarr.shape))
    counterarr = sitk.GetArrayFromImage(sitk.Image(image.GetSize(), sitk.sitkVectorUInt8, model.output_shape[-1]))
    paarry = sitk.GetArrayFromImage(sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, model.output_shape[-1]))

    if not args.maskfile or not pathlib.Path(args.maskfile).exists():
        mask_array = np.ones(labelarr.shape)
    else:
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

                patchimagearray = image_array[ix:ix+patch_size[0], iy:iy+patch_size[1], iz:iz+patch_size[2], :]
                patchimagearray = patchimagearray[np.newaxis, ...]#.transpose((0,2,3,1,4))

                is_in_liver = mask_array[ix:ix+patch_size[0], iy:iy+patch_size[1], iz:iz+patch_size[2]].sum() != 0
                if patchimagearray.shape[-1] == len(args.predfile_list)+1:
                    is_in_liver = is_in_liver and patchimagearray[..., -1].sum() != 0
                
                pavec = model.predict_on_batch(patchimagearray)[0] if is_in_liver else np.zeros(model.output_shape[1:])
                # pavec = pavec.transpose((2,0,1,3))
                
                paarry[ix:ix+patch_size[0], iy:iy+patch_size[1], iz:iz+patch_size[2], :] += pavec
                counterarr[ix:ix+patch_size[0], iy:iy+patch_size[1], iz:iz+patch_size[2], :] += \
                    np.ones(pavec.shape, dtype=np.uint8) if is_in_liver else np.zeros(pavec.shape, dtype=np.uint8)
                
    print('Re resampling pred mask ...', end='', flush=True)
    counterarr[counterarr == 0] = 1
    paarry = paarry / counterarr
    print('done')
    print('labelarr : {}'.format(labelarr.shape))

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # # 背景の確率 Map
    # background_probbability = paarry[..., 0]
    # background_probbability[mask_array == 0] = 0
    # SaveVolume(save_dir / f'{args.patient_id}_background_probability.mha', background_probbability, image)
    # # HCC の確率 Map
    # hcc_probability = paarry[..., 1]
    # hcc_probability[mask_array == 0] = 0
    # SaveVolume(save_dir / f'{args.patient_id}_hcc_probability.mha', hcc_probability, image)
    # # cyst の確率 Map
    # cyst_probability = paarry[..., 2]
    # cyst_probability[mask_array == 0] = 0
    # SaveVolume(save_dir / f'{args.patient_id}_cyst_probability.mha', cyst_probability, image)
    # # angioma の確率 Map
    # angioma_probability = paarry[..., 3]
    # angioma_probability[mask_array == 0] = 0
    # SaveVolume(save_dir / f'{args.patient_id}_angioma_probability.mha', angioma_probability, image)
    # Argmax から求めた正解ラベル
    labelarr = paarry
    labelarr = np.argmax(paarry, axis=-1).astype(dtype=np.int8)
    # mask_array = extract_max_island(mask_array + labelarr)
    labelarr[mask_array == 0] = 0
    SaveVolume(save_dir / f'{args.outfilename}', labelarr, image)

if __name__ == '__main__':
    args = ParseArgs()
    if ValidateArgs(args):
        tf.app.run(main=main, argv=[sys.argv[0]])
