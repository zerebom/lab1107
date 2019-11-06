import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import os
'''
評価に必要なdirづくりをするコード
expreriments/実験名/res直下に予測ラベルができてるので、
expreriments/実験名/ref dirを作成し、そこにraw_dataをcopyしてくる
kits19/data/case_00000/segmentation.nii.gz -> /res/case00000.nii.gz となる
python3 /home/kakeya/Desktop/higuchi/20191021/Keras/src/test/preprocess_evaluate.py /home/kakeya/Desktop/higuchi/20191021/Keras/experiments/1104_75epochs --ref_dir /home/kakeya/Desktop/higuchi/data
'''
ROOT_DIR='/home/kakeya/Desktop/higuchi/'

def ValidateArgs(args):
    if not (ROOT_DIR / args.experiments_dir).is_dir():
        print(f'Experiments directory({(ROOT_DIR /args.experiments_dir)}) is not dir')
        return False
    if not args.ref_dir.is_dir():
        print(f'Reference directory({args.ref_dir}) is not dir')
        return False
    return True

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments_dir')
    parser.add_argument('--ref_dir', default='/home/kakeya/Desktop/higuchi/data')

    args = parser.parse_args()
    args.experiments_dir = Path(args.experiments_dir)
    args.ref_dir = Path(args.ref_dir)

    return args
def make_concat_label(kidney_path,CCRCC_path=None,cyst_path=None):
    '''concat&return labels'''
    print(kidney_path.is_file(),CCRCC_path.is_file(),cyst_path.is_file())
    if not kidney_path.is_file():
        assert ValueError(f'{kidney_path} isn`t exist')
    kid=sitk.GetArrayFromImage(sitk.ReadImage(str(kidney_path)))
    ccrcc=sitk.GetArrayFromImage(sitk.ReadImage(str(CCRCC_path))) if CCRCC_path.is_file() else np.zeros(kid.shape)
    cyst=sitk.GetArrayFromImage(sitk.ReadImage(str(cyst_path))) if cyst_path.is_file() else np.zeros(kid.shape)
    label_array=np.zeros(kid.shape)

    #rare label overlap non rare labels
    label_array[kid==1]=1
    label_array[ccrcc==1]=2
    label_array[cyst==1]=3
    
    label=sitk.GetImageFromArray(label_array)
    return(label)




def main (args):
    resorce_file_list=sorted((ROOT_DIR/args.experiments_dir/'res').glob('*.nii.gz'))
    if len(resorce_file_list)==0:
        assert ValueError('resorce_file_list dosen`t find.')
    os.makedirs(ROOT_DIR/args.experiments_dir/'ref',exist_ok=True)
    for res in resorce_file_list:
        #.nii.gzはピリオドが二回あるので、splitで除去した

        kidney_path=(args.ref_dir/res.name.split('.')[0]/'kidney.nii.gz')
        ccrcc_path=(args.ref_dir/res.name.split('.')[0]/'CCRCC.nii.gz')
        cyst_path=(args.ref_dir/res.name.split('.')[0]/'cyst.nii.gz')
        label=make_concat_label(kidney_path,ccrcc_path,cyst_path)
        sitk.WriteImage(label,str(ROOT_DIR/args.experiments_dir/'ref'/res.name))



if __name__ == "__main__":
    args =ParseArgs()
    if ValidateArgs(args):
        main(args)
