import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import os
import yaml
'''
評価に必要なdirづくりをするコード
expreriments/実験名/res直下に予測ラベルができてるので、
expreriments/実験名/ref dirを作成し、そこにraw_dataをcopyしてくる
kits19/data/case_00000/segmentation.nii.gz -> /res/case00000.nii.gz となる
python3 /home/kakeya/Desktop/higuchi/20191021/Keras/src/test/preprocess_evaluate.py /home/kakeya/Desktop/higuchi/20191021/Keras/experiments/1104_75epochs --ref_dir /home/kakeya/Desktop/higuchi/data
'''

def get_yml(args):
    with open(args.setting_yml_path) as file:
        return yaml.load(file)

def ValidateArgs(args,yml):
    OWN_DIR= Path(yml['DIR']['OWN'])
    DATA_DIR= Path(yml['DIR']['DATA'])


    if not OWN_DIR.is_dir():
        print(f'Experiments directory({OWN_DIR}) is not dir')
        return False
    if not DATA_DIR.is_dir():
        print(f'Reference directory({DATA_DIR}) is not dir')
        return False
    return True

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-yml','--setting_yml_path',type=str)
    args = parser.parse_args()

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

def main (args,yml):
    ROOT_DIR = Path(yml['DIR']['ROOT'])
    DATA_DIR = Path(yml['DIR']['DATA'])
    OWN_DIR= Path(yml['DIR']['OWN'])

    resorce_file_list=sorted((ROOT_DIR/OWN_DIR/'res').glob('*.nii.gz'))
    if len(resorce_file_list)==0:
        assert ValueError('resorce_file_list dosen`t find.')
    os.makedirs(ROOT_DIR/OWN_DIR/'ref',exist_ok=True)
    for res in resorce_file_list:
        #.nii.gzはピリオドが二回あるので、splitで除去した

        kidney_path=(DATA_DIR/res.name.split('.')[0]/'kidney.nii.gz')
        ccrcc_path=(DATA_DIR/res.name.split('.')[0]/'CCRCC.nii.gz')
        cyst_path=(DATA_DIR/res.name.split('.')[0]/'cyst.nii.gz')
        label=make_concat_label(kidney_path,ccrcc_path,cyst_path)
        sitk.WriteImage(label,str(ROOT_DIR/OWN_DIR/'ref'/res.name))


if __name__ == "__main__":
    args =ParseArgs()
    yml=get_yml(args)
    if ValidateArgs(args,yml):
        main(args,yml)
