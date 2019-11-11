import subprocess
import pathlib
import os
import argparse
import yaml
'''
TODO:setting.ymlにパッチのsuffixなども書いておく
'''

def ParseArgs():
    # from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('-yml','--setting_yml_path',type=str,default='/home/kakeya/Desktop/higuchi/20191107/experiment/sigle_channel/setting.yml')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.setting_yml_path) as file:
        yml = yaml.load(file)
        ROOT_DIR = yml['DIR']['ROOT']
        DATA_DIR = yml['DIR']['DATA']

        train_cid=yml['CID']['TRAIN']
        val_cid=yml['CID']['VAL']
        test_cid=yml['CID']['TEST']

        train_patch=yml['PATCH_DIR']['TRAIN']
        val_patch=yml['PATCH_DIR']['VAL']
    
    cid_list=train_cid+val_cid+test_cid
    nii_list=['SE2.nii.gz','SE3.nii.gz','kidney.nii.gz','CCRCC.nii.gz','cyst.nii.gz']

    for cid in cid_list:
        cmd1=f'cd {DATA_DIR}/00{cid}'
        cmd2='pwd'
        cmd3=f'python3 {ROOT_DIR}/src/create_hist_equal_patch.py {" ".join(nii_list)} --size 60 60 20 -st -suffix standerd05'
        cmd4=f'python3 {ROOT_DIR}/src/create_hist_equal_patch.py {" ".join(nii_list)} --size 48 48 16 -st -suffix standerd05'
        for cmd in [cmd1,cmd2,cmd3,cmd4]:

            subprocess.call(cmd.split(),shell=True)

if __name__== '__main__':
    args = ParseArgs()
    main(args)

