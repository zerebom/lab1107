import subprocess
import pathlib
import os
import yaml
import argparse


def ParseArgs():
    # from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('-yml', '--setting_yml_path', type=str,
                        default='/home/kakeya/Desktop/higuchi/20191107/experiment/sigle_channel/setting.yml')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)


    args = parser.parse_args()
    return args


def get_yml(args):
    with open(args.setting_yml_path) as file:
        return yaml.load(file)


def main(args, yml):
    patient_ids = yml['CID']['TEST']
    OWN_DIR = yml['DIR']['OWN']
    DATA_DIR = yml['DIR']['DATA']
    WIGHT_PATH = yml['PRED_WEIGHT']
    save_dir = f'{OWN_DIR}/res'
    os.makedirs(save_dir, exist_ok=True)

    for patient_id in patient_ids:
        dir_name = f'{DATA_DIR}/00{patient_id}/'
        mask_path = f'{DATA_DIR}/00{patient_id}/kidney.nii.gz'
        SE2_volume = pathlib.Path(dir_name) / 'SE2.nii.gz'
        SE3_volume = pathlib.Path(dir_name) / 'SE3.nii.gz'
        cmd = f'python3 pred3D_unet_med.py {WIGHT_PATH} {SE2_volume} {SE3_volume} -g={args.gpuid} -mask={mask_path} -yml={args.setting_yml_path} --save_dir={save_dir} --outfilename=00{patient_id}.nii.gz --stepscale=2 --class_num=4 --maskfile={mask_path}'
        subprocess.call(cmd.split())


if __name__ == "__main__":
    args = ParseArgs()
    yml = get_yml(args)
    main(args, yml)
