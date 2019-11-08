import subprocess
import pathlib
import os

patient_ids = [
       '115', '119', '121', '124', '126', '127', '134', '135', '136', '138', '140', '142', '145', '146', '147', '149', '150', '151', '152', '154', '157', '159'
]

exp_names = [
    '1104_75epochs'
]

for exp_name in exp_names:
    root_dir =f'../experiments/{exp_name}'
    os.makedirs(root_dir,exist_ok=True)

    save_dir = f'{root_dir}/res'
    os.makedirs(save_dir,exist_ok=True)
    weights_file = f'/home/kakeya/Desktop/higuchi/20191021/Keras/src/weights/20191104/weights-e050_unet_liver_tumor_and_cyst_3cls.hdf5'

    for patient_id in patient_ids:
        dir_name = f'/home/kakeya/Desktop/higuchi/data/00{patient_id}/'
        mask_path = f'/home/kakeya/Desktop/higuchi/data/00{patient_id}/kidney.nii.gz'

        #kid_volume = pathlib.Path(dir_name) / 'kidney.nii.gz'
        #CCRCC_volume = pathlib.Path(dir_name) / 'CCRCC.nii.gz'
        #cyst_volume = pathlib.Path(dir_name) / 'cyst.nii.gz'
        SE2_volume = pathlib.Path(dir_name) / 'SE2.nii.gz'
        SE3_volume = pathlib.Path(dir_name) / 'SE3.nii.gz'

        cmd = f'python3 pred3D_unet.py {weights_file} {SE2_volume} {SE3_volume} --save_dir={save_dir} --outfilename=00{patient_id}.nii.gz --stepscale=2 --class_num=4 --maskfile={mask_path}'
        subprocess.call(cmd.split())


