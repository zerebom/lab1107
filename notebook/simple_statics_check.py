#%%
"""medのデータの統計量を取得するコード
最初の数カラムしか使ってない。
"""
import pandas as pd
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import cv2
from pathlib import Path
from tqdm import tqdm
# %%
"""kits2019のデータの統計量を取得するコード"""

candidate_list=[
   '001','002','003','004','005','006','007','008','009','010',
   '011','012','013','014','015','016','017','018','019','021',
   '022','023','024','025','026','028','029','030','031','032',
   '033','034','035','036','037','038','039','040','041','044',
   '045','046','047','051','053','055','057','058','060','061',
   '062','063','064','065','066','067','068','069','071','072',
   '073','074','076','077','078','079','082','083','086','088',
   '090','093','094','095','096','097','098','101','102','103',
   '104','105','107','109','112','113','115','117','118','119',
   '121','122','123','124','125','126','127','130','134','135',
   '136','138','140','142','145','146','147','149','150','151',
   '152','154','157','159'
]

image_list=['SE2','SE3']
label_list=['kidney','CCRCC','cyst']
ROOT_DIR='/home/kakeya/Desktop/higuchi/data'

def _read_niigz(path:Path)->np.array:
    if os.path.isfile(path):
        data = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(data)
    else:
        return np.zeros((10,10,10))

def extract_ROI(voxel_label: np.array, voxel_image=None, threshold=0):
    '''voxelデータの関心領域以外を除去する
        indexと切り取ったvoxelを返す    
    '''
    sliceIndex = []
    # 高さ方向の腎臓、腎臓がんの範囲特定
    for z in range(len(voxel_label[:,0, 0])):
        if np.where(voxel_label[z, :, :] != threshold, True, False).any():
            sliceIndex.append(z)
    return voxel_label[sliceIndex, :, :], voxel_image[sliceIndex, :, :], sliceIndex


all_df=pd.DataFrame(columns=['index'])
all_df['index']=[int(cid) for cid in candidate_list]
all_df=all_df.set_index('index')

st_df=pd.DataFrame(columns=['shape','count','lumi_mean','lumi_std','lumi_max','lumi_min'])

image_list=['SE2','SE3']
label_list=['kidney','CCRCC','cyst']
for image in image_list:
    for label in label_list:
        for cid in tqdm(candidate_list):
            cid_path=Path(ROOT_DIR)/f'00{cid.zfill(3)}'

            path = cid_path/f'{image}.nii.gz'
            im_arr=_read_niigz(path)

            path = cid_path/f'{label}.nii.gz'
            if os.path.isfile(path):
                label_arr=_read_niigz(path)
                label_arr,im_arr,sid = extract_ROI(label_arr,im_arr,threshold=0)
                slice_shape=label_arr.shape

                im_arr = np.where(label_arr == 1, im_arr, -1300)

                flat_im = im_arr.reshape(-1)
                flat_im = flat_im[flat_im > -1000]
                st_df.at[int(cid), 'shape'] = slice_shape
                st_df.at[int(cid), 'count'] = flat_im.shape[0]
                st_df.at[int(cid), 'lumi_mean'] = flat_im.mean()
                st_df.at[int(cid), 'lumi_std'] = flat_im.std()
                st_df.at[int(cid), 'lumi_max'] = flat_im.max()
                st_df.at[int(cid), 'lumi_min'] = flat_im.min()
            else:
                st_df.at[int(cid), 'shape'] = np.nan
                st_df.at[int(cid), 'count'] = np.nan
                st_df.at[int(cid), 'lumi_mean'] = np.nan
                st_df.at[int(cid), 'lumi_std'] = np.nan
                st_df.at[int(cid), 'lumi_max'] = np.nan
                st_df.at[int(cid), 'lumi_min'] = np.nan
        all_df=all_df.merge(st_df,how='left',suffixes=('',f'_{image}_{label[:3]}'),right_index=True ,left_index=True)
        

#%%