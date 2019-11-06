#%% 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import glob 


# check_lesion each candidates
#%%
DATA_PATH='/home/kakeya/Desktop/higuchi/DNP/data'
sorted(list(Path(DATA_PATH).glob('*/CCRCC*')))[0].parent.name


#%%
df=pd.DataFrame(np.zeros((160,6)))
df.columns=['kidney','ccrcc','cyst','SE2','SE3','num']
for cid in range(1,160):
    ccrcc=Path(DATA_PATH+f'/00{str(cid).zfill(3)}/CCRCC.nii.gz').is_file()*1
    kidney=Path(DATA_PATH+f'/00{str(cid).zfill(3)}/kidney.nii.gz').is_file()*1
    cyst=Path(DATA_PATH+f'/00{str(cid).zfill(3)}/cyst.nii.gz').is_file()*1
    SE2=Path(DATA_PATH+f'/00{str(cid).zfill(3)}/SE2.nii.gz').is_file()*1
    SE3=Path(DATA_PATH+f'/00{str(cid).zfill(3)}/SE3.nii.gz').is_file()*1
    num=len(list(Path(DATA_PATH+f'/00{str(cid).zfill(3)}').glob('*')))
    df.loc[cid,:]=[kidney,ccrcc,cyst,SE2,SE3,num]
    df=df.astype(int)


df.to_csv('./20191011/output/EDA_result/lesion_list.csv')

#%%
len(list(Path(DATA_PATH).glob('*/tumor_48*')))


#%%
df[(df['kidney']==1)&((df['ccrcc']==0)&(df['cyst']==1))]


#%%
image_volume='/home/kakeya/Desktop/higuchi/DNP/data/00003/CCRCC.nii.gz'


#%%
label_path='/home/kakeya/Desktop/higuchi/DNP/data/00001/tumor_48x48x16/patch_no_onehot_21387.npy'
np.unique(np.load(label_path))

#%%
