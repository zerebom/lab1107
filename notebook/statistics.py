#%%
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
# %%
"""kits2019のデータの統計量を取得するコード"""

def extract_ROI(voxel_label: np.array, voxel_image=None, threshold=0):
    '''voxelデータの関心領域以外を除去する
        indexと切り取ったvoxelを返す    
    '''
    sliceIndex = []

    # 高さ方向の腎臓、腎臓がんの範囲特定
    for x in range(len(voxel_label[0, 0, :])):
        if np.where(voxel_label[:, :, x] != threshold, True, False).any():
            sliceIndex.append(x)
    return voxel_image[:, :, sliceIndex], voxel_label[:, :, sliceIndex], sliceIndex


def add_statistics_df(in_df, i, arr_list):
    '''対象領域のflattenしたnumpy_arrayの統計量を取り、dfに追加する。
    予め、指定した列を持つDFを作成しておくこと
    '''
    in_df.at[i, 'count'] = arr_list.shape[0]
    in_df.at[i, 'lumi_mean'] = arr_list.mean()
    in_df.at[i, 'lumi_std'] = arr_list.std()
    in_df.at[i, 'lumi_max'] = arr_list.max()
    in_df.at[i, 'lumi_min'] = arr_list.min()
    return in_df


#%%

class PlotStatistics(object):
    ''' 
    統計量を取得し、histgramと dataframeを作成するクラス。
    使用する患者や、ラベルの数が動的に変化しても大丈夫なようにクラスにした
    candidate_list=['001','003'...]
    image_list=['SE2,'SE3']
    label_list=['CCRCC','cyst']

    '''

    def __init__(self, candidate_list, image_list, label_list, ROOT_DIR,img_save_dir,df_save_path):
        self.candidate_list = candidate_list
        self.image_list = image_list
        self.label_list = label_list
        self.ROOT_DIR = ROOT_DIR
        self.all_list = image_list + label_list
        self.image_dist_lim = [-1024, 1024]
        self.label_dist_lim = [-512, 512]
        self.img_save_dir=img_save_dir
        self.df_save_path=df_save_path

        cols = ['count', 'lumi_mean', 'lumi_std', 'lumi_max', 'lumi_min']
        self.all_df = pd.DataFrame(columns=cols)
        #add candidate Id
        self.all_df['candidate_id']=[x.zfill(3) for x in candidate_list]

        self.label_df_dic = {}
        for label in label_list:
            self.label_df_dic[label] = pd.DataFrame(columns=cols)

    def _read_niigz(self, cid:str)->np.array:
        self.image_dict, self.label_dict = {}, {}
        for name in self.image_list:
            path = Path(self.ROOT_DIR) / f'00{str(cid).zfill(3)}' / f'{name}.nii.gz'
            if os.path.isfile(path):
                data = sitk.ReadImage(str(path))
                self.image_dict[name] = sitk.GetArrayFromImage(data)
            else:
                self.image_dict[name]=np.zeros((10,10,10))

        for name in self.label_list:
            path = Path(self.ROOT_DIR) / f'00{str(cid).zfill(3)}' /  f'{name}.nii.gz'
            if os.path.isfile(path):        
                data = sitk.ReadImage(str(path))
                self.label_dict[name] =sitk.GetArrayFromImage(data)
            else:
                self.image_dict[name]=np.zeros((10,10,10))

    def _extract_ROI(self, voxel_kidney: np.array, voxel_image=None, threshold=0):
        '''voxelデータの関心領域以外を除去する
            indexと切り取ったvoxelを返す    
        '''
        sliceIndex = []
        # 高さ方向の腎臓、腎臓がんの範囲特定
        for x in range(len(voxel_kidney[0, 0, :])):
            if np.where(voxel_kidney[:, :, x] != threshold, True, False).any():
                sliceIndex.append(x)
        return voxel_image[:, :, sliceIndex], voxel_kidney[:, :, sliceIndex], sliceIndex

    def _add_statistics_df(self, in_df, i, arr_list):
        '''対象領域のflattenしたnumpy_arrayの統計量を取り、dfに追加する。
        予め、指定した列を持つDFを作成しておくこと
        '''
        in_df.at[i, 'count'] = arr_list.shape[0]
        in_df.at[i, 'lumi_mean'] = arr_list.mean()
        in_df.at[i, 'lumi_std'] = arr_list.std()
        in_df.at[i, 'lumi_max'] = arr_list.max()
        in_df.at[i, 'lumi_min'] = arr_list.min()
        return in_df

    def plot_grid_slice(self, images, masks, max_images=128, grid_width=16, cid=1,label='label'):
        grid_height = int(max_images / grid_width)
        fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
        for i in range(max_images):
            ax = axs[int(i / grid_width)-1, i % grid_width]
            ax.imshow(images[:, :, i], cmap="bone")
            ax.imshow(masks[:, :, i], alpha=0.3, cmap="Reds", vmin=0, vmax=2)
            ax.axis('off')
        plt.suptitle("Chest X-rays, Red: Pneumothorax.")
        plt.savefig(f"{self.img_save_dir}/grid_{label}_{cid}.png", format='png', dpi=300)

    def only_makedf_main(self):
        count=0
        for i, cid in enumerate(self.candidate_list):
            print(cid,end=',')
            self._read_niigz(cid)
            if not 'kidney' in self.label_dict:
                print(cid,'has no kidney')
                continue

            # 時相ごと
            for _, image in enumerate(self.image_list):
                # has kidney idx slice
                im, kid_mask, sIdx = self._extract_ROI(self.label_dict['kidney'], self.image_dict[image])
                # ラベルごと(ccrcc,cyst)
                for k, label in enumerate(self.label_list):
                    if label in self.label_dict:
                        #対象ラベルの腎臓を含む領域
                        slabel=self.label_dict[label][:,:,sIdx]
                        #no label area define 1300
                        im1 = np.where(slabel == 1, im, -1300)

                        flatten_im_all = im.reshape(-1)
                        flatten_im1 = im1[im1 > -1000]

                    # すべてのデータを記録
                    if k == 0:
                        add_statistics_df(self.all_df, i, flatten_im_all)
                    else:
                        add_statistics_df(self.label_df_dic[label], i, flatten_im1)

            count+=1

            # labelごとのにdfをmergeしている
        for label, label_df in self.label_df_dic.items():
            self.all_df = self.all_df.join(label_df, lsuffix='', rsuffix=f'_{label}')
        self.all_df.to_csv(self.df_save_path)

    def main(self,grid_plot_flg=True):
        # 患者ごと
        os.makedirs(self.img_save_dir, exist_ok=True)
        count=0
        for i, cid in enumerate(self.candidate_list):
            print(cid,end=',')
            self._read_niigz(cid)
            if not 'kidney' in self.label_dict:
                print(cid,'has no kidney')
                continue

            if count % 10 == 0:
                fig, axs = plt.subplots(10, len(self.label_list) + 1, figsize=(12, 20))
            # 時相ごと
            for _, image in enumerate(self.image_list):
                # has kidney idx slice
                im, kid_mask, sIdx = self._extract_ROI(self.label_dict['kidney'], self.image_dict[image])
                # ラベルごと(ccrcc,cyst)
                for k, label in enumerate(self.label_list):
                    #対象ラベルの腎臓を含む領域
                    slabel=self.label_dict[label][:,:,sIdx]

                    if grid_plot_flg:
                        self.plot_grid_slice(im, slabel, max_images=len(sIdx), cid=i,label=label)
                    im1 = np.where(slabel == 1, im, -1300)

                    flatten_im_all = im.reshape(-1)
                    flatten_im1 = im1[im1 > -1000]

                    # すべてのデータを記録
                    if k == 0:
                        sns.distplot(flatten_im_all, kde=False, ax=axs[i % 10, k], label=image)
                        axs[i % 10, k].set_title(f'cid:{i},type:all')
                        axs[i % 10, k].set_xlim(self.image_dist_lim)
                        add_statistics_df(self.all_df, i, flatten_im_all)

                    # ラベル内に含まれるデータだけを記録
                    sns.distplot(flatten_im1, kde=False, ax=axs[i % 10, k + 1], label=image)
                    axs[i % 10, k + 1].set_title(f'cid:{i},type:{label}')
                    axs[i % 10, k + 1].set_xlim(self.label_dist_lim)
                    add_statistics_df(self.label_df_dic[label], i, flatten_im1)

            if count % 10 == 9:
                plt.tight_layout()
                plt.savefig(f"{self.img_save_dir}/{i-9}_{i}.png", format='png', dpi=300)
                plt.cla()

            count+=1

            # labelごとのにdfをmergeしている
        for label, label_df in self.label_df_dic.items():
            self.all_df = self.all_df.join(label_df, lsuffix='', rsuffix=f'_{label}')
        self.all_df.to_csv(self.df_save_path)
#%%
# candidate_list=['002','004','005','006','009','010','011','012','013','014',
# '015','017','018','019','030','031','037','039','040','045',
# '047','053','055','061','062','063','064','065','066','067',
# '068','069','073','074','076','078','079','082','093','094',
# '095','097','098','101','102','103','104','113','117','118',
# '122','125','130','135','136','138','142','145','146','147',
# '149','151','152','154','157']

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
   '152','154','157','159']

image_list=['SE2','SE3']
label_list=['kidney','CCRCC','cyst']
# ROOT_DIR='/home/kakeya/Desktop/higuchi/DNP/data'
ROOT_DIR='/home/kakeya/Desktop/higuchi/data'

img_save_dir='./output'
df_save_path='./output/statistics_1021.csv'
ploter=PlotStatistics(candidate_list,image_list,label_list, ROOT_DIR,img_save_dir,df_save_path)

ploter.only_makedf_main()

#%%
ploter.label_dict
#%%
new_candidate_list=[ 1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  21,  22,  23,  24,  25,  26,  28,
        29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        44,  45,  46,  47,  51,  53,  55,  57,  58,  60,  61,  62,  63,
        64,  65,  66,  67,  68,  69,  71,  72,  73,  74,  76,  77,  78,
        79,  82,  83,  86,  88,  90,  93,  94,  95,  96,  97,  98, 101,
       102, 103, 104, 105, 107, 109, 112, 113, 115, 117, 118, 119, 121,
       122, 123, 124, 125, 126, 127, 130, 134, 135, 136, 138, 140, 142,
       145, 146, 147, 149, 150, 151, 152, 154, 157, 159]

len([str(z).zfill(3) for z in new_candidate_list])


#%%
