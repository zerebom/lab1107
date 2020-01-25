import numpy as np
from pathlib import Path
import SimpleITK as sitk
import pandas as pd
import yaml
import argparse

'''
python3 ./src/Keras/dataset/making_deeds_data.py -d /home/kakeya/Desktop/higuchi/data -s /home/kakeya/ssd/deeds_data_130 -m
'''
def ParseArgs():
    # from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str,
                        default='/home/kakeya/ssd/deeds_data')
    
    parser.add_argument('-s', '--save_dir', type=str,
                        default='/home/kakeya/ssd/deeds_data')  
    parser.add_argument('-m','--make_niigz', action='store_true') 

    args = parser.parse_args()
    return args


        
class DeedsPreprocessor(object):
    '''
    Deedsを使うときの前処理を行うクラス。
    - スペーシング幅を調整したあとに解像度をpadding/cropで調整する。
    - マスクがある場所の周辺のみをスライスに使用する。どのスライスを採用するかは、DataFrameに保存する
    - ↑の調整をしたデータを新たにディレクトリを作って保存する。
    '''

    def __init__(self, data_dir, save_dir,st_path,df_path='./deeds_statistics2.csv', start_id=0, kidney='kidney.nii.gz',
                 ccrcc='CCRCC.nii.gz', cyst='cyst.nii.gz', SEs=['SE2.nii.gz', 'SE3.nii.gz']):
        """
        data_dir...前処理前nii.gzが格納されているディレクトリ
        data_dir...前処理後nii.gzが格納するディレクトリ
        st_path...統計データcsvのパス（事前に作っておく必要あり)
        start_id...前処理をどの患者Idから始めるか（途中で中断したときに用いる）
        その他...nii.gzの名前
        """

        self.data_dir = data_dir
        self.save_dir = save_dir
        self.kidney = kidney
        self.ccrcc = ccrcc
        self.cyst = cyst
        self.SEs = SEs
        self.st_df = pd.read_csv(st_path)
        self.start_id=start_id
        self.spacing = (0.7, 0.7, 2.0)
        self.df_path = df_path
        cols = ['index','count', 'count_CCR', 'count_cys', 'lumi_mean']
        # statistics_dfから必要なデータだけを抽出する。
        self.mini_df = self.st_df[cols].reset_index().fillna(0)
        

    @staticmethod
    def CropCenter(img: np.array, croped_size=512):
        """
        zyxに並んでいる必要がある。
        Spacing幅を変えたときにxyをcrop
        """
        if img.shape[1] < croped_size:
            raise ValueError('this img is lager than croped_size')
        if img.shape[0] == img.shape[1]:
            raise ValueError('this img is not square.')
        _, y, x = img.shape
        startx = x // 2 - (croped_size // 2)
        return img[:, startx:startx + croped_size, startx:startx + croped_size]

    @staticmethod
    def PadCenter(img: np.array, padded_size=(130,512,512)):
        """
        zyxに並んでいる必要がある。
        Spacing幅を変えたときにxyをpadding
        """
        if img.shape[1] > padded_size[1]:
            raise ValueError('this img is lager than padded_size')
        if img.shape[1] != img.shape[2]:
            raise ValueError('this img is not square.')

        hw_pad_size = padded_size[1] - img.shape[1]
        d_pad_size = padded_size[0] - img.shape[0]

        r_hw_pad = hw_pad_size // 2
        l_hw_pad = int(np.ceil(hw_pad_size / 2))
        t_d_pad = d_pad_size // 2
        u_d_pad = int(np.ceil(d_pad_size / 2))
        pad_arr= np.pad(img, ((t_d_pad, u_d_pad), (r_hw_pad, l_hw_pad), (r_hw_pad, l_hw_pad)), 'constant',constant_values=(-1024,-1024))
        return np.clip(pad_arr,-1024,1024)
        # return np.pad(img, ((t_d_pad, u_d_pad), (r_hw_pad, l_hw_pad), (r_hw_pad, l_hw_pad)), 'minimum')

    def get_data(self, path: str, image=True):
        """pathからスペーシング幅を決めて画像とarrayを返す"""
        if Path(path).is_file():
            img = sitk.ReadImage(str(path))
            # boolを渡して、interpolatorを変更している
            img = self.resampleImage(img, self.spacing, image=image)
            img_array = sitk.GetArrayFromImage(img)
            return True, img_array
        else:
            print(f'{path} is not exist.')
            return False, np.zeros((100, 100, 100))

    def save_niigz(self, save_path: str, array: np.array):
        # (z,x,y)の方向になるようにする。
        DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
        array = array.astype(np.int16)
        save_image = sitk.GetImageFromArray(array)
        save_image.SetSpacing(self.spacing)
        save_image.SetDirection(DIRECTION)
        if not save_path.parent.is_dir():
            save_path.parent.mkdir()
            print(save_path)
        sitk.WriteImage(save_image, str(save_path), True)

    @staticmethod
    def resampleImage(img, new_spacing, image=True):
        '''Spacing幅を変えるときのコード
            sitk.SetSpacingだけではnumpy.shapeは変更されないので注意。
        '''
        def calc_new_size():
            np_size = np.array(img.GetSize())
            np_cur_spacing = np.array(img.GetSpacing())
            np_new_spacing = np.array(new_spacing)

            np_new_size = np_size * np_cur_spacing / new_spacing
            return np_new_size.astype(int).tolist()

        origin = img.GetOrigin()
        size = img.GetSize()
        cur_spacing = img.GetSpacing()
        direction = img.GetDirection()

        new_size = calc_new_size()
        transform = sitk.Transform()                # default is 'IdentityTransform'
        # interpolator = sitk.sitkNearestNeighbor
        interpolator = sitk.sitkBSpline if image else sitk.sitkNearestNeighbor
        return sitk.Resample(img, new_size, transform, interpolator,
                             origin, new_spacing, direction, 0.0, img.GetPixelIDValue())

    @staticmethod
    def resampleMask(img, base_spacing, ref_img):
        '''Mask画像用のResamplingメソッド
            base_spacing:resampleImageに通す前のspacing幅
            ref_image:resampleImageに通した後のsitk.Image
        '''
        img.SetSpacing(base_spacing)                # change with the same spacing of reference
        transform = sitk.Transform()                # default is 'IdentityTransform'
        interpolator = sitk.sitkNearestNeighbor
        # interpolator = sitk.sitkBSpline
        return sitk.Resample(img, ref_img.GetSize(), transform, interpolator,
                             ref_img.GetOrigin(), ref_img.GetSpacing(),
                             ref_img.GetDirection(), 0.0, img.GetPixelIDValue())
    
    @staticmethod
    def profile_image(image: sitk):
        """ロードしたデータの統計量を標準出力する"""
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        size = image.GetSize()
        print(">> spacing={}  , direction={}, size={}".format(spacing, direction, size))

    @staticmethod
    def check_slice_edge(label: np.array):
        """3次元ラベル二値画像。でどのスライスからデータが存在しているかidxの始点終点を返す"""
        label = label.reshape(label.shape[0], -1)
        boollabel = np.all(label == 0, axis=1)
        st = np.where(boollabel.astype(int) == 0)[0][0]
        en = np.where(boollabel.astype(int) == 0)[0][-1]
        return st, en
    
    def get_data_paths(self, cid):
        '''患者IDを渡すと、保存先と読み出し元パスをリストで返す'''
        load_dir = Path(f'{self.data_dir}/{str(cid).zfill(5)}')
        save_dir = Path(f'{self.save_dir}/{str(cid).zfill(5)}')

        image_load_paths = [load_dir / p for p in [self.SEs[0], self.SEs[1]]]
        image_save_paths = [save_dir / p for p in [self.SEs[0], self.SEs[1]]]

        label_load_paths = [load_dir / p for p in [self.kidney, self.ccrcc, self.cyst]]
        label_save_paths = [save_dir / p for p in [self.kidney, self.ccrcc, self.cyst]]
        whole_save_paths = image_save_paths + label_save_paths

        return image_load_paths, image_save_paths, label_load_paths, label_save_paths, whole_save_paths
      
    def main(self):
        '''DOING:存在しないときの処理'''
        df = self.mini_df.copy()
        df['right'] = 0
        df['left'] = 0
        df['slices'] = 0
        df['spacing_xy'] = 0.0
        df['spacing_z'] = 0.0
        df['up_kid_slice'] = 0
        df['down_kid_slice'] = 0
        df['st'] = 0
        df['en'] = 0

        for i in range(3):
            df[f'origin_{i}'] = 0.0

        cids = df['index'].values
        print(cids)
        cids=cids[cids>self.start_id]
        df.reset_index(drop=True, inplace=True)
        for i, cid in enumerate(cids):
            print(i)
            count = 0

            load_dir = Path(f'{self.data_dir}/{str(cid).zfill(5)}')
            save_dir = Path(f'{self.save_dir}/{str(cid).zfill(5)}')

            image_load_paths = [load_dir / p for p in [self.SEs[0], self.SEs[1]]]
            image_save_paths = [save_dir / p for p in [self.SEs[0], self.SEs[1]]]

            label_load_paths = [load_dir / p for p in [self.kidney, self.ccrcc, self.cyst]]
            label_save_paths = [save_dir / p for p in [self.kidney, self.ccrcc, self.cyst]]
            whole_save_paths = image_save_paths + label_save_paths


            # パスがあるかどうかのboolとarrayのリスト
            image_datas = [self.get_data(path, image=True) for path in image_load_paths]
            label_datas = [self.get_data(path, image=False) for path in label_load_paths]
            whole_datas = image_datas + label_datas

            se2_arr = image_datas[0][1]
            ccr_arr = label_datas[1][1]
            kid_arr = label_datas[0][1]
            kid = label_datas[0][0]
            
            
            if len(image_datas) != 2 or not kid or len(label_datas) == 0:
                print(len(image_datas),len(label_datas))
                print(f'{cid} is not exist SEs or labels.')
                continue

            print('before_shape:', se2_arr.shape)

            whole_label_arr = np.copy(kid_arr)
            for exist, array in label_datas:
                if not exist:
                    continue
                else:
                    whole_label_arr = np.logical_or(whole_label_arr, array)

            # ラベルの始点・終点を調べている
            try:
                st, en = self.check_slice_edge(whole_label_arr)
            except:
                print(f'{cid} can`t find st,en')
                continue
                
            
            df.at[i, ['st', 'en']] = st, en

            for (exist, array), save_path in zip(whole_datas, whole_save_paths):
                if not exist:
                    continue
                imst = max(st, st - 10)
                imen = en + 10

                array = array[imst:imen, :, :]
                if count == 0:
                    print('raw_shape:', array.shape)
                if array.shape[2] > 512:
                    array = self.CropCenter(array)
                
                array = self.PadCenter(array)
                if count == 0:
                    print('proccessed shape:', array.shape)
                self.save_niigz(save_path, array)
                count += 1

            # ガンがどっちにあるかを調べている。
            right = ccr_arr[:, :, 256:]
            left = ccr_arr[:, :, :256]
            df.at[i, 'slices'] = ccr_arr.shape[0]
            df.at[i, 'right'] = 1 if right.sum() > 0 else 0
            df.at[i, 'left'] = 1 if left.sum() > 0 else 0

        df.to_csv(self.df_path)

class DeedsStatisticsRecoder(DeedsPreprocessor):
    '''
    Deedsで変形元、変形先のペアを見つけるための統計量を集計するクラス。
    DeedsPreprocessorのmainを入れ替えただけ。
    '''

    def __init__(self,data_dir,save_dir,st_path,df_path='./deeds_statistics.csv', \
                start_id=0, kidney='kidney.nii.gz',
                ccrcc='CCRCC.nii.gz', cyst='cyst.nii.gz', SEs=['SE2.nii.gz', 'SE3.nii.gz']):
                
        super(DeedsStatisticsRecoder,self).__init__(data_dir,save_dir,st_path,df_path,start_id,kidney,ccrcc,cyst,SEs)

    
    def recode_slice_data2df(self):
        '''deedsをする時に使う統計データを収集する'''
        df = self.mini_df.copy()
        #df初期化
        for col in ['right', 'left', 'slices', 'st', 'en']:
            df[col] = 0

        cids = self.mini_df['index'].values
        cids = cids[cids > self.start_id]
        for i, cid in enumerate(cids):
            image_load_paths, image_save_paths, label_load_paths,\
                label_save_paths, whole_save_paths = self.get_data_paths(cid)

            label_datas = [self.get_data(path, image=False) for path in label_load_paths]
            # 必須のパスが存在しているかどうか
            SE_EXIST = True if sum([Path(path).is_file() for path in image_load_paths]) == 2 else False

            ccr_arr = label_datas[1][1]
            kid_arr = label_datas[0][1]
            kid = label_datas[0][0]
            whole_label_arr = np.copy(kid_arr)

            if not SE_EXIST or not kid or len(label_datas) == 0:
                print(f'{cid} is not exist SEs or labels.')
                continue

            for exist, array in label_datas:
                if not exist:
                    continue
                else:
                    whole_label_arr = np.logical_or(whole_label_arr, array)

            st, en = self.check_slice_edge(whole_label_arr)

            df.at[i, ['st', 'en']] = st, en
            # ガンがどっちにあるかを調べている。
            right = ccr_arr[:, :, 256:]
            left = ccr_arr[:, :, :256]
            df.at[i, 'slices'] = ccr_arr.shape[0]
            df.at[i, 'right'] = 1 if right.sum() > 0 else 0
            df.at[i, 'left'] = 1 if left.sum() > 0 else 0

        df.to_csv(self.df_path)


if __name__ == "__main__":
    args=ParseArgs()
    if args.make_niigz:
        dp=DeedsPreprocessor(args.data_dir,args.save_dir,'/home/kakeya/Desktop/higuchi/20191107/output/csv/statistics.csv')
        dp.main()
    else:
        dsr=DeedsStatisticsRecoder(args.data_dir,args.save_dir,'/home/kakeya/Desktop/higuchi/20191107/output/csv/statistics.csv')
        dsr.recode_slice_data2df()