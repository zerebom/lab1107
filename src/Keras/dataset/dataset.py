import pathlib
import pandas as pd
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import random
import cv2
import SimpleITK as sitk


class Loader():
    def __init__(self, data_dir, patch_dir_name='patch'):
        self.data_dir = pathlib.Path(data_dir)
        self.patch_dir_name = patch_dir_name

    def __create_dataframe(self, id_list):
        data = []
        for patient_id in id_list:
            patient_dir = self.data_dir / f'00{patient_id}' / self.patch_dir_name
            images = sorted(patient_dir.glob('patch_image_*.npy'))
            labels = sorted(patient_dir.glob('patch_no_onehot_*.npy'))
            if len(images) == 0 or len(labels) == 0:
                print(f'{patient_id} is no data')
            for image, label in zip(images, labels):
                data.append(['image', patient_id, image])
                data.append(['label', patient_id, label])
        return pd.DataFrame(data, columns=['type', 'id', 'path'])

    def load_train(self, train_id_list):
        self.train_dataset = self.__create_dataframe(train_id_list)
        return self.train_dataset

    def load_valid(self, valid_id_list):
        self.valid_dataset = self.__create_dataframe(valid_id_list)
        return self.valid_dataset

    def load_test(self, test_id_list):
        self.test_dataset = self.__create_dataframe(test_id_list)
        return self.test_dataset

class Generator(Sequence):
    def __init__(self, dataset, batch_size=4, nclasses=2,
                 enable_random_crop=True, enable_random_flip=False, enable_random_norm=False,
                 crop_size=(48, 48, 32), threshold=400, weight_method=None,clip=None,single=True):
        self.dataset = dataset.copy()
        self.batch_size = batch_size
        self.nclasses = nclasses
        self.threshold = threshold
        self.weight_method = weight_method
        #[-79,304]
        self.clip=clip

        self.data_cnt = 0
        self.single=single
        self.indices = []
        for patient_id in self.dataset.id.unique():
            # ある患者のdataset
            patient_dataset = self.dataset[self.dataset.id == patient_id]
            #images,labels2つあるから÷2?
            count = patient_dataset.count().id//2
            #threshold=inf
            if count >= self.threshold:
                count = self.threshold
            self.indices += np.random.choice((patient_dataset.index // 2).unique(), count, replace=False).tolist()
            #全員分のデータ数をカウント
            self.data_cnt += count
        self.cnt = 0

        self.enable_random_crop = enable_random_crop
        self.enable_random_flip = enable_random_flip
        self.enable_random_norm = enable_random_norm
        self.crop_size = crop_size
        
        random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.data_cnt / self.batch_size))

    def _decode_binsys_labels(self,label_list)->np.array:
        '''
        args array_list 
        this method decode binary systems label by rule base.
        this method will replace by each experience. 
        the reference of this rule base was recorded in check_data.py
        '''
        tmp_arr = np.array(label_list)

        # lable 1+3 ,2+3 ,1+3+5 -> 3
        tmp_arr = np.where(tmp_arr > 5, 5, tmp_arr) 
        # label 1+2 -> 2
        tmp_arr = np.where(tmp_arr == 4, 3, tmp_arr)
        # decode binary system -> tens system if arr >0(has some label)
        label_arr= np.where(tmp_arr > 2, np.log2(tmp_arr - 1)+1, tmp_arr).astype(np.uint8)
        return label_arr

    def __getitem__(self, idx):
        image_dataset = self.dataset[self.dataset.type == 'image']
        label_dataset = self.dataset[self.dataset.type == 'label']

        idx_list = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        image_list = [np.load(str(image_dataset['path'].iloc[idx])) for idx in idx_list]
        label_list = [np.load(str(label_dataset['path'].iloc[idx])) for idx in idx_list]
        
        X = np.array(image_list)

        try:
            Y = self._decode_binsys_labels(label_list)
        except:
            Y=np.zeros(np.array(label_list).shape)
            print([str(label_dataset['path'].iloc[idx]) for idx in idx_list])
        #print('first',X.shape,Y.shape)
        X=X.astype(np.float32)


        # flagがTrueならaugmentをする。Xはクロップ、YはNorm
        X, Y = self.__augment(X, Y, self.enable_random_crop, self.__random_crop)
        X, Y = self.__augment(X, Y, self.enable_random_norm, self.__random_norm)
        Y = to_categorical(Y, num_classes=self.nclasses)
        #print('medium',X.shape,Y.shape)
        if self.clip:
            X=np.clip(self.clip[0],self.clip[1])
        if self.weight_method:
            Y = self.weight_method(X, Y, self.nclasses)
        if self.single:
            X=X[:,:,:,:,0]
            X=X[...,np.newaxis]

        return X, Y

    def __next__(self):
        result = self[self.cnt]
        self.cnt += 1
        return result

    def on_epoch_end(self):
        self.indices = []
        for patient_id in self.dataset.id.unique():
            patient_dataset = self.dataset[self.dataset.id == patient_id]
            count = patient_dataset.count().id//2
            if count >= self.threshold:
                count = self.threshold
            self.indices += np.random.choice((patient_dataset.index // 2).unique(), count, replace=False).tolist()
        random.shuffle(self.indices)
    # no use
    def __norm(self, x, w=350, l=40):
        x = np.clip(x, l - w//2, l + w//2)
        x = x - (l - w//2)
        x = x / w
        return x

    # use in crop_inner
    def __scale_array(self, array, scale_rate):
        image = sitk.GetImageFromArray(array)
        # 拡大縮小後のサイズを計算
        scaled_size = [round(s * scale_rate) for s in image.GetSize()]

        # パラメータの設定
        #  scaleTransform = sitk.ScaleTransform(3, [scale_rate] * 3)
        #  scaleTransform.SetCenter(np.array(image.GetSize())/2)
        affinTransform = sitk.AffineTransform(3)
        affinTransform.Scale([1/scale_rate] * 3)

        # 拡大縮小の実行
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetSize(scaled_size)
        resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
        resampleFilter.SetTransform(affinTransform)
        resampleFilter.SetOutputPixelType(image.GetPixelIDValue())
        scaled = resampleFilter.Execute(image)

        return sitk.GetArrayFromImage(scaled)

    def __random_crop_inner(self, array, crop_size, seed):
        # 拡大縮小の倍率や x と y で切り出し位置を統一するため， seed を与える
        np.random.seed(seed)
        # 1.2~0.8の範囲で拡大縮小
        scale_rate = (1.2 - 0.8) * np.random.rand() + 0.8
        # 拡大縮小
        scaled = self.__scale_array(array, scale_rate)
        scaled = scaled[..., np.newaxis]

        # 各軸における余白を計算
        margins = [s - c for s, c in zip(scaled.shape, crop_size)]
        # 各軸におけるランダムな切り出し位置を計算
        shifts = [np.random.randint(m + 1) for m in margins]

        # random crop
        if len(array.shape) == 3:
            cropped = scaled[shifts[0] : shifts[0]+crop_size[0],
                             shifts[1] : shifts[1]+crop_size[1],
                             shifts[2] : shifts[2]+crop_size[2]]
        else:
            cropped = scaled[shifts[0] : shifts[0]+crop_size[0],
                             shifts[1] : shifts[1]+crop_size[1],
                             shifts[2] : shifts[2]+crop_size[2], :]
        return cropped

    def __random_crop(self, x, y):
        np.random.seed()
        seed = np.random.randint(1024)
        cropped_x = self.__random_crop_inner(x, self.crop_size, seed)
        cropped_y = self.__random_crop_inner(y, self.crop_size, seed)

        return cropped_x, cropped_y

    def __random_norm(self, x, y):
        w = 350 + np.random.randint(-10, 10) * 10
        l =  40 + np.random.randint(-10, 10) * 5
        x = np.clip(x, l - w//2, l + w//2)
        x = x - (l - w//2)
        x = x / w
        return x, y

    def __augment(self, X, Y, flag, method):
        if flag:
            augmented_X = []
            augmented_Y = []
            for x, y in zip(X, Y):
                augmented_x, augmented_y = method(x, y)
                augmented_X.append(augmented_x)
                augmented_Y.append(augmented_y)
            X = np.array(augmented_X)
            Y = np.array(augmented_Y)
        #deconmption if Keras NO DECONPOTITION
        # return X,Y
        return np.squeeze(X), Y


















class SingleGenerator(Sequence):
    def __init__(self, dataset, batch_size=4, nclasses=2,
                 enable_random_crop=True, enable_random_flip=False, enable_random_norm=False,
                 crop_size=(48, 48, 32), threshold=400, weight_method=None):
        self.dataset = dataset.copy()
        self.batch_size = batch_size
        self.nclasses = nclasses
        self.threshold = threshold
        self.weight_method = weight_method

        self.data_cnt = 0
        self.indices = []
        for patient_id in self.dataset.id.unique():
            # ある患者のdataset
            patient_dataset = self.dataset[self.dataset.id == patient_id]
            #images,labels2つあるから÷2?
            count = patient_dataset.count().id//2
            #threshold=inf
            if count >= self.threshold:
                count = self.threshold
            self.indices += np.random.choice((patient_dataset.index // 2).unique(), count, replace=False).tolist()
            #全員分のデータ数をカウント
            self.data_cnt += count
        self.cnt = 0

        self.enable_random_crop = enable_random_crop
        self.enable_random_flip = enable_random_flip
        self.enable_random_norm = enable_random_norm
        self.crop_size = crop_size
        
        random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.data_cnt / self.batch_size))

    def _decode_binsys_labels(self,label_list)->np.array:
        '''
        args array_list 
        this method decode binary systems label by rule base.
        this method will replace by each experience. 
        the reference of this rule base was recorded in check_data.py
        '''
        tmp_arr = np.array(label_list)
        # lable 1+3 ,2+3 ,1+3+5 -> 3
        tmp_arr = np.where(tmp_arr > 5, 5, tmp_arr) 
        # label 1+2 -> 2
        tmp_arr = np.where(tmp_arr == 4, 3, tmp_arr)
        # decode binary system -> tens system if arr >0(has some label)
        label_arr= np.where(tmp_arr > 2, np.log2(tmp_arr - 1)+1, tmp_arr).astype(np.uint8)
        return label_arr

    def __getitem__(self, idx):
        image_dataset = self.dataset[self.dataset.type == 'image']
        label_dataset = self.dataset[self.dataset.type == 'label']

        idx_list = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        image_list = [np.load(str(image_dataset['path'].iloc[idx])) for idx in idx_list]
        label_list = [np.load(str(label_dataset['path'].iloc[idx])) for idx in idx_list]

        X = np.array(image_list)
        shape=list(X.shape)
        shape=shape[:-1]        

        X=X[:,:,:,:,1].reshape(shape[0],shape[1],shape[2],shape[3],1)

        Y = self._decode_binsys_labels(label_list)
        #print('first',X.shape,Y.shape)

        # flagがTrueならaugmentをする。Xはクロップ、YはNorm
        X, Y = self.__augment(X, Y, self.enable_random_crop, self.__random_crop)
        X, Y = self.__augment(X, Y, self.enable_random_norm, self.__random_norm)
        Y = to_categorical(Y, num_classes=self.nclasses)
        #print('medium',X.shape,Y.shape)

        if self.weight_method:
            Y = self.weight_method(X, Y, self.nclasses)
        

        return X, Y

    def __next__(self):
        result = self[self.cnt]
        self.cnt += 1
        return result

    def on_epoch_end(self):
        self.indices = []
        for patient_id in self.dataset.id.unique():
            patient_dataset = self.dataset[self.dataset.id == patient_id]
            count = patient_dataset.count().id//2
            if count >= self.threshold:
                count = self.threshold
            self.indices += np.random.choice((patient_dataset.index // 2).unique(), count, replace=False).tolist()
        random.shuffle(self.indices)
    # no use
    def __norm(self, x, w=350, l=40):
        x = np.clip(x, l - w//2, l + w//2)
        x = x - (l - w//2)
        x = x / w
        return x

    # use in crop_inner
    def __scale_array(self, array, scale_rate):
        image = sitk.GetImageFromArray(array)
        # 拡大縮小後のサイズを計算
        scaled_size = [round(s * scale_rate) for s in image.GetSize()]

        # パラメータの設定
        #  scaleTransform = sitk.ScaleTransform(3, [scale_rate] * 3)
        #  scaleTransform.SetCenter(np.array(image.GetSize())/2)
        affinTransform = sitk.AffineTransform(3)
        affinTransform.Scale([1/scale_rate] * 3)

        # 拡大縮小の実行
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetSize(scaled_size)
        resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
        resampleFilter.SetTransform(affinTransform)
        resampleFilter.SetOutputPixelType(image.GetPixelIDValue())
        scaled = resampleFilter.Execute(image)

        return sitk.GetArrayFromImage(scaled)

    def __random_crop_inner(self, array, crop_size, seed):
        # 拡大縮小の倍率や x と y で切り出し位置を統一するため， seed を与える
        np.random.seed(seed)
        # 1.2~0.8の範囲で拡大縮小
        scale_rate = (1.2 - 0.8) * np.random.rand() + 0.8
        # 拡大縮小
        scaled = self.__scale_array(array, scale_rate)
        scaled = scaled[..., np.newaxis]

        # 各軸における余白を計算
        margins = [s - c for s, c in zip(scaled.shape, crop_size)]
        # 各軸におけるランダムな切り出し位置を計算
        shifts = [np.random.randint(m + 1) for m in margins]

        # random crop
        if len(array.shape) == 3:
            cropped = scaled[shifts[0] : shifts[0]+crop_size[0],
                             shifts[1] : shifts[1]+crop_size[1],
                             shifts[2] : shifts[2]+crop_size[2]]
        else:
            cropped = scaled[shifts[0] : shifts[0]+crop_size[0],
                             shifts[1] : shifts[1]+crop_size[1],
                             shifts[2] : shifts[2]+crop_size[2], :]
        return cropped

    def __random_crop(self, x, y):
        np.random.seed()
        seed = np.random.randint(1024)
        cropped_x = self.__random_crop_inner(x, self.crop_size, seed)
        cropped_y = self.__random_crop_inner(y, self.crop_size, seed)

        return cropped_x, cropped_y

    def __random_norm(self, x, y):
        w = 350 + np.random.randint(-10, 10) * 10
        l =  40 + np.random.randint(-10, 10) * 5
        x = np.clip(x, l - w//2, l + w//2)
        x = x - (l - w//2)
        x = x / w
        return x, y

    def __augment(self, X, Y, flag, method):
        if flag:
            augmented_X = []
            augmented_Y = []
            for x, y in zip(X, Y):
                augmented_x, augmented_y = method(x, y)
                augmented_X.append(augmented_x)
                augmented_Y.append(augmented_y)
            X = np.array(augmented_X)
            Y = np.array(augmented_Y)
        #deconmption if Keras NO DECONPOTITION
        # return X,Y
        return X, Y
# class SingleGenerator(Generator):
#     '''using single layer'''
#     def __init__(self,dataset, batch_size=4, nclasses=2,
#                  enable_random_crop=True, enable_random_flip=False, enable_random_norm=False,
#                  crop_size=(48, 48, 32), threshold=400, weight_method=None):
#         super().__init__(dataset, batch_size, nclasses,
#             enable_random_crop, enable_random_flip, enable_random_norm,
#             crop_size, threshold, weight_method)

#     def __getitem__(self, idx):

#         image_dataset = self.dataset[self.dataset.type == 'image']
#         label_dataset = self.dataset[self.dataset.type == 'label']

#         idx_list = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
#         image_list = [np.load(str(image_dataset['path'].iloc[idx])) for idx in idx_list]
#         label_list = [np.load(str(label_dataset['path'].iloc[idx])) for idx in idx_list]

#         X = np.array(image_list)
#         print(X.shape)

#         #only using SE2
#         X=X[:,:,:,:,1].reshape(self.batch_size,60,60,20,1)
#         print(X.shape)
        
#         Y = self._decode_binsys_labels(label_list)
#         #print('first',X.shape,Y.shape)

#         # flagがTrueならaugmentをする。Xはクロップ、YはNorm
#         X, Y = self.__augment(X, Y, self.enable_random_crop, self.__random_crop)
#         X, Y = self.__augment(X, Y, self.enable_random_norm, self.__random_norm)
#         Y = to_categorical(Y, num_classes=self.nclasses)
#         #print('medium',X.shape,Y.shape)

#         if self.weight_method:
#             Y = self.weight_method(X, Y, self.nclasses)
        

#         return X, Y

#         def __augment(self, X, Y, flag, method):
#             if flag:
#                 augmented_X = []
#                 augmented_Y = []
#                 for x, y in zip(X, Y):
#                     augmented_x, augmented_y = method(x, y)
#                     augmented_X.append(augmented_x)
#                     augmented_Y.append(augmented_y)
#                 X = np.array(augmented_X)
#                 Y = np.array(augmented_Y)
#             #deconmption if Keras NO DECONPOTITION
#             # return X,Y
#             return np.squeeze(X), Y