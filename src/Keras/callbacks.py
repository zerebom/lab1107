import pathlib
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np

class CustomizedLearningRateScheduler(Callback):
    def __init__(self,
                 monitor='val_loss',
                 patience=0,
                 decay=0.1,
                 verbose=1):
        self.monitor = monitor
        self.patience = patience
        self.decay = decay
        self.verbose = verbose

        # 最小値の履歴
        self.monitor_list = []
        # これまでの検証で最良の重み
        self.min_weights = None
        # 学習率を下げる前の余分に学習した epoch 数
        self.over_epochs = 0

    def on_epoch_begin(self, epoch, logs=None):
        # epoch1 のときは何もしない
        if len(self.monitor_list) == 0:
            return

        minimum = min(self.monitor_list)
        min_epoch = self.monitor_list.index(minimum)
        # 直前に最小値が更新されていた場合，最良の重みを保持して超過学習 epoch 数は 0
        if epoch - min_epoch == 1:
            self.min_weights = self.model.get_weights()
            self.over_epochs = 0
        # 最小値の epoch との差が patience を超えたら，重みを読み直して学習率を下げる
        elif epoch - min_epoch - self.over_epochs > self.patience:
            self.model.set_weights(self.min_weights)
            if self.verbose:
                print(f'\nCustomizedLearningRateScheduler reloads the best weights on validation so far. (Epoch: {min_epoch+1})')

            lr = float(K.get_value(self.model.optimizer.lr)) * self.decay
            if self.verbose:
                print(f'CustomizedLearningRateScheduler setting learning rate to {lr}.')
            K.set_value(self.model.optimizer.lr, lr)

            # 余分に学習したエポック数は除く
            self.over_epochs = epoch - min_epoch
            # 学習率を下げた分， patience を長くする
            self.patience /= self.decay


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        self.monitor_list.append(current)

class CometLogImageUploader(Callback):
    def __init__(self, experiment, dataset, upload_steps=1000, xy=48, z=16):
        self.experiment = experiment
        self.dataset = dataset
        self.upload_steps = upload_steps

        self.xy = xy
        self.sqrt_z = np.int(np.ceil(np.sqrt(z)))

    def on_batch_end(self, batch, logs=None):
        if batch % self.upload_steps != 0:
            return

        X, Y = self.dataset[batch]
        x, y = X[0:1], Y[0:1]
        p = self.model.predict(x)

        self.experiment.log_image(self._3d_image_flatten(x[0]), name='x', image_minmax=(-135,215))
        self.experiment.log_image(self._3d_image_flatten(y[0]), name='y', image_minmax=(   0,  1))
        self.experiment.log_image(self._3d_image_flatten(p[0]), name='p', image_minmax=(   0,  1))

    def _3d_image_flatten(self, image):
        # flatten = np.zeros((48*4, 48*4, image.shape[-1]))
        flatten = np.zeros((self.xy * self.sqrt_z, self.xy * self.sqrt_z, image.shape[-1]))
        for i in range(self.sqrt_z):
            for j in range(self.sqrt_z):
                flatten[self.xy * i : self.xy * (i+1), self.xy * j : self.xy * (j+1)] = image[..., self.sqrt_z * i + j, :]

        return flatten
