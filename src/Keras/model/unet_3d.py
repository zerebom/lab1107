# import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Cropping3D, Conv3DTranspose, MaxPool3D, concatenate, BatchNormalization, Dropout, Activation
from tensorflow.keras.utils import multi_gpu_model

class UNet3D(Model):
    def __init__(self, input_shape, nclasses, use_bn=True, use_dropout=True):
        # n =conv times
        x = Input(shape=input_shape, name='x')
        y, contract1 = UNet3D.__create_conv_block(x,   64, n=2, use_bn=use_bn, name='contract1')
        y, contract2 = UNet3D.__create_conv_block(y,  128, n=2, use_bn=use_bn, name='contract2')
        y, contract3 = UNet3D.__create_conv_block(y,  256, n=2, use_bn=use_bn, name='contract3')
        y, contract4 = UNet3D.__create_conv_block(y,  512, n=2, use_bn=use_bn, name='contract4', apply_pooling=False)

        if use_dropout:
            y = Dropout(0.5)(y)
            contract3 = Dropout(0.5)(contract3)
            contract2 = Dropout(0.5)(contract2)
            contract1 = Dropout(0.5)(contract1)

        y = UNet3D.__create_up_conv_block(y, contract3, 256, n=2, use_bn=use_bn, name='expand3')
        y = UNet3D.__create_up_conv_block(y, contract2, 128, n=2, use_bn=use_bn, name='expand2')
        y = UNet3D.__create_up_conv_block(y, contract1,  64, n=2, use_bn=use_bn, name='expand1')

        y = Conv3D(nclasses, (1,1,1), activation='softmax', padding='same', name=f'segment',
                   kernel_initializer='he_normal', bias_initializer='zeros')(y)

        super(UNet3D, self).__init__(inputs=x, outputs=y)

    @classmethod
    def __create_conv_block(cls, x, filters, n=2, use_bn=True, apply_pooling=True, name='convblock'):
        for i in range(1, n+1):
            x = Conv3D(filters, (3,3,3), padding='same', name=f'{name}_conv{i}',
                       kernel_initializer='he_normal', bias_initializer='zeros')(x)
            if use_bn:
                x = BatchNormalization(name=f'{name}_BN{i}')(x)
            x = Activation('relu', name=f'{name}_relu{i}')(x)

        conv_result = x

        if apply_pooling:
            x = MaxPool3D(pool_size=(2,2,2), name=f'{name}_pooling')(x)

        return x, conv_result

    @classmethod
    def __create_up_conv_block(cls, x, contract_part, filters, n=2, use_bn=True, name='upconvblock'):
        # upconv x
        x = Conv3DTranspose(x.shape[-1].value, (2,2,2), strides=(2,2,2), padding='same', use_bias=False, name=f'{name}_upconv',
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)

        # concatenate contract4 and x
        x = concatenate([contract_part, x])

        # conv x 2 times
        for i in range(1, n+1):
            x = Conv3D(filters, (3,3,3), padding='same', name=f'{name}_conv{i}',
                       kernel_initializer='he_normal', bias_initializer='zeros')(x)
            if use_bn:
                x = BatchNormalization(name=f'{name}_BN{i}')(x)
            x = Activation('relu', name=f'{name}_relu{i}')(x)

        return x

if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # from tensorflow.keras.utils.vis_utils import model_to_dot
    # from tensorflow.keras. import model_to_dot

    model = UNet3D((96, 96, 32, 1), 2)
    import ipdb; ipdb.set_trace()
    with open('unet_3d_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    # with open('unet_3d_model.svg', 'wb') as f:
    #     f.write(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
