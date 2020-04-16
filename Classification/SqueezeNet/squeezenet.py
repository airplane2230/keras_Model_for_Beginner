# Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016).
# SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size. arXiv preprint arXiv:1602.07360.
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.models import Model

import numpy as np

sq1x1 = 'squeeze1x1'
exp1x1 = 'expand1x1'
exp3x3 = 'expand3x3'

def fire_module(x, fire_id, SR=0.125):
    base_e = 128
    incr = 128
    pct3x3 = 0.5
    freq = 2

    ei = base_e + (incr * np.floor((fire_id - 1) / freq))
    e1x1 = np.int(ei * pct3x3)
    e3x3 = np.int(ei * (1 - pct3x3))
    si = np.int(SR * (e1x1 + e3x3))
    s_id = 'fire' + str(fire_id + 1) + '/'

    x = Convolution2D(si, (1, 1), padding='valid', name=s_id + sq1x1, kernel_initializer = 'he_normal')(x)
    x = Activation('relu', name=s_id + 'relu' + sq1x1)(x)

    left = Convolution2D(e1x1, (1, 1), padding='valid', kernel_initializer='he_normal', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + 'relu' + exp1x1)(left)

    right = Convolution2D(e3x3, (3, 3), padding='same', kernel_initializer='he_normal', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + 'relu' + exp3x3)(right)

    x = concatenate([left, right], axis=3, name=s_id + 'concat')

    return x

# SqueezeNet with Simple-bypass
# In paper, Figure 2's Middle
def SqueezeNet(input_shape=(224, 224, 3), SR=0.125, output_type = 1000):
    inputs = Input(input_shape, name='inputs')

    conv1 = Convolution2D(96, (7, 7), strides=(2, 2), padding='same', name='conv1',
                          kernel_initializer='he_normal')(inputs)
    conv1 = Activation('relu', name='relu_conv1')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)

    fire2 = fire_module(maxpool1, fire_id=1, SR=SR)
    fire3 = fire_module(fire2, fire_id=2, SR=SR)
    add_1 = Add()([fire2, fire3])

    fire4 = fire_module(add_1, fire_id=3, SR=SR)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(fire4)
    fire5 = fire_module(maxpool4, fire_id=4, SR=SR)
    add_2 = Add()([maxpool4, fire5])

    fire6 = fire_module(add_2, fire_id=5, SR=SR)
    fire7 = fire_module(fire6, fire_id=6, SR=SR)
    add_3 = Add()([fire6, fire7])

    fire8 = fire_module(add_3, fire_id=7, SR=SR)
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(fire8)
    fire9 = fire_module(maxpool8, fire_id=8, SR=SR)
    add_4 = Add()([maxpool8, fire9])

    x = Dropout(0.5, name='drop9')(add_4)
    x = Convolution2D(256, (1, 1), padding='valid', kernel_initializer='he_normal', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, name = 'dense_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(128, name = 'dense_2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(output_type, name='dense_3', kernel_initializer='he_normal' )(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='squeezenet')

    return model

# model = SqueezeNet()
# model.summary()