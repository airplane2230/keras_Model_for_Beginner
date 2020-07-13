from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Multiply, Add, Reshape
from tensorflow.keras.models import Model

import numpy as np

def squeeze_and_excitation(init, filters, reduction_ratio = 16):
    x = GlobalAveragePooling2D()(init)
    x = Dense(filters // reduction_ratio, activation='relu')(x)
    x = Dense(filters, activation='sigmoid')(x)
    x = Reshape(target_shape=(1, 1, filters))(x)

    x = Multiply()([init, x])

    return x

def conv_block(x, filters, kernel_size, strides = (1, 1), padding = 'same', activation = 'relu'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x

def se_resnet_block(x, filters):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # make residual
    residual = Conv2D(filters=filters*4, kernel_size = (1, 1), strides=(1, 1), padding='same')(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = conv_block(x, filters, (3, 3), strides=(1, 1))
    x = conv_block(x, filters*4, (1, 1), strides=(1, 1), padding='same')

    x = squeeze_and_excitation(x, filters*4, reduction_ratio = 16)

    x = Add()([x, residual])

    return x

def SEResNet(data_shape, depth_list, init_filters = 64):
    inputs = Input(shape=data_shape)

    # init layers
    x = Conv2D(init_filters, (7, 7), strides=2, activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=2)(x)

    # increased filter size: 1, 2, 4, 8, ...
    for i, depth in enumerate(depth_list):
        filter_num = init_filters * np.power(2, i)

        for _ in range(depth):
            x = se_resnet_block(x, filter_num)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Classifier
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1000, activation = 'softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

data_shape = (224, 224, 3)

model = SEResNet(data_shape, depth_list=[3, 4, 6, 3], init_filters=64)
model.summary()


