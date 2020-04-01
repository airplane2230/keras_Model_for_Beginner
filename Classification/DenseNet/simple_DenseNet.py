from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from tensorflow.keras.models import Model


# No Dropout
# Just Simple DenseNet with Keras(TensorFlow 2.x)

# BN-ReLU-Conv(1 and 3)
def conv_block(x, num_filter):
    num_channel = num_filter * 4

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_channel, (1, 1), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filter, (3, 3), padding='same', use_bias=False)(x)

    return x


def transition_block(x):
    x = Conv2D(num_filter, (1, 1), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((2, 2), strides=2)(x)

    return x


def dense_block(inputs, num_layers, num_filter, growth_rate=24):
    concat_layer = inputs

    for i in range(num_layers):
        x = conv_block(concat_layer, num_filter)
        concat_layer = Concatenate()([concat_layer, x])

    if (growth_rate):
        num_filter += growth_rate

    return concat_layer


# init parameters
# DenseNet - 121
img_size = 224
num_filter = 64
growth_rate = 24
layer_list = [6, 12, 24, 16]

reduction = 0.
compression = 1. - reduction

classes = 1000

# Make Model
inputs = Input(shape=(img_size, img_size, 3))
init_layer = Conv2D(num_filter, (7, 7), strides=2, padding='same', use_bias=False)(inputs)
init_layer = MaxPooling2D((3, 3), strides=2, padding='same')(init_layer)

_transition = init_layer

for i in range(3):
    num_layer = layer_list[i]

    _dense = dense_block(_transition, num_layer, num_filter)
    _transition = transition_block(_dense)

    num_filter = int(num_filter * compression)

final_dense = dense_block(_transition, layer_list[-1], num_filter)

x = GlobalAveragePooling2D()(final_dense)
x = Dense(classes, activation='softmax')(x)

dense_model = Model(inputs=inputs, outputs=x)
# dense_model.summary()