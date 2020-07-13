from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from tensorflow.keras.models import Model

# No Dropout
# Just Simple DenseNet with Keras(TensorFlow 2.x)

# BN-ReLU-Conv(1 and 3)
def conv_block(x, n_filters):
    num_channel = n_filters * 4

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_channel, (1, 1), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3, 3), padding='same', use_bias=False)(x)

    return x

def transition_block(x, n_filters):
    x = Conv2D(n_filters, (1, 1), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((2, 2), strides=2)(x)

    return x

def dense_block(inputs, num_layers, n_filters, growth_rate=24):
    concat_layer = inputs

    for i in range(num_layers):
        x = conv_block(concat_layer, n_filters)
        concat_layer = Concatenate()([concat_layer, x])

    if (growth_rate):
        n_filters += growth_rate

    return concat_layer

def DenseNet(data_shape, n_filters, classes, growth_rate):
    # Make Model
    inputs = Input(shape=data_shape)
    init_layer = Conv2D(n_filters, (7, 7), strides=2, padding='same', use_bias=False)(inputs)
    init_layer = MaxPooling2D((3, 3), strides=2, padding='same')(init_layer)

    _transition = init_layer

    for i in range(3):
        num_layer = layer_list[i]

        # dense block
        _dense = dense_block(_transition, num_layer, n_filters, growth_rate)
        # transition block
        _transition = transition_block(_dense, n_filters)

        n_filters = int(n_filters * compression)

    final_dense = dense_block(_transition, layer_list[-1], n_filters)

    x = GlobalAveragePooling2D()(final_dense)
    outputs = Dense(classes, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

# init parameters
# DenseNet - 121
img_size = 224
n_filters = 64
growth_rate = 24
layer_list = [6, 12, 24, 16]

reduction = 0.
compression = 1. - reduction

classes = 1000

dense_model = DenseNet((img_size, img_size, 3), n_filters, classes, growth_rate)
dense_model.summary()