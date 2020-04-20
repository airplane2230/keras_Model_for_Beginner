from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def depthwiseconv_block(inputs, n_filters, strides, alpha = 1.0):
    n_filters = int(n_filters * alpha)

    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides,
                        padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(n_filters, kernel_size=(1, 1), strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def conv_block(inputs, n_filters, strides, alpha = 1.0):
    n_filters = int(n_filters * alpha)

    x = Conv2D(n_filters, kernel_size=(3, 3), strides=strides,
               padding = 'same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def MobileNetV1(data_shape):
    inputs = Input(shape=data_shape, name='inputs')

    layer_list = [
        (64, (1, 1)),
        (128, (2, 2)),
        (128, (1, 1)),
        (256, (2, 2)),
        (256, (1, 1)),
        (512, (2, 2)),
        *[(512, (1, 1)) for _ in range(5)],
        (1024, (2, 2)),
        (1024, (1, 1))
    ]

    x = conv_block(inputs, n_filters=32, strides=(2, 2), alpha=1.0)

    for (n_filters, strides) in layer_list:
        x = depthwiseconv_block(x, n_filters, strides, alpha=1.0)

    x = GlobalAveragePooling2D(name = 'GAP')(x)
    x = Dense(1000, activation = 'softmax', name='outputs')(x)

    model = Model(inputs = inputs, outputs = x)

    return model

# model = MobileNetV1((224, 224, 3))
# model.summary()