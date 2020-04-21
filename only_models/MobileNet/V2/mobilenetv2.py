from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import Input, Add, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model

def conv_block(inputs, n_filters, kernel_size, strides):
    x = Conv2D(n_filters, kernel_size, padding = 'same', strides=strides)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    return x

def bottleneck(inputs, n_filters, kernel_size, strides, t, is_add = False):
    exp_filters = int(inputs.shape[-1] * t)
    print(exp_filters, n_filters, inputs.shape[-1], inputs.shape)

    x = conv_block(inputs, exp_filters, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel_size=kernel_size, strides=(strides, strides), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    x = Conv2D(n_filters, (1, 1), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)

    if(is_add):
        x = Add()([x, inputs])

    print(inputs.shape, x.shape)
    print('\n')

    return x

def inverted_residual_block(inputs, n_filters, kernel_size, strides, t, n):
    x = bottleneck(inputs, n_filters, kernel_size, strides, t)

    for _ in range(1, n):
        x = bottleneck(x, n_filters, kernel_size, 1, t, is_add = True)

    return x

def MobileNetV2(data_shape):
    # n = [1, 2, 3, 4, 3, 3, 1]

    inputs = Input(shape=data_shape)

    init_conv = conv_block(inputs, 32, (3, 3), (2, 2))

    x = inverted_residual_block(init_conv, 16, (3, 3), strides = 1, t = 1, n = 1)
    x = inverted_residual_block(x, 24, (3, 3), strides = 2, t = 6, n = 2)
    x = inverted_residual_block(x, 32, (3, 3), strides= 2, t = 6, n = 3)
    x = inverted_residual_block(x, 64, (3, 3), strides= 2, t = 6, n = 4)
    x = inverted_residual_block(x, 96, (3, 3), strides = 1, t = 6, n = 3)
    x = inverted_residual_block(x, 160, (3, 3), strides= 2, t = 6, n = 3)
    x = inverted_residual_block(x, 320, (3, 3), strides= 1, t = 6, n = 1)

    x = conv_block(x, 1280, (1, 1), (1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)

    outputs = conv_block(x, 1280, (1, 1), (1, 1))

    model = Model(inputs = inputs, outputs = outputs)

    return model

data_shape = (224, 224, 3)

model = MobileNetV2(data_shape)
model.summary()