from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Multiply, Reshape
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model

def hard_swish(x):
    return x * relu(x + 3.0, max_value=6.0) / 6.0

def relu6(x):
    return relu(x, max_value=6.0)

def activations(inputs, nl):
    if(nl == "HS"):
        x = hard_swish(inputs)
    elif(nl == "RE"):
        x = relu6(inputs)

    return x

def conv_block(inputs, n_filters, kernel_size, strides, nl):
    x = Conv2D(n_filters, kernel_size, padding = 'same', strides=strides)(inputs)
    x = BatchNormalization()(x)
    x = activations(x, nl)

    return x

def squeeze(inputs):
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation = 'relu')(x)
    x = Dense(input_channels, activation = 'hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x

def bottleneck(inputs, n_filters, kernel_size, strides, t, nl, is_add = False, is_squeeze = False):
    exp_filters = int(inputs.shape[-1] * t)

    x = conv_block(inputs, t, (1, 1), strides=1, nl=nl)

    x = DepthwiseConv2D(kernel_size=kernel_size, strides=(strides, strides), padding='same')(x)
    x = BatchNormalization()(x)
    x = activations(x, nl)

    if is_squeeze:
        x = squeeze(x)

    x = Conv2D(n_filters, (1, 1), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)

    if(is_add):
        x = Add()([x, inputs])

    return x

# MobilNetV3 - Small
def MobileNetV3(data_shape, classes = 1000, include_top = False):
    inputs = Input(data_shape)

    x = conv_block(inputs, 16, (3, 3), (2, 2), nl = "HS")

    x = bottleneck(x, 16, (3, 3), strides=1, t=16, nl="RE", is_squeeze=False)
    x = bottleneck(x, 24, (3, 3), strides=2, t=64, nl="RE", is_squeeze=False)
    x = bottleneck(x, 24, (3, 3), strides=1, t=72, nl="RE", is_squeeze=False)
    x = bottleneck(x, 40, (5, 5), strides=2, t=72, nl="RE", is_squeeze=True)
    x = bottleneck(x, 40, (5, 5), strides=1, t=120, nl="RE", is_squeeze=True)
    x = bottleneck(x, 40, (5, 5), strides=1, t=120, nl="RE", is_squeeze=True)
    x = bottleneck(x, 80, (3, 3), strides=2, t=240, nl="HS", is_squeeze=False)
    x = bottleneck(x, 80, (3, 3), strides=1, t=200, nl="HS", is_squeeze=False)
    x = bottleneck(x, 80, (3, 3), strides=1, t=184, nl="HS", is_squeeze=False)
    x = bottleneck(x, 80, (3, 3), strides=1, t=184, nl="HS", is_squeeze=False)
    x = bottleneck(x, 112, (3, 3), strides=1, t=480, nl="HS", is_squeeze=True)
    x = bottleneck(x, 112, (3, 3), strides=1, t=672, nl="HS", is_squeeze=True)
    x = bottleneck(x, 160, (5, 5), strides=2, t=672, nl="HS", is_squeeze=True)
    x = bottleneck(x, 160, (5, 5), strides=1, t=960, nl="HS", is_squeeze=True)
    x = bottleneck(x, 160, (5, 5), strides=1, t=960, nl="HS", is_squeeze=True)

    x = conv_block(x, 960, (1, 1), (1, 1), nl = "HS")
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 960))(x)
    x = Conv2D(1280, (1, 1), (1, 1))(x)
    x = activations(x, "HS")

    if include_top:
        x = Conv2D(classes, (1, 1), padding='same', activation = 'softmax')(x)
        x = Reshape((classes, ))(x)

    model = Model(inputs=inputs, outputs=x)

    return model

model = MobileNetV3((224, 224, 3))
model.summary()