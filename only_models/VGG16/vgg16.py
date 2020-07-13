from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def layer_block(x, num_filter, kernel_size = 3, s = 1, padding = 'same', iter_num = 2):
    for _ in range(iter_num):
        x = Conv2D(num_filter, kernel_size=kernel_size, strides=s,
                   padding=padding, activation = 'relu', )(x)

    x = MaxPooling2D((2, 2))(x)

    return x

def VGG16(data_shape, include_top = True):
    inputs = Input(shape=data_shape)

    x = layer_block(inputs, 64)
    x = layer_block(x, 128)
    x = layer_block(x, 256)
    x = layer_block(x, 512)
    x = layer_block(x, 512)

    if include_top:
        x = Flatten()(x)
        x = Dense(1024, activation = 'relu')(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dense(100, activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = x)

    return model

vgg16 = VGG16((224, 224, 3), include_top=True)
vgg16.summary()