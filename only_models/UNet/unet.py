from tensorflow.python.keras.layers import Input, concatenate, Activation, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.python.keras.models import Model

def down(inputs, n_filters, filter_size = (3, 3), padding = 'same', kernel_initializer = 'he_normal',
         use_maxpool = True):
    conv = Conv2D(n_filters, filter_size, padding=padding, kernel_initializer=kernel_initializer)(inputs)
    conv = Activation('relu')(conv)
    conv = Conv2D(n_filters, filter_size, padding=padding, kernel_initializer=kernel_initializer)(conv)
    conv = Activation('relu')(conv)

    if(use_maxpool):
        max_pool = MaxPool2D(pool_size=(2, 2))(conv)
        return conv, max_pool
    return conv

def up(inputs, merged_inputs, n_filters,
       filter_size = (3, 3), padding = 'same',
       kernel_initializer = 'he_normal'):

    up = Conv2DTranspose(n_filters, kernel_size = filter_size,
                        strides = (2, 2), padding = padding, kernel_initializer = kernel_initializer)(inputs)
    merge_x = concatenate([up, merged_inputs], axis = -1)
    x = Conv2D(n_filters, kernel_size=filter_size, padding=padding, kernel_initializer=kernel_initializer)(merge_x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, kernel_size=filter_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = Activation('relu')(x)

    return x


def Unet(data_shape):
    inputs = Input(shape=data_shape)

    conv1, max_pool1 = down(inputs, 64) # (224, 224, 64)
    conv2, max_pool2 = down(max_pool1, 128) # (112, 112, 128)
    conv3, max_pool3 = down(max_pool2, 256) # (56, 56, 256)
    conv4, max_pool4 = down(max_pool3, 512) # (28, 28, 512)
    conv5 = down(max_pool4, 1024, use_maxpool=False) # (28, 28, 1024)

    up6 = up(conv5, conv4, 512)
    up7 = up(up6, conv3, 256)
    up8 = up(up7, conv2, 128)
    up9 = up(up8, conv1, 64)

    outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(up9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


test_model = Unet(448, 448, 1)
test_model.summary()
