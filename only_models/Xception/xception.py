from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, Activation, Dense
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size, padding = 'same',
               strides = 1, activation = 'relu'):
    x = Conv2D(filters, kernel_size, padding = padding,
               strides = strides)(x)
    x = BatchNormalization()(x)

    if activation:
        x = Activation(activation)(x)

    return x

def sepconv_block(x, filters, kernel_size, padding = 'same',
                  strides = 1, activation = 'relu', depth_multiplier = 1):
    x = SeparableConv2D(filters, kernel_size, padding = padding,
                        strides = strides, depth_multiplier = depth_multiplier)(x)
    x = BatchNormalization()(x)

    if activation:
        x = Activation(activation)(x)

    return x

def Xception(data_shape):
    '''
        Xception: Deep Learning with Depthwise Separable Convolutions
    '''

    inputs = Input(shape = data_shape)
    
    # Entry flow
    x = conv_block(inputs, 32, (3, 3), strides = 2)
    x = conv_block(x, 64, (3, 3))
        
    for filters in [128, 256, 728]:
        residual = conv_block(x, filters, (1, 1), strides = 2, activation = None)

        x = Activation('relu')(x)
        x = sepconv_block(x, filters, (3, 3))
        x = sepconv_block(x, filters, (3, 3), activation = None)
        x = MaxPooling2D((3, 3), padding = 'same', strides = 2)(x)

        x = Add()([residual, x])

    # Middle flow
    for _ in range(8):
        residual = x

        x = Activation('relu')(x)
        x = sepconv_block(x, 728, (3, 3))
        x = sepconv_block(x, 728, (3, 3))
        x = sepconv_block(x, 728, (3, 3), activation = None)

        x = Add()([residual, x])

    # Exit flow
    residual = conv_block(x, 1024, (1, 1), strides = 2, activation = None)

    x = Activation('relu')(x)
    x = sepconv_block(x, 728, (3, 3))
    x = sepconv_block(x, 1024, (3, 3), activation = None)
    x = MaxPooling2D((3, 3), padding = 'same', strides = 2)(x)

    x = Add()([residual, x])

    x = sepconv_block(x, 1536, (3, 3))
    x = sepconv_block(x, 2048, (3, 3))

    x = GlobalAveragePooling2D()(x)
    
    # Classifier: option
    x = Dense(2048, activation = 'relu')(x)
    outputs = Dense(1000, activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = outputs)

    return model

model = Xception((299, 299, 3))
model.summary()