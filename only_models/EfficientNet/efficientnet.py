from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Dropout
from tensorflow.keras.layers import BatchNormalization, Lambda, Activation, Multiply
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model

import math

def round_filters(filters, width_coef, depth_div, min_depth):
    multiplier = float(width_coef)
    divisor = int(depth_div)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if(new_filters< 0.9 * filters):
        new_filters += divisor

    return int(new_filters)

def round_repeats(repeats, depth_div):
    multiplier = depth_div

    if not multiplier:
        return repeats

    return int(math.ceil(multiplier * repeats))

class EffArgs(object):
    def __init__(self, input_filters, output_filters, kernel_size,
                 strides, num_repeat, se_ratio, expand_ratio):
        self.input_filters = input_filters
        self.output_filters =  output_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio

def SEBlock(input_filters, se_ratio, expand_ratio):
    reduced_filters = max(1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    def block(inputs):
        x = Lambda(lambda a: K.mean(a, axis = [1, 2], keepdims=True))(inputs)
        x = Conv2D(reduced_filters, (1, 1), strides=1, padding = 'same')(x)
        x = tf.nn.swish(x)

        x = Conv2D(filters, (1, 1), strides=1, padding = 'same')(x)
        x = Activation('sigmoid')(x)
        out = Multiply()([x, inputs])

        return out

    return block

def MBConvBlock(input_filters, output_filters, kernel_size, strides,
                expand_ratio, se_ratio, id_skip, drop_connect_rate):
    is_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):
        if(expand_ratio != 1):
            x = Conv2D(filters, (1, 1), strides=1, padding = 'same')(inputs)
            x = BatchNormalization()(x)
            x = tf.nn.swish(x)
        else:
            x = inputs

        x = DepthwiseConv2D(kernel_size, strides, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = tf.nn.swish(x)

        if is_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio)(x)

        x = Conv2D(output_filters, (1, 1), strides=1, padding = 'same')(x)
        x = BatchNormalization()(x)

        if id_skip:
            if (strides == 1) and (input_filters == output_filters):
                if drop_connect_rate:
                    x = Dropout(drop_connect_rate)(x)
                    # x = DropConnect(drop_connect_rate)(x)

                x = Add()([x, inputs])

        return x

    return block

def EfficientNet(data_shape, width_coef, depth_div, min_depth, block_args_list,
                 drop_connect_rate = 0.):
    inputs = Input(shape=data_shape)

    x = Conv2D(round_filters(32, width_coef, depth_div, min_depth),
               kernel_size=(3, 3), strides=(2, 2), padding = 'same', use_bias = False)(inputs)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    for i, args in enumerate(block_args_list):
        args.input_filters = round_filters(args.input_filters, width_coef, depth_div, min_depth)
        args.output_filters = round_filters(args.output_filters, width_coef, depth_div, min_depth)
        args.num_repeat = round_repeats(args.num_repeat, depth_div)

        x = MBConvBlock(args.input_filters, args.output_filters, args.kernel_size,
                        args.strides, args.expand_ratio, args.se_ratio,
                        True, i * drop_connect_rate_per_block)(x)

        if args.num_repeat > 1:
            args.input_filters = args.output_filters
            args.strides = 1

        for _ in range(args.num_repeat - 1):
            x = MBConvBlock(args.input_filters, args.output_filters, args.kernel_size,
                            args.strides, args.expand_ratio, args.se_ratio,
                            True, drop_connect_rate_per_block * i)(x)

    x = Conv2D(round_filters(1280, width_coef, depth_div, min_depth),
               (1, 1), strides=1, padding = 'same')(x)
    x = BatchNormalization()(x)
    outputs = tf.nn.swish(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

block_args_list = [
    EffArgs(32, 16, 3, 1, 1, 0.25, 1),
    EffArgs(16, 24, 3, 2, 2, 0.25, 6),
    EffArgs(24, 40, 5, 2, 2, 0.25, 6),
    EffArgs(40, 80, 3, 2, 3, 0.25, 6),
    EffArgs(80, 112, 5, 1, 3, 0.25, 6),
    EffArgs(112, 192, 5, 2, 4, 0.25, 6),
    EffArgs(192, 320, 3, 1, 1, 0.25, 6)
]

model = EfficientNet((224, 224, 3), width_coef=1.0,
                     depth_div=1.0, min_depth=None,
                     block_args_list = block_args_list, drop_connect_rate=0.)
model.summary()