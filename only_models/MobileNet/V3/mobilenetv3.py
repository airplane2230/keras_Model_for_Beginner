import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten

from keras_radam import RAdam
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils.generic_utils import get_custom_objects

def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

def mish(x):
    return x * K.tanh(K.softplus(x))

get_custom_objects().update({'custom_activation': Activation(Hswish)})

def conv2d_block(inputs, filters, kernel, strides, is_use_bias = False, padding = 'same',
                activation = 'RE', name = None):
    x = Conv2D(filters, kernel, strides = strides, padding = padding, use_bias = is_use_bias)(inputs)
    x = BatchNormalization()(x)

    if activation == 'RE':
        x = ReLU(name=name)(x)
    elif activation == 'HS':
        x = Activation(Hswish, name=name)(x)
    elif activation == 'MS':
        x = Activation(mish, name = name)(x)
    else:
        raise NotImplementedError

    return x

def depthwise_block(inputs, kernel = (3, 3), strides = (1, 1), activation = 'RE',
                   is_use_se = True):
    x = DepthwiseConv2D(kernel_size = kernel, strides = strides, depth_multiplier = 1,
                       padding = 'same')(inputs)
    x = BatchNormalization()(x)

    if is_use_se:
        x = se_block(x)

    if activation == 'RE':
        x = ReLU()(x)
    elif activation == 'HS':
        x = Activation(Hswish)(x)
    elif activation == 'MS':
        x = Activation(mish)(x)
    else:
        raise NotImplementedError

    return x

def global_depthwise_block(inputs):
    assert inputs.shape[1] == inputs.shape[2]

    kernel_size = inputs.shape[1]
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1),
                        depth_multiplier=1, padding='valid')(inputs)

    return x

# squeeze and excite
def se_block(inputs, ratio = 4, pooling_type = 'avg'):
    filters = inputs._keras_shape[-1]
    se_shape = (1, 1, filters)

    if pooling_type == 'avg':
        se = GlobalAveragePooling2D()(inputs)
    elif pooling_type == 'depthwise':
        se = global_depthwise_block(inputs)
    else:
        raise NotImplementedError

    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    return multiply([inputs, se])

def bottleneck_block(inputs, out_dim, kernel, strides, expansion_dim,
                    is_use_bias = False, shortcut = True, is_use_se = True,
                    activation = 'RE', *args):
    with tf.name_scope('bottleneck_block'):
        bottleneck_dim = expansion_dim

        # pointwise
        x = conv2d_block(inputs, bottleneck_dim, kernel = (1, 1),
                         strides = (1, 1), is_use_bias = is_use_bias, activation = activation)
        # depthwise
        x = depthwise_block(x, kernel = kernel, strides = strides, is_use_se = is_use_se,
                           activation = activation)

        # pointwise
        x = Conv2D(out_dim, (1, 1), strides = (1, 1), padding = 'same')(x)
        x = BatchNormalization()(x)

        if shortcut and strides == (1, 1):
            in_dim = K.int_shape(inputs)[:-1]
            if in_dim != out_dim:
                ins = Conv2D(out_dim, (1, 1), strides = (1, 1), padding = 'same')(inputs)
                x = Add()([x, ins])
            else:
                x = Add()([x, ins])

    return x

def mobilenetv3(input_size = 224, output_type = 3, model_config = None,
                pooling_type = 'avg', activation = 'HS',
                classification_type = None,
               optimizer_type = 'adam',
               lr = None):
    inputs = Input(shape = (input_size, input_size, 3))

    net = conv2d_block(inputs, 16, kernel = (3, 3), strides = (2, 2), is_use_bias = False,
                      padding = 'same', activation = activation)

    config_list = model_config

    for config in config_list:
        net = bottleneck_block(net, *config)

    # ** final layers
    net = conv2d_block(net, 480, kernel=(3, 3), strides=(1, 1),
                         is_use_bias=True, padding='same', activation = activation, name='output_map')

    if pooling_type == 'avg':
        net = GlobalAveragePooling2D()(net)
    elif pooling_type == 'depthwise':
        net = global_depthwise_block(net)
    else:
        raise NotImplementedError

    # ** shape=(None, channel) --> shape(1, 1, channel)
    pooled_shape = (1, 1, net._keras_shape[-1])

    net = Reshape(pooled_shape)(net)
    # net = Conv2D(480, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)

    if(classification_type == 'categorical'):
        net = Conv2D(output_type, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
        net = Flatten()(net)
        net = Activation('softmax')(net)
    elif(classification_type == 'binary'):
        net = Conv2D(1, (1, 1), strides=(1, 1))(net)
        net = Flatten()(net)
        net = Activation('sigmoid')(net)

    model = Model(inputs=inputs, outputs=net)

    if optimizer_type == 'adam':
        optimizer = Adam(lr = lr)
    else:
        optimizer = RAdam(learning_rate = lr)

    loss = 'binary_crossentropy' if classification_type == 'binary' else 'categorical_crossentropy'
    metrics = [keras.metrics.binary_accuracy] if classification_type == 'binary' else [keras.metrics.categorical_accuracy]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
