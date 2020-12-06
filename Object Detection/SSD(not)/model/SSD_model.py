import tensorflow as tf

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Flatten, add, Activation, Concatenate
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model

from utils.PriorBox import PriorBox

def _conv_block(input_tensor, s,
                c, n, t, stage):
    # s : strides
    # c : channel
    # n : iter
    # t : factor

    conv_name_base = 'res' + str(stage) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'
    x = None

    # Strides == 1 block
    if (s == 1):
        shortcut = None
        for i in range(n):
            x = Conv2D(c, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a_' + str(i))(input_tensor)
            x = BatchNormalization(name=bn_name_base + '2a_' + str(i))(x)
            x = Activation('relu')(x)
            x = DepthwiseConv2D((3, 3), depth_multiplier=t, padding='same', name=conv_name_base + '2bdepth_' + str(i))(
                x)
            x = BatchNormalization(name=bn_name_base + '2b_' + str(i))(x)
            x = Activation('relu')(x)
            x = Conv2D(c, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c_' + str(i))(x)
            x = BatchNormalization(name=bn_name_base + '2c_' + str(i))(x)
            x = Activation('linear')(x)

            if (shortcut is None):
                shortcut = Conv2D(c, (1, 1), strides=s, padding='same', kernel_initializer='he_normal',
                                  name=conv_name_base + '1_' + str(i))(input_tensor)
            else:
                shortcut = Conv2D(c, (1, 1), strides=s, padding='same', kernel_initializer='he_normal',
                                  name=conv_name_base + '1_' + str(i))(x)

            x = add([x, shortcut], name='c_add_' + str(stage) + '_' + str(i))
    # Strides == 2 block
    elif (s == 2):
        for i in range(n):
            x = Conv2D(c, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a_' + str(i))(input_tensor)
            x = BatchNormalization(name=bn_name_base + '2a_' + str(i))(x)
            x = Activation('relu')(x)
            x = DepthwiseConv2D((3, 3), strides=s, depth_multiplier=t, padding='same',
                                name=conv_name_base + '2bdepth_' + str(i))(x)
            x = BatchNormalization(name=bn_name_base + '2b_' + str(i))(x)
            x = Activation('relu')(x)
            x = Conv2D(c, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c_' + str(i))(x)
            x = BatchNormalization(name=bn_name_base + '2c_' + str(i))(x)
            x = Activation('linear')(x)
    return x

def _SSD_Conv_fc(x, filter, kernel_size, strides = (1, 1)):
    net = Conv2D(filter, kernel_size = kernel_size, strides = strides)(x)
    x = BatchNormalization()(net)
    x = Activation('relu')(x)

    print(x.shape, 'SSD_Conv_fc')

    return x, net

def _SSD_Conv(x, filter, kernel_size, strides):
    x = Conv2D(filter // 2, kernel_size = (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D(kernel_size=kernel_size, strides = strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    net = Conv2D(filter, (1, 1), padding='same')(x)
    x = BatchNormalization()(net)
    x = Activation('relu')(x)

    print(x.shape, 'SSD_Conv')

    return x, net

def _detections(x, feature_map_num, bbox_num, min_s, max_s, num_classes):
    # bbox location
    mbox_loc = Conv2D(bbox_num * 4, (3, 3), padding='same', name=str(feature_map_num) + '_mbox_loc')(x)
    mbox_loc_flat = Flatten(name = str(feature_map_num) + '_mbox_loc_flat')(mbox_loc)

    # class confidence
    mbox_conf = Conv2D(bbox_num * num_classes, (3, 3), padding = 'same', name = str(feature_map_num) + '_mbox_conf')(x)
    mbox_conf_flat = Flatten(name = str(feature_map_num) + '_mbox_conf_flat')(mbox_conf)

    # Anchor box candidate
    mbox_priorbox = PriorBox(min_s, max_s, feature_map_num, bbox_num, name = str(feature_map_num) + 'mbox_prior_box')(x)

    return mbox_loc_flat, mbox_conf_flat, mbox_priorbox

def SSD(input_shape, num_classes):
    img_input = Input(shape=input_shape)

    # MobileNetV2
    # init
    print('model init')
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same',
               kernel_initializer='he_normal', name='conv1')(img_input)  # (112, 112, 32)

    mobile_conv1 = _conv_block(x, c=16, s=1, n=1, t=1, stage=2)  # (112, 112, 16)
    mobile_conv2 = _conv_block(mobile_conv1, c=24, s=2, n=2, t=6, stage=3)  # (56, 56, 24)
    mobile_conv3 = _conv_block(mobile_conv2, c=32, s=2, n=3, t=6, stage=4)  # (28, 28, 32)
    mobile_conv4 = _conv_block(mobile_conv3, c=64, s=2, n=3, t=6, stage=5)  # (14, 14, 64)
    mobile_conv5 = _conv_block(mobile_conv4, c=160, s=1, n=4, t=6, stage=6)  # (14, 14, 96)
    mobile_conv6 = _conv_block(mobile_conv5, c=160, s=2, n=3, t=6, stage=7)  # (7, 7, 160)
    mobile_conv7 = _conv_block(mobile_conv6, c=320, s=1, n=1, t=6, stage=8)  # (7, 7, 320)

    fc6, fc6_for_feature = _SSD_Conv_fc(mobile_conv7, 1024, kernel_size = (3, 3), strides=(2, 2))
    fc7, fc7_for_feature = _SSD_Conv_fc(fc6, 1024, kernel_size= (1, 1))
    conv8_2, conv8_2_for_feature = _SSD_Conv(fc7, 512, kernel_size=(3, 3), strides = (2, 2))
    conv9_2, conv9_2_for_feature = _SSD_Conv(conv8_2, 512, kernel_size=(3, 3), strides = (1, 1))
    conv10_2, conv10_2_for_feature = _SSD_Conv(conv9_2, 512, kernel_size=(3, 3), strides = (1, 1))

    clf1_mbox_loc_flat, clf1_mbox_conf_flat, clf1_mbox_priorbox = _detections(mobile_conv4, 1, 4, 0.2, 0.9, num_classes)
    clf2_mbox_loc_flat, clf2_mbox_conf_flat, clf2_mbox_priorbox = _detections(fc6_for_feature, 2, 4, 0.2, 0.9, num_classes)
    clf3_mbox_loc_flat, clf3_mbox_conf_flat, clf3_mbox_priorbox = _detections(fc7_for_feature, 3, 6, 0.2, 0.9, num_classes)
    clf4_mbox_loc_flat, clf4_mbox_conf_flat, clf4_mbox_priorbox = _detections(conv8_2_for_feature, 4, 6, 0.2, 0.9, num_classes)
    clf5_mbox_loc_flat, clf5_mbox_conf_flat, clf5_mbox_priorbox = _detections(conv9_2_for_feature, 5, 6, 0.2, 0.9, num_classes)
    clf6_mbox_loc_flat, clf6_mbox_conf_flat, clf6_mbox_priorbox = _detections(conv10_2_for_feature, 6, 4, 0.2, 0.9, num_classes)

    mbox_loc = Concatenate(axis = 1, name = 'mbox_loc')([clf1_mbox_loc_flat, clf2_mbox_loc_flat,
                                                         clf3_mbox_loc_flat, clf4_mbox_loc_flat,
                                                         clf5_mbox_loc_flat, clf6_mbox_loc_flat])
    mbox_conf = Concatenate(axis = 1, name = 'mbox_conf')([clf1_mbox_conf_flat, clf2_mbox_conf_flat,
                                                           clf3_mbox_conf_flat, clf4_mbox_conf_flat,
                                                           clf5_mbox_conf_flat, clf6_mbox_conf_flat])
    mbox_priorbox = Concatenate(axis = 1, name = 'mbox_priorbox')([clf1_mbox_priorbox, clf2_mbox_priorbox,
                                                                   clf3_mbox_priorbox, clf4_mbox_priorbox,
                                                                   clf5_mbox_priorbox, clf6_mbox_priorbox])

    print('Brfore Reshape', mbox_loc.shape, mbox_conf.shape, mbox_priorbox.shape)

    num_boxes = mbox_loc.shape[-1] // 4

    mbox_loc = Reshape((num_boxes, 4), name = 'mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes), name = 'mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name = 'mbox_conf_final')(mbox_conf)
    print('After Reshape', mbox_loc.shape, mbox_conf.shape, mbox_priorbox.shape)

    predictions = Concatenate(axis = 2, name = 'predictions')([mbox_loc, mbox_conf, mbox_priorbox])
    print('predictions shape ', predictions.shape)

    model = Model(inputs = img_input, outputs = predictions)

    return model