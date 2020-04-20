from keras.applications import MobileNet
import tensorflow as tf
from keras.layers import *
from keras.optimizers import *
from keras.models import Model

from keras.optimizers import Adam
import keras

def efficientnet(img_h, img_w, img_ch, pretrained_weights = None, classification_type = 'categorical',
             output_type = 3):
    mobilenet = MobileNet(weights=None, include_top = False, input_shape = (img_h, img_w, img_ch))

    # classifier
    x = GlobalAveragePooling2D(name='global_average_pooling_first')(mobilenet.output)
    x = Dense(256, name='fc_1', kernel_initializer='he_normal')(x)
    #     x = Dense(256, name='fc_1', kernel_initializer = 'lecun_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #     x = Activation('selu')(x)

    #     x = Dense(128, name='fc_1', kernel_initializer = 'he_normal')(x)
    x = Dense(128, name='fc_2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #     x = Activation('selu')(x)

    if (classification_type == 'categorical'):
        x = Dense(output_type, name='dense_3', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)
    elif (classification_type == 'binary'):
        x = Dense(1, name='dense_3', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)

    model = Model(inputs=mobilenet.input, outputs=x)
    adam = Adam(lr=1e-4)

    loss = 'binary_crossentropy' if classification_type == 'binary' else 'categorical_crossentropy'
    metrics = [keras.metrics.binary_accuracy] if classification_type == 'binary' else [
        keras.metrics.categorical_accuracy]

    model.compile(optimizer=adam, loss=loss, metrics=metrics)

    if (pretrained_weights):
        print('load weigts:{}'.format(pretrained_weights))
        model.load_weights(pretrained_weights)

    return model
