import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Add, BatchNormalization, Activation
from tensorflow.keras.models import Model

# Convolution Block
def conv_block(x, filters, kernel_size, strides, padding = 'same'):
  x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(x)
  x = BatchNormalization()(x)

  return x

# ResNet Block
def res_block(x, filters, strides, kernel_size):
  residual = Conv2D(filters = filters, kernel_size = 1, strides = strides, padding = 'same')(x)

  x = conv_block(x, filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')
  x = Activation('relu')(x)
  x = conv_block(x, filters = filters, kernel_size = kernel_size, strides = 1, padding= 'same')

  x = Add()([x, residual])
  x = Activation('relu')(x)

  return x

def make_res_block(x, num_blocks, filters, strides, kernel_size):
  # downsample이 필요한 시점과 필요하지 않은 시점을 위해 하나는 떼어둠
  x = res_block(x, filters, strides, kernel_size)

  for _ in range(num_blocks - 1):
    x = res_block(x, filters, strides = 1, kernel_size = kernel_size)

  return x

def resnet_34(input_shape):

  inputs = Input(shape = input_shape)

  x = Conv2D(64, kernel_size = 7, strides = 2, padding = 'same')(inputs)
  x = MaxPooling2D(pool_size = (3, 3), strides = 2, padding = 'same')(x)

  # ResNet Blcok Stage
  init_block = make_res_block(x, num_blocks = 3, filters = 64, strides = 1, kernel_size = 3)
  second_block = make_res_block(init_block, num_blocks = 4, filters = 128, strides = 2, kernel_size = 3)
  third_block = make_res_block(second_block, num_blocks = 6, filters = 256, strides = 2, kernel_size = 3)
  fourth_blcok = make_res_block(third_block, num_blocks = 3, filters = 512, strides = 2, kernel_size = 3)

  # Global Average Pooling
  gap = GlobalAveragePooling2D()(fourth_blcok)

  # FCN
  outputs = Dense(1000, activation = 'softmax')(gap)

  model = Model(inputs = inputs, outputs = outputs)

  return model

model = resnet_34(input_shape = (224, 224, 3))
model.summary()