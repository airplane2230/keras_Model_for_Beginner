from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np


class PriorBox(Layer):
    def __init__(self, s_min=None, s_max=None,
                 feature_map_number=None, num_box=None, **kwargs):
        '''

        :param img_size:
        :param s_min:
        :param s_max:
        :param feature_map_number: [1, 2, 3, 4, 5, 6]
        '''

        self.default_boxes = []
        self.num_box = num_box
        if s_min <= 0:
            raise Exception('min_size must be positive')
        self.s_min = s_min
        self.s_max = s_max
        self.feature_map_number = feature_map_number
        self.aspect_ratio = [[1., 1 / 1, 2., 1 / 2],
                             [1., 1 / 1, 2., 1 / 2],
                             [1., 1 / 1, 2., 1 / 2, 3., 1 / 3],
                             [1., 1 / 1, 2., 1 / 2, 3., 1 / 3],
                             [1., 1 / 1, 2., 1 / 2, 3., 1 / 3],
                             [1., 1 / 1, 2., 1 / 2]]

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.width = input_shape[2]
        self.height = input_shape[1]

        self.s_k = self.get_sk(self.s_max, self.s_min, 6, self.feature_map_number)
        self.s_k1 = self.get_sk(self.s_max, self.s_min, 6, self.feature_map_number + 1)

        super(PriorBox, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2], 4)

    @tf.function
    def call(self, x):
        feature_map_ratio = self.aspect_ratio[self.feature_map_number - 1]
        s = 0.0

        default_boxes = None
        for eleh in range(self.height):
            center_y = (eleh + 0.5) / float(self.height)
            for elew in range(self.width):
                center_x = (elew + 0.5) / float(self.width)
                for ratio in feature_map_ratio:
                    s = self.s_k

                    if (ratio == 1.0):
                        s = np.sqrt(self.s_k * self.s_k1)

                    box_width = s * np.sqrt(ratio)
                    box_height = s / np.sqrt(ratio)

                    if default_boxes is None:
                        default_boxes = np.array([center_x, center_y, box_width, box_height]).reshape(-1, 4)
                    else:
                        default_boxes = np.concatenate(
                            (default_boxes, np.array([[center_x, center_y, box_width, box_height]])), axis=0)

        boxes_tensor = np.expand_dims(default_boxes, axis=0)
        boxes_tensor = tf.tile(tf.constant(boxes_tensor, dtype='float32'), (tf.shape(x)[0], 1, 1))

        return boxes_tensor

    def get_sk(self, s_max, s_min, m, k):
        '''
        :param s_max:
        :param s_min:
        :param m: number of feature map
        :param k: k-th feature map
        :return:
        '''
        sk_value = s_min + ((s_max - s_min) / (m - 1.0)) * (k - 1)

        return sk_value