import numpy as np
import tensorflow as tf

def intersection(rect1, rect2):
    """
    intersecton of units
    compute boarder line top, left, right and bottom.
    rect is defined as [ top_left_x, top_left_y, width, height ]
    """
    top = np.max(rect1[1], rect2[1])
    left = np.max(rect1[0], rect2[0])
    right = np.min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    bottom = np.min(rect1[1] + rect1[3], rect2[1] + rect2[3])

    result = tf.where(tf.math.logical_and(tf.greater(bottom, top), tf.greater(right, left)),
                      (bottom - top) * (right - left), 0)

    return result


def jaccard(rect1, rect2):
    """
    Jaccard index.
    Jaccard index is defined as #(A∧B) / #(A∨B)

    len_rect1 : 4
    len_rect2 : 4

    """

    # len_rect1_ : 4, len_rect2_ : 4
    rect1_ = []
    for i in range(len(rect1)):
        cond_value = tf.where(rect1[i] >= 0, rect1[i], 0)
        rect1_.append(cond_value)

    rect2_ = []
    for i in range(len(rect2)):
        cond_value = tf.where(rect2[i] >= 0, rect2[i], 0)
        rect2_.append(cond_value)

    s = tf.add(tf.multiply(rect1_[2], rect1_[3]), tf.multiply(rect2_[2], rect2_[3]))

    # rect1 and rect2 => A∧B
    intersect = intersection(rect1_, rect2_)

    # rect1 or rect2 => A∨B
    union = s - intersect

    # A∧B / A∨B
    return tf.divide(intersect, union)


def corner2center(rect):
    """
    rect is defined as [ top_left_x, top_left_y, width, height ]
    """
    center_x = (2 * rect[0] + rect[2]) * 0.5
    center_y = (2 * rect[1] + rect[3]) * 0.5

    return tf.stack([center_x, center_y, abs(rect[2]), abs(rect[3])])


def center2corner(rect):
    """
    rect is defined as [ top_left_x, top_left_y, width, height ]
    """
    corner_x = rect[0] - rect[2] * 0.5
    corner_y = rect[1] - rect[3] * 0.5

    return tf.stack([corner_x, corner_y, tf.math.abs(rect[:, 2]), tf.amth.abs(rect[:, 3])])


def convert2diagonal_points(rect):
    """
    convert rect format
    Args:
        input format is...
        [ top_left_x, top_left_y, width, height ]
    Returns:
        output format is...
        [ top_left_x, top_left_y, bottom_right_x, bottom_right_y ]
    """
    return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]


def convert2wh(rect):
    """
    convert rect format
    Args:
        input format is...
        [ top_left_x, top_left_y, bottom_right_x, bottom_right_y ]
    Returns:
        output format is...
        [ top_left_x, top_left_y, width, height ]
    """
    result = tf.stack([rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]])

    return result