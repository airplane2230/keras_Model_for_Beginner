import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from utils.computation import convert2wh, corner2center
from utils.Matching import Matcher

class MultiboxLoss(object):
    '''
        loss func defiend as Loss = (Loss_conf + a * Loss_loc) / N
        need for total loss.

        Need list:
            confidence loss
            location loss
            positive list
            negative list
    '''

    def __init__(self, batch_size):
        self.batch_size = batch_size

    # bbox 의 loc에 대한 loss 계산
    def _smooth_L1_Loss(self, y_true, y_pred, pos):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)

        # shape : [?, num_boxes, 4] -> [?, num_boxes]
        return tf.reduce_sum(l1_loss, axis=-1) * pos

    # bbox의 class에 대한 loss 계산
    def _softmax_Loss(self, y_true, y_pred, pos, neg):
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)

        # positive는 IOU를 합격한 박스 1 또는 0
        # 맞춘 만큼 loss값을 감소시킴.
        pos_loss = (tf.log(tf.exp(y_pred) / (tf.reduce_sum(tf.exp(y_pred), axis=-1))))

        # IOU는 불합격했지만, 클래스가 있을 확률이 높은 박스 1 또는 0
        # 맞춘 만큼 loss를 감소시킴
        neg_loss = tf.log(y_pred)

        softmax_loss = -tf.reduce_sum((y_true * (pos_loss + neg_loss)), axis=- 1) * (pos + neg)

        return softmax_loss

    # total_loss 계산
    def comute_loss(self, y_true, y_pred):
        """ Compute multibox loss
        # Arguments
            @y_true:
                tensor of shape (?, num_object, 4 + 4) -> [?, ?, 8]
            @y_pred:
                tensor of shape(?, num_boxes, 4 + num_classes(4) + 4)

            @class_num = 4

            @configration of y_pred + y_true:
                y_pred[:, :, :4]:
                    bbox_loc
                y_pred[:, :, 4:8]:
                    class_confidence
                y_pred[:, :, 10:]:
                    mbox_priorbox(cx, cy, w, h)
        """
        # default_boxes shape : (batch_size, 938, 4) --> 8 is batch_size
        default_boxes = y_pred[:, :, -4:]

        positives = []
        negatives = []
        ex_gt_labels = []
        ex_gt_boxes = []

        num_boxes = y_pred.shape[1]
        matcher = Matcher(num_boxes, default_boxes)
        print('make Matcher=======================')

        actual_locs = []
        actual_labels = []

        for i in range(self.batch_size):
            # y_true[i][:, :, :4], [-1, 4] --> (i, num_box, 4)
            # tf.reshape(y_true[i][:, :, :4], [-1, 4]) --> (num_box, 4)
            locs = y_true[i, :, :4].to_tensor()
            # labels shape: (num_box, )
            labels = tf.math.argmax(y_true[0, :, 4:].to_tensor(), axis = 1)

            for loc in locs:
                loc = convert2wh(loc)
                loc = corner2center(loc)
                actual_locs.append(loc)

            for label in labels:
                actual_labels.append(label)

            pred_locs = y_pred[i][:, :4]  # <class 'tensorflow.python.framework.ops.Tensor'> (938, 4)
            pred_confs = y_pred[i][:, 4:-4]  # <class 'tensorflow.python.framework.ops.Tensor'> (938, 25)

            print('go in matcher.matching#####')
            pos_list, neg_list, t_gtl, t_gtb = matcher.matching(pred_confs, pred_locs, actual_labels, actual_locs, i)
            positives.append(pos_list)  # (?, default_box_num_pos)
            negatives.append(neg_list)  # (?, default_box_num_neg)
            ex_gt_labels.append(t_gtl)  # (?, default_box_num_label)
            ex_gt_boxes.append(t_gtb)  # (?, default_box_num, loc)

        ex_gt_labels_to_categorical = to_categorical(ex_gt_labels)
        # 클래스에 대한 손실 함수
        conf_loss = self._softmax_Loss(ex_gt_labels_to_categorical,
                                       y_pred[:, :, 4:8],
                                       positives, negatives)  # [?, 984]

        # 박스 위치에 대한 손실 함수
        loc_loss = self._smooth_L1_Loss(y_true[:, :, :4],
                                        y_pred[:, :, 4], positives)  # [?, 984]

        total_loss = tf.reduce_sum(conf_loss + loc_loss)

        return total_loss