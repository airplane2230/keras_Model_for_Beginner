import numpy as np
import tensorflow as tf

from utils.computation import jaccard

classes = 21

class Matcher:
    def __init__(self, num_boxes, default_boxes):
        """
                initializer require feature-map shapes and default boxes
                Args:
                    fmap_shapes: feature-map's shape
                    default_boxes: generated default boxes
                """
        self.num_boxes = num_boxes
        self.default_boxes = default_boxes

    def matching(self, pred_confs, pred_locs, actual_labels, actual_locs, batch_size):
        """
                match default boxes and bouding boxes.
                matching computes pos and neg count for the computation of loss.
                now, the most noting point is that it is not important that
                whether class label is correctly predicted.
                class label loss is evaled by loss_conf
                matches variable have some Box instance and most of None.
                if jaccard >= 0.5, that matches box has Box(gt_loc, gt_label).
                then, sort by pred_confs loss and extract 3*pos boxes, which they
                have Box([], classes) => background.
                when compute losses, we need transformed ground truth labels and locations
                because each box has self confidence and location.
                so, we should prepare expanded labels and locations whose size is as same as len(matches).
                Args:
                    pred_confs: predicated confidences
                    pred_locs: predicated locations
                    actual_labels: answer class labels
                    actual_locs: answer box locations
                Returns:
                    postive_list: if pos -> 1 else -> 0
                    negative_list: if neg and label is not classes(not unknown class) 1 else 0
                    expanded_gt_labels: gt_label if pos else classes
                    expanded_gt_locs: gt_locs if pos else [0, 0, 0, 0]
        """
        self.pos = 0
        self.neg = 0
        pos_list = []
        neg_list = []
        expanded_gt_labels = []
        expanded_gt_locs = []

        matches = []
        matched = []
        # pred_confs.shsape, pred_locs.shape, len(actual_locs), len(actual_labels)
        # (938, 21) (938, 4) 2 2
        print('In Matcher1!===============================')
        for _ in range(self.num_boxes):
            matches.append(None)  # len is 938

        test_int = 0
        for gt_label, gt_box in zip(actual_labels, actual_locs):
            for i in range(len(matches)):
                # jacc : Tensor("PartitionedCall:0", shape=(), dtype=float64) () <class 'tensorflow.python.framework.ops.Tensor'>
                jacc = jaccard(gt_box, self.default_boxes[batch_size, i])  # self.default_boxes[batch_size, i] -> (4, )
                if (tf.math.greater_equal(jacc, 0.5)):
                    matches[i] = 4
                    self.pos += 1
                    matched.append(gt_label)

        print('In Matcher2!===============================')

        # neg, pos 비율 값
        neg_pos = 5

        max_length = tf.multiply(neg_pos, self.pos)

        loss_confs = []
        loss_conf_cnt = 0

        # pred_confs.shape : (938, 21)
        # 각 default box의 confidence에 대해서
        for i in range(pred_confs.get_shape()[0]):
            pred_conf = pred_confs[i]  # (21, )
            # 각 예측 값을 소프트 맥스함수를 통해 해당 클래스 인덱스로 치환한다.
            # (num_box, num_class) -> (num_box, )
            pred = tf.reduce_max(
                tf.divide(tf.math.exp(pred_conf), (tf.reduce_sum(tf.math.exp(pred_conf)) + 1e-5)))  # (), Tensor
            loss_confs.append(pred)
            loss_conf_cnt += 1

        # loss_conf_cnt : 938, int
        # max_length : tf.multimul(:)

        size = tf.math.minimum(loss_conf_cnt, neg_pos * self.pos)  # Tensor("Minimum:0", shape=(), dtype=int32)

        # TopKV2(values=<tf.Tensor 'TopKV2:0' shape=(None,) dtype=float64>, indices=<tf.Tensor 'TopKV2:1' shape=(None,) dtype=int32>)
        indices = tf.math.top_k(loss_confs, size)  # class : <clas`s 'tensorflow.python.ops.gen_nn_ops.TopKV2'>
        indice_values = indices[1]
        print(indice_values.__class__, indice_values.dtype, indice_values.shape, indice_values)
        print('In Matcher3!===============================')

        for i in range(indice_values.get_shape()[0]):
            # i : 0 ~
            temp_index = indice_values[i]
            print(temp_index, 'temp_index')

            # negative를 적당히 사용해야 되는데,
            # positive * neg_pos 비율보다 높으면 좋지 않으므로 break 시킨다.
            if self.neg > self.pos * neg_pos:
                break

            matches_index = tf.gather(matches, temp_index)
            pred_confs_index = tf.gather(pred_confs, temp_index)
            pred_conf = tf.argmax(pred_confs_index)

            # classes - 1은 배경을 의미함.
            # 박스가 안겹치면서 배경이 아닌 경우
            # False Negative -> 박스가 없다고 판단했지만 객체가 있을 수도 있는 부분 -하종우
            if (matches_index is None and (classes - 1 != pred_conf)):
                matches[temp_index] = 1
                self.neg += 1

        print('In Matcher4!===============================')

        # matches는 None이거나 Box instance가 들어있는 array이다.
        for box in matches:
            # 박스가 없으면
            if box is None:
                pos_list.append(0)
                neg_list.append(0)
                expanded_gt_labels.append(classes - 1)
                expanded_gt_locs.append([0] * 4)
            # False Negative 부분
            elif (0 == len(box.loc)):
                pos_list.append(0)
                neg_list.append(1)
                expanded_gt_labels.append(classes - 1)
                expanded_gt_locs.append([0] * 4)
            # 박스가 존재한다면
            else:
                pos_list.append(1)
                neg_list.append(0)
                expanded_gt_labels.append(box.index)
                expanded_gt_locs.append(box.loc)

        return pos_list, neg_list, expanded_gt_labels, expanded_gt_locs