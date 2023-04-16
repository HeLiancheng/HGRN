from typing import Union, Tuple, List
import numpy as np
import tensorflow.compat.v1 as tf
import scipy.sparse as sp

from .utils import matrix_to_tensor, sparse_matrix_to_tensor
from ..preprocessing import normalize_attributes
from .highOrderInfo import calc_A_hat


class Model:
    def __init__(
            self, attr_matrix: Union[np.ndarray, sp.spmatrix],
            adj: sp.spmatrix,
            labels: np.ndarray, sess: tf.Session):
        attr_mat_norm = normalize_attributes(attr_matrix)
        self.gcn_adj = sparse_matrix_to_tensor(calc_A_hat(adj))
        self.attr_mat_norm = matrix_to_tensor(attr_mat_norm)

        self.labels_np = labels
        self.labels = tf.constant(labels, dtype=tf.int32)
        self.sess = sess
        self.nnodes, self.nfeatures = attr_matrix.shape
        self.nclasses = np.max(labels) + 1
        self.reg_vars = []
        self.step = tf.train.create_global_step()

    def _calc_cross_entropy(self):
        logits_subset = tf.gather(self.logits, self.idx, axis=0)
        labels_subset = tf.gather(self.labels, self.idx, axis=0)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_subset, logits=logits_subset,
            name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
        return cross_entropy_mean

    def _calc_cross_entropy_pseudo(self):
        labels_subset = tf.gather(self.labels, self.idx, axis=0)

        score_coo = tf.cast(tf.where(self.score > self.similarity_rate), tf.int32)
        index_arr = tf.cast(tf.where(tf.equal(score_coo[:, 0], tf.expand_dims(self.idx, axis=-1))), tf.int32)
        temp = tf.gather(score_coo, index_arr[:, 1])
        new_logits = tf.gather(self.logits, temp[:, 1])
        new_labels = tf.gather(labels_subset, index_arr[:, 0])
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=new_labels, logits=new_logits,
            name='cross_entropy_pseudo')
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        return cross_entropy_mean

    def _calc_cross_class(self):
        logits = tf.nn.softmax(self.logits)
        trans = tf.transpose(logits)
        result = tf.matmul(trans, logits)
        diag_vector = tf.diag_part(result)
        diag_matrix = tf.matrix_diag(diag_vector)
        result = result - diag_matrix
        sum_result = tf.reduce_sum(result, [0,1], keep_dims=False)

        length = tf.cast(self.logits.shape[0], tf.float32)
        sum_result = sum_result/length
        return sum_result

    def _calc_f1_accuracy(self, predictions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        predictions_subset = tf.gather(predictions, self.idx, axis=0)
        labels_subset = tf.gather(self.labels, self.idx, axis=0)
        with tf.variable_scope('Accuracy'):
            correct = tf.equal(predictions_subset, labels_subset)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('Accuracy', accuracy)
        with tf.variable_scope('F1_score'):
            def sum_cast(x, dtype=tf.float32):
                return tf.reduce_sum(tf.cast(x, dtype))

            for i in range(self.nclasses):
                pred_is_i = tf.equal(predictions_subset, tf.constant(i))
                label_is_i = tf.equal(labels_subset, tf.constant(i))
                true_pos = sum_cast(pred_is_i & label_is_i)
                false_pos = sum_cast(pred_is_i & ~label_is_i)
                false_neg = sum_cast(~pred_is_i & label_is_i)
                precision = tf.cond(
                    tf.equal(false_pos, 0),
                    lambda: 1.,
                    lambda: true_pos / (true_pos + false_pos))
                recall = tf.cond(
                    tf.equal(false_neg, 0),
                    lambda: 1.,
                    lambda: true_pos / (true_pos + false_neg))
                if i == 0:
                    avg_precision = precision
                    avg_recall = recall
                else:
                    avg_precision += precision
                    avg_recall += recall
            avg_precision /= self.nclasses
            avg_recall /= self.nclasses
            f1_score = (2 * (avg_precision * avg_recall)
                        / (avg_precision + avg_recall))
            tf.summary.scalar('F1_score', f1_score)
        return f1_score, accuracy


    def graph_reconstruct_construct(self):
        adj_dense = tf.sparse_tensor_to_dense(self.gcn_adj)
        trans = tf.transpose(self.logits)
        result = tf.matmul(self.logits, trans)
        logits = tf.nn.softmax(result)
        ones = tf.ones_like(result)
        #invers_adj = ones - adj_dense
        #invers_result = ones - logits
        loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=adj_dense, logits=logits)
        #loss2 = tf.nn.softmax_cross_entropy_with_logits(invers_result, invers_adj)
        loss = tf.reduce_mean(loss1)
        return loss * 0.1


    def _build_loss(self, reg_lambda: float):
        with tf.variable_scope('Loss'):
            cross_entropy_mean = self._calc_cross_entropy()
            l2_reg = tf.add_n([
                tf.nn.l2_loss(weight) for weight in self.reg_vars])


            self.loss = cross_entropy_mean + reg_lambda * l2_reg
            #tf.summary.scalar('l2_reg', l2_reg)
            tf.summary.scalar('loss', self.loss)

    def _build_training(self, learning_rate: float):
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(
            loss=self.loss, global_step=tf.train.get_global_step())

    def _build_results(self):
        self.predictions = tf.argmax(
            self.logits, axis=1, output_type=tf.int32)
        self.f1_score, self.accuracy = self._calc_f1_accuracy(
            self.predictions)
        self.summary = tf.summary.merge_all()

    def get_vars(self) -> List[np.ndarray]:
        return self.sess.run(tf.trainable_variables())

    def set_vars(self, new_vars: List[np.ndarray]):
        set_all = [
            var.assign(new_vars[i])
            for i, var in enumerate(tf.trainable_variables())]
        self.sess.run(set_all)

    def get_predictions(self) -> np.ndarray:
        inputs = {
            self.idx: np.arange(self.nnodes),
            self.isTrain: False}
        return self.sess.run(self.predictions, feed_dict=inputs)
