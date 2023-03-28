from typing import List, Callable, Union
import tensorflow.compat.v1 as tf
from .model import Model

from .utils import mixed_dropout
from .highOrderInfo import InfoPropagation_hgrn
import numpy as np

sparse_dot = tf.sparse_tensor_dense_matmul


class HGRN(Model):

    def _build_layer(
            self, X: Union[tf.Tensor, tf.SparseTensor], out_size: int,
            activation: Callable[[tf.Tensor, str], tf.Tensor],
            regularize: bool = True, keep_prob: float = 0.5) -> tf.Tensor:
        W = tf.get_variable(
            'weights',
            [X.get_shape()[1], out_size],
            dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer())
        if regularize:
            self.reg_vars.append(W)

        X_drop = mixed_dropout(X, keep_prob)

        if isinstance(X, tf.SparseTensor):
            Z_inner = sparse_dot(X_drop, W)
        else:
            Z_inner = X_drop @ W
        return activation(Z_inner)

    def _build_attn(
            self, X: tf.Tensor, out_size: int, keep_prob: float) -> tf.Tensor:
        W = tf.get_variable(
            'weights',
            [X.get_shape()[0], out_size],
            dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer())
        self.reg_vas_attn.append(W)
        w_tensor = tf.expand_dims(W, axis=2, name=None)
        X_drop = mixed_dropout(X, keep_prob)

        attn = tf.matmul(X_drop, w_tensor)
        attn = tf.squeeze(attn)
        ceof = tf.nn.softmax(attn, axis=1)
        ceof = tf.expand_dims(ceof, axis=2, name=None)
        result = X_drop * ceof
        result = tf.reduce_sum(result, axis=1)
        return result

    def sp_attn_head(self, seq_fts, adj_mat, activation, nb_nodes, lambda_graph=0.2, coef_drop=0.2):
        # simplest self-attention possible
        with tf.variable_scope(f'attn_1'):
            f_1 = self._build_layer(
                seq_fts, 1,
                activation=lambda x: x,
                regularize=False,
                keep_prob=1)
        with tf.variable_scope(f'attn_2'):
            f_2 = self._build_layer(
                seq_fts, 1,
                activation=lambda x: x,
                regularize=False,
                keep_prob=1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)
        coefs = tf.SparseTensor(indices=coefs.indices,
                                values=self.gcn_adj.values * (1 - lambda_graph) + coefs.values * lambda_graph,
                                dense_shape=coefs.dense_shape)
        coef = coefs

        coefs = tf.SparseTensor(indices=coefs.indices,
                                values=tf.nn.dropout(coefs.values, coef_drop),
                                dense_shape=coefs.dense_shape)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)

        return activation(vals), coef  # activation

    def sp_attn(self, seq_fts, adj_mat, activation, nb_nodes, lambda_graph=0.2, coef_drop=0.2):
        edge = tf.transpose(adj_mat.indices)
        seq1 = tf.gather(seq_fts, edge[0, :], axis=0)
        seq2 = tf.gather(seq_fts, edge[1, :], axis=0)
        edge_h = tf.concat((seq1, seq2), axis=1)

        # simplest self-attention possible
        with tf.variable_scope(f'attn'):
            f_1 = self._build_layer(
                edge_h, 1,
                activation=lambda x: x,
                regularize=False,
                keep_prob=1)

        lrelu = tf.SparseTensor(indices=adj_mat.indices,
                                values=tf.squeeze(tf.nn.leaky_relu(f_1)),
                                dense_shape=adj_mat.dense_shape)
        coefs = tf.sparse_softmax(lrelu)
        coefs = tf.SparseTensor(indices=coefs.indices,
                                values=self.gcn_adj.values * (1 - lambda_graph) + coefs.values * lambda_graph,
                                dense_shape=coefs.dense_shape)
        coef = coefs

        coefs = tf.SparseTensor(indices=coefs.indices,
                                values=tf.nn.dropout(coefs.values, coef_drop),
                                dense_shape=coefs.dense_shape)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)

        return activation(vals), coef  # activatio


    def build_model(
            self, propagation: InfoPropagation_hgrn,
            hiddenunits: List[int] = [16], reg_lambda: float = 1e-3,
            reg_attn: float = 1e-4, lambda_graph: float = 0.2, learning_rate: float = 0.01,
            keep_prob: float = 0.5, keep_feature: float = 0.6, keep_prob_layers: float = 0.8,
            activation_fn: Callable[[tf.Tensor, str], tf.Tensor] = tf.nn.relu):
        self.isTrain = tf.placeholder(tf.bool, [], name='isTrain')
        self.idx = tf.placeholder(tf.int32, [None], name='idx')
        self.propagation = propagation
        self.hiddenunits = hiddenunits

        keep_prob = tf.maximum(tf.cast(~self.isTrain, tf.float32), keep_prob)
        keep_feature = tf.maximum(tf.cast(~self.isTrain, tf.float32), keep_feature)
        keep_prob_layers = tf.maximum(tf.cast(~self.isTrain, tf.float32), keep_prob_layers)

        self.Zs = [self.attr_mat_norm]
        # Hidden layers
        for i, hiddenunit in enumerate(self.hiddenunits):
            with tf.variable_scope(f'layer_{i}'):
                first_layer = i == 0
                keep_prob_current = keep_feature if first_layer else 1.
                self.Zs.append(self._build_layer(
                    self.Zs[-1], hiddenunit,
                    activation=activation_fn,
                    regularize=first_layer,
                    keep_prob=keep_prob_current))

        # Last layer
        with tf.variable_scope(f'layer_{len(self.hiddenunits)}'):
            self.logits_local = self._build_layer(
                self.Zs[-1], self.nclasses,
                activation=lambda x: x,
                regularize=False, keep_prob=keep_feature)

        # Propagation
        self.propa = self.propagation.build_model(self.logits_local, keep_prob, lambda_graph)

        # layerAttention
        with tf.variable_scope(f'layer_attention'):
            self.logits = self._build_attn(
                self.propa, self.propa.get_shape()[2], keep_prob_layers)

        self._build_loss(reg_lambda, reg_attn)
        self._build_training(learning_rate)
        self._build_results()

