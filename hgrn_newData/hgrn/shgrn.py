from typing import List, Callable, Union
import tensorflow.compat.v1 as tf
from .model import Model

from .utils import mixed_dropout
from .highOrderInfo import InfoPropagation

sparse_dot = tf.sparse_tensor_dense_matmul


class SHGRN(Model):

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
        w_tensor = tf.expand_dims(W, axis=2, name=None)
        X_drop = mixed_dropout(X, keep_prob)

        attn = tf.matmul(X_drop, w_tensor)
        attn = tf.squeeze(attn)
        ceof = tf.nn.softmax(attn, axis=1)
        ceof = tf.expand_dims(ceof, axis=2, name=None)
        result = X_drop * ceof
        result = tf.reduce_sum(result, axis=1)
        return result


    def build_model(
            self, propagation: InfoPropagation,
            hiddenunits: List[int] = [16], reg_lambda: float = 1e-3, learning_rate: float = 0.01,
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
        self.propa = self.propagation.build_model(self.logits_local, self.gcn_adj, keep_prob)
        # layerAttention
        with tf.variable_scope(f'layer_attention'):
            self.logits = self._build_attn(
                self.propa, self.propa.get_shape()[2], keep_prob_layers)


        self._build_loss(reg_lambda)
        self._build_training(learning_rate)
        self._build_results()
