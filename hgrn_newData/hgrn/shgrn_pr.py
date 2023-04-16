from typing import List, Callable, Union
import tensorflow.compat.v1 as tf
from .model import Model

from .utils import mixed_dropout
from .highOrderInfo import InfoPropagation_pr

sparse_dot = tf.sparse_tensor_dense_matmul


class SHGRN_PR(Model):

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
            self, propagation: InfoPropagation_pr,
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

        self.attr_mat_norm = tf.sparse_tensor_to_dense(self.attr_mat_norm)

        # Propagation
        self.propa = self.propagation.build_model(self.attr_mat_norm, self.gcn_adj, keep_prob)
        
        Zs = []
        # Hidden layers
        with tf.variable_scope(f'layer_0'):
            W1 = tf.get_variable(
                'weights',
                [self.nfeatures, hiddenunits[0]],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())
            self.reg_vars.append(W1)

            for layer in self.propa:
                X_drop = mixed_dropout(layer, keep_feature)

                l = layer @ W1
                l = activation_fn(l)
                Zs.append(l)



        lastZs = []
        # Last layer
        with tf.variable_scope(f'layer_last'):
            W2 = tf.get_variable(
                'weights',
                [hiddenunits[0], self.nclasses],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())

            for layer in Zs:
                X_drop = mixed_dropout(layer, keep_feature)
                l = layer @ W2
                lastZs.append(l)

        propa_result = tf.stack(lastZs, axis=1)

        # layerAttention
        with tf.variable_scope(f'layer_attention'):
            self.logits = self._build_attn(
                propa_result, propa_result.get_shape()[2], keep_prob_layers)


        self._build_loss(reg_lambda)
        self._build_training(learning_rate)
        self._build_results()
