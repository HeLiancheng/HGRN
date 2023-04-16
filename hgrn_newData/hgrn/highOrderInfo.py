import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf

from .utils import sparse_matrix_to_tensor, sparse_dropout

sparse_dot = tf.sparse_tensor_dense_matmul


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


class InfoPropagation:
    def __init__(self, niter: int):
        self.niter = niter


    def build_model(self, Z: tf.Tensor, A_hat_tf: tf.SparseTensor, keep_prob: float) -> tf.Tensor:
        with tf.variable_scope(f'Propagation'):
            Zs_prop = Z
            Layer_z = [Zs_prop]
            for _ in range(self.niter):
                A_drop = sparse_dropout(A_hat_tf, keep_prob)
                Zs_prop = sparse_dot(A_drop, Zs_prop)
                Layer_z.append(Zs_prop)

            Zs_result = tf.stack(Layer_z, axis=1)
            return Zs_result

class InfoPropagation_pr:
    def __init__(self, niter: int):
        self.niter = niter


    def build_model(self, Z: tf.Tensor, A_hat_tf: tf.SparseTensor, keep_prob: float) -> tf.Tensor:
        with tf.variable_scope(f'Propagation'):
            Zs_prop = Z
            Layer_z = [Zs_prop]
            for _ in range(self.niter):
                A_drop = sparse_dropout(A_hat_tf, keep_prob)
                Zs_prop = sparse_dot(A_drop, Zs_prop)
                Layer_z.append(Zs_prop)

            return Layer_z

        