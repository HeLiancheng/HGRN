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

def build_cosin(x: tf.Tensor, gcn_adj: tf.SparseTensor, lambda_graph=0.2):
    output_1 = tf.nn.l2_normalize(x, axis=1, epsilon=1e-10, name='nn_l2_norm')
    score = tf.matmul(output_1, tf.transpose(output_1))
    values = tf.squeeze(tf.gather_nd(score, gcn_adj.indices))
    coef = tf.SparseTensor(indices=gcn_adj.indices,
                               values=values,
                               dense_shape=gcn_adj.dense_shape)
    coefs = tf.sparse_softmax(coef)
    coefs = tf.SparseTensor(indices=coefs.indices,
                                values=gcn_adj.values * (1 - lambda_graph) + coefs.values * lambda_graph,
                                dense_shape=coefs.dense_shape)

    return coefs

class InfoPropagation:
    def __init__(self, adj_matrix: sp.spmatrix, niter: int):
        self.niter = niter
        self.A_hat = calc_A_hat(adj_matrix)

    def build_model(self, Z: tf.Tensor, keep_prob: float) -> tf.Tensor:
        with tf.variable_scope(f'Propagation'):
            A_hat_tf = sparse_matrix_to_tensor(self.A_hat)
            Zs_prop = Z
            Layer_z = [Zs_prop]
            for _ in range(self.niter):
                A_drop = sparse_dropout(A_hat_tf, keep_prob)
                Zs_prop = sparse_dot(A_drop, Zs_prop)
                #Layer_z.append(Zs_prop)

            #Zs_result = tf.stack(Layer_z, axis=1)
            return Zs_prop

        
class InfoPropagation_hgrn:
    def __init__(self, adj_matrix: sp.spmatrix, niter: int):
        self.niter = niter
        self.A_hat = calc_A_hat(adj_matrix)

    def build_model(self, Z: tf.Tensor, keep_prob: float, lambda_graph: float) -> tf.Tensor:
        with tf.variable_scope(f'Propagation'):
            gcn_adj = sparse_matrix_to_tensor(self.A_hat)
            A_hat_tf = build_cosin(Z, gcn_adj, lambda_graph=lambda_graph)
            Zs_prop = Z
            Layer_z = [Zs_prop]
            for _ in range(self.niter):
                A_drop = sparse_dropout(A_hat_tf, keep_prob)
                Zs_prop = sparse_dot(A_drop, Zs_prop)
                Layer_z.append(Zs_prop)

            Zs_result = tf.stack(Layer_z, axis=1)
            return Zs_result