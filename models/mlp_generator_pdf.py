import numpy as np
import tensorflow as tf
from mlp import mlp_layer

"""
generator p(z)p(x|z)
"""


def generator(dimX, dimH, dimZ, last_activation, name):
    # construct p(x|z)
    fc_layers = [dimZ, dimH, dimH, dimX]
    l = 0
    pxz_mlp_layers = []
    N_layers = len(fc_layers) - 1
    for i in range(N_layers):
        if i < N_layers - 1:
            activation = 'relu'
        else:
            activation = last_activation
        name_layer = name + '_pxz_mlp_l%d' % l
        with tf.variable_scope('vae'):
            pxz_mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i + 1], activation, name_layer))
        l += 1

    def pxz_params(z):
        out = z
        for layer in pxz_mlp_layers:
            out = layer(out)
        return out

    return pxz_params
