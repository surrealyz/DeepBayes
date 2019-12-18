from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, os
sys.path.extend(['alg/', 'models/', 'utils/'])
from utils import load_data, save_params, load_params, init_variables
from visualisation import plot_images
from alg.vae_new import construct_optimizer

dimZ = 256
dimH = 512
n_iter = 100
batch_size = 1000
lr = 1e-4
K = 1
checkpoint = -1
global lambda_y


def main(data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint):
    # load data
    from utils_spam import data_spam
    X_train, Y_train, X_test, Y_test = data_spam(train_start=0, train_end=295870,
                                                  test_start=0, test_end=126082)
    print(X_train[0])
    print(Y_train[0])
    dimY = Y_train.shape[1]
 
    # if vae_type == 'A':
    #     from conv_generator_mnist_A import generator
    # if vae_type == 'B':
    #     from conv_generator_mnist_B import generator
    # if vae_type == 'C':
    #     from conv_generator_mnist_C import generator
    # if vae_type == 'D':
    #     from conv_generator_mnist_D import generator
    # if vae_type == 'E':
    #     from conv_generator_mnist_E import generator
    if vae_type == 'F': 
        from models.mlp_generator_spam_F import generator
    # if vae_type == 'G':
    #     from models.conv_generator_mnist_G import generator
    # from conv_encoder_mnist import encoder_gaussian as encoder
    from models.mlp_encoder_spam import encoder_gaussian as encoder
    input_shape = X_train[0].shape
    dimX = input_shape[0]
    print('input_shape:', input_shape)
    print('dimX:', dimX)

    # then define model
    dec = generator(dimX, dimH, dimZ, dimY, 'linear', 'gen')
    n_layers_enc = 2
    enc = encoder(dimX, dimH, dimZ, dimY, n_layers_enc, 'enc')

    # define optimisers
    X_ph = tf.placeholder(tf.float32, shape=(batch_size,)+input_shape)
    Y_ph = tf.placeholder(tf.float32, shape=(batch_size, dimY))
    print('X_ph.shape:', X_ph.shape)
    print('Y_ph.shape:', Y_ph.shape)
    print('X_train.shape:', X_train.shape)
    print('Y_train.shape:', Y_train.shape)

    ll = 'l2'
    identity = lambda x: x
    fit, eval_acc = construct_optimizer(X_ph, Y_ph, [identity, enc], dec, ll, K, vae_type)

    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if not os.path.isdir('save/'):
        os.mkdir('save/')
        print('create path save/')
    path_name = data_name + '_conv_vae_%s/' % (vae_type + '_' + str(dimZ))
    if not os.path.isdir('save/'+path_name):
        os.mkdir('save/'+path_name)
        print('create path save/' + path_name)
    filename = 'save/' + path_name + 'checkpoint'
    if checkpoint < 0:
        print('training from scratch')
        init_variables(sess)
    else:
        load_params(sess, filename, checkpoint)
    checkpoint += 1
  
    import keras.backend
    keras.backend.set_session(sess)
    keras.backend.set_learning_phase(0)

    # now start fitting 
    n_iter_ = min(n_iter,10)
    beta = 1.0 # here beta
    for i in range(int(n_iter/n_iter_)):
        fit(sess, X_train, Y_train, n_iter_, lr, beta)
        # print training and test accuracy
        eval_acc(sess, X_train, Y_train, 'train', beta)
        eval_acc(sess, X_test, Y_test, 'test', beta)

    # save param values
    save_params(sess, filename, checkpoint, scope = 'vae') 
    checkpoint += 1

if __name__ == '__main__':
    data_name = 'spam'
    vae_type = str(sys.argv[1]) 
    main(data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint)
    
