from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, os
sys.path.extend(['alg/', 'models/', 'utils/'])
from utils import load_data, save_params, load_params, init_variables
from visualisation import plot_images
from alg.vae_new_pdf_F import construct_optimizer
from sklearn import datasets
import requests
import json

dimZ = 512
dimH = 1024
n_iter = 100
batch_size = 10
lr = 1e-4
K = 1
checkpoint = 6
global lambda_y

HOST = 'localhost'
PORT = 8501

def shuffle_data(x, y):
    idx = np.arange(0 , len(x))
    np.random.shuffle(idx)
    x_shuffle = np.array([x[i] for i in idx])
    y_shuffle = np.array([y[i] for i in idx])
    return x_shuffle, y_shuffle

def query_tf(X):
    payload = {'instances': X.tolist()}
    REST_URL='http://%s:%d/v1/models/baseline:predict' % (HOST, PORT)
    r = requests.post(REST_URL, json=payload)
    if r.status_code == 200:
        json_decoder = json.JSONDecoder()
        res = json_decoder.decode(r.text)
        return np.array([item['y_pred'] for item in res['predictions']])

def main(attack_name, start_num, epoch_num, data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint, categorical=True, bin_num=2):
    # load data
    #beta = 0.1  # here beta
    beta = 1  # here beta

    from utils_pdf import data_pdf, to_categorical
    X_train, Y_train, X_test, Y_test, X_attack_0, Y_attack_0 = data_pdf(train_start=0, train_end=13190,
                                    test_start=0, test_end=6146, attack_start=0, attack_end=1600)

    dimY = Y_train.shape[1]

    X_test, Y_test = shuffle_data(X_test, Y_test)

    attack_data = '/home/yz/code/ext/models/'+attack_name+'.queries.libsvm'
    X_attack, Y_attack = datasets.load_svmlight_file(attack_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)
    X_attack = X_attack.toarray()
    Y_attack = to_categorical(Y_attack, 2)
    print('Eval PDF X_attack shape:', X_attack.shape)

    if vae_type == 'F':
        if categorical:
            from models.mlp_generator_categorical_pdf_F import generator
        else:
            #from models.mlp_generator_pdf_F import generator
            exit()

    from models.mlp_encoder_pdf_F import encoder_gaussian as encoder
    input_shape = X_train[0].shape
    dimX = input_shape[0]
    print('input_shape:', input_shape)
    print('dimX:', dimX)

    # then define model
    if categorical:
        dec = generator(dimX, dimH, dimZ, dimY, 'linear', 'gen', bin_num=bin_num)
    else:
        #dec = generator(dimX, dimH, dimZ, dimY, 'linear', 'gen')
        exit()
    n_layers_enc = 2
    enc = encoder(dimX, dimH, dimZ, dimY, n_layers_enc, 'enc')

    # define optimisers
    X_ph = tf.placeholder(tf.float32, shape=(batch_size,)+input_shape)
    Y_ph = tf.placeholder(tf.float32, shape=(batch_size, dimY))

    # print('X_ph.shape:', X_ph.shape)
    # print('Y_ph.shape:', Y_ph.shape)
    # print('X_train.shape:', X_train.shape)
    # print('Y_train.shape:', Y_train.shape)

    #ll = 'l2'
    ll = 'xe'
    identity = lambda x: x
    fit, eval_prob = construct_optimizer(X_ph, Y_ph, [identity, enc], dec, ll, K, vae_type, categorical)

    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if not os.path.isdir('save/'):
        os.mkdir('save/')
        print('create path save/')
    path_name = data_name + '_conv_vae_%s' % (vae_type + '_' + str(dimZ)) + '_beta_{}_ll_{}/'.format(beta, ll)
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

    # check likelihood by batches
    start = 0
    end = start_num
    print('\n***** CHEKING ATTACK DATA *****\n')
    for epoch in range(epoch_num):
        print('===== total:%d, num of queries:%d =====' % (end, end-start))
        eval_prob(sess, X_attack[:end], Y_attack[:end], 'attack', beta)
        start = end
        end = start * 2
        sys.stdout.flush()

    total = 0
    start = 0
    end = start_num
    for epoch in range(epoch_num):
        start = end
        end = start * 2
    total = end

    print('=== making %d queries ===' % total)
    Y_hat = query_tf(X_test[:total])
    Y_hat = to_categorical(Y_hat, 2)

    start = 0
    end = start_num
    print('\n***** CHEKING TEST DATA *****\n')
    for epoch in range(epoch_num):
        print('===== total:%d, num of queries:%d =====' % (end, end-start))
        eval_prob(sess, X_test[:end], Y_hat[:end], 'attack', beta)
        start = end
        end = start * 2
        sys.stdout.flush()

if __name__ == '__main__':
    data_name = 'pdf'
    vae_type = 'F'
    attack_name = sys.argv[1]
    start_num = int(sys.argv[2])
    epoch_num = int(sys.argv[3])
    main(attack_name, start_num, epoch_num, data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint, categorical=True)