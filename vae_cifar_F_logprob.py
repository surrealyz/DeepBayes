from __future__ import print_function

import numpy as np
import tensorflow as tf
import torch
import sys, os
sys.path.extend(['alg/', 'models/', 'utils/', '/home/yz/code/ext/'])
from utils import load_data, save_params, load_params, init_variables
from visualisation import plot_images
from vae_new import construct_optimizer
from sklearn import datasets
import requests
import json
import wrn
from torch.utils.data import TensorDataset, DataLoader

dimZ = 128#32
dimH = 1000
n_iter = 200
batch_size = 50
lr = 5e-5
K = 1
checkpoint = 0
data_path = 'cifar_data/'

def shuffle_data(x, y):
    idx = np.arange(0 , len(x))
    np.random.shuffle(idx)
    x_shuffle = np.array([x[i] for i in idx])
    y_shuffle = np.array([y[i] for i in idx])
    return x_shuffle, y_shuffle

def main(attack_name, start_num, epoch_num, data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint, categorical=True, bin_num=2):
    # load data
    #beta = 0.1  # here beta
    beta = 1  # here beta

    from import_data_cifar10 import load_data_cifar10, to_categorical
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X_train, X_test, Y_train, Y_test = load_data_cifar10(data_path, labels=labels, conv=True)
    dimY = Y_train.shape[1]

    X_test, Y_test = shuffle_data(X_test, Y_test)

    attack_data_path = '/home/yz/code/ext/models/'+attack_name+'.queries.csv'
    attack = np.loadtxt(attack_data_path, delimiter=",")
    X_attack = attack[:, 1:]
    Y_attack = attack[:, :1].flatten()
    Y_attack = to_categorical(Y_attack, 10)
    print('CIFAR10 X_attack shape:', X_attack.shape)
    # reshape X_attack
    X_attack = X_attack.reshape(len(Y_attack), 3, 32, 32).transpose(0, 2, 3, 1)
    print('After transpoe, X_attack shape:', X_attack.shape)

    if vae_type == 'F':
        from conv_generator_cifar10_F import generator
    else:
        exit()

    from conv_encoder_cifar10 import encoder_gaussian as encoder
    shape_high = (32, 32)
    input_shape = (32, 32, 3)
    n_channel = 64

    # input_shape = X_attack[0].shape
    # dimX = input_shape[0]
    # print('input_shape:', input_shape)
    # print('dimX:', dimX)

    # then define model
    dec = generator(input_shape, dimH, dimZ, dimY, n_channel, 'sigmoid', 'gen')
    enc, enc_conv, enc_mlp = encoder(input_shape, dimH, dimZ, dimY, n_channel, 'enc')

    # define optimisers
    X_ph = tf.placeholder(tf.float32, shape=(batch_size,)+input_shape)
    Y_ph = tf.placeholder(tf.float32, shape=(batch_size, dimY))
    ll = 'l2'
    fit, eval_prob = construct_optimizer(X_ph, Y_ph, [enc_conv, enc_mlp], dec, ll, K, vae_type)

    # print('X_ph.shape:', X_ph.shape)
    # print('Y_ph.shape:', Y_ph.shape)
    # print('X_train.shape:', X_train.shape)
    # print('Y_train.shape:', Y_train.shape)

    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
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

    # check likelihood by batches
    start = 0
    end = start_num
    print('\n***** CHEKING ATTACK DATA *****\n')
    for epoch in range(epoch_num):
        print('===== total:%d, num of queries:%d =====' % (end, end-start))
        eval_prob(sess, X_attack[:end], Y_attack[:end], 'attack', beta)
        if epoch == 0:
            start = end
            end = end + start_num
        elif epoch < epoch_num-1:
            tmp = start
            start = end
            end = end + (end-tmp)*2
        sys.stdout.flush()
    return
    # total = 0
    # start = 0
    # end = start_num
    # for epoch in range(epoch_num):
    #     start = end
    #     end = start
    total = end

    print('=== making %d test queries ===' % total)
    target_model = wrn.wrn28_2()
    ckpt = torch.load('/home/yz/code/ext/wrn28_2.ckp')
    target_model.load_state_dict(ckpt['state_dict'])
    X_test = X_test[:total].transpose(0, 3, 1, 2)
    Y_test = Y_test[:total]
    x_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(Y_test).float()
    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=True)

    Y_hat = []
    for x_batch, y_batch in test_loader:
        scores = target_model(x_batch)
        preds = scores.max(1)[1]
        y = y_batch.max(1)[1]
        Y_hat.append(y.data)
    Y_hat = [item for sublist in Y_hat for item in sublist]
    Y_hat = to_categorical(Y_hat, 10)

    start = 0
    end = start_num
    print('\n***** CHEKING TEST DATA *****\n')
    for epoch in range(epoch_num):
        print('===== total:%d, num of queries:%d =====' % (end, end-start))
        eval_prob(sess, X_test[:end].transpose(0, 2, 3, 1), Y_hat[:end], 'attack', beta)
        if epoch == 0:
            start = end
            end = end + start_num
        else:
            tmp = start
            start = end
            end = end + (end-tmp)*2
        sys.stdout.flush()

if __name__ == '__main__':
    data_name = 'all'
    vae_type = 'F'
    attack_name = sys.argv[1]
    start_num = int(sys.argv[2])
    epoch_num = int(sys.argv[3])
    main(attack_name, start_num, epoch_num, data_name, vae_type, dimZ, dimH, n_iter, batch_size, K, checkpoint, categorical=True)
