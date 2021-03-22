from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import warnings

import os, socket
from sklearn import datasets

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def data_pdf(datadir='/home/yz/data/traintest_all_500test/',
            train_start=0, train_end=13190, test_start=0, test_end=6146,
            attack_start=0, attack_end=1600):
    """
    Load and preprocess PDF dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    train_data = datadir+'train_data.libsvm'
    X_train, Y_train = datasets.load_svmlight_file(train_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)

    X_train = X_train.toarray()

    test_data = datadir+'test_data.libsvm'
    X_test, Y_test = datasets.load_svmlight_file(test_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)
    X_test = X_test.toarray()

    attack_data = '/home/yz/code/ext/models/steal_baseline_v2.queries.libsvm'
    X_attack, Y_attack = datasets.load_svmlight_file(attack_data,
                                       n_features=3514,
                                       multilabel=False,
                                       zero_based=False,
                                       query_id=False)
    X_attack = X_attack.toarray()

    # train = np.loadtxt(datadir+"train_data.csv", delimiter=",")
    # test = np.loadtxt(datadir+"test_data.csv", delimiter=",")
    # attack = np.loadtxt('/home/yz/code/ext/models/'+"steal_baseline_v2.csv", delimiter=",")
    # X_train = train[:, 1:]
    # Y_train = train[:, :1].flatten()
    # X_test = test[:, 1:]
    # Y_test = test[:, :1].flatten()
    # X_attack = attack[:, 1:]
    # Y_attack = attack[:, :1].flatten()
    #
    # X_train = X_train[train_start:train_end]
    # Y_train = Y_train[train_start:train_end]
    # X_test = X_test[test_start:test_end]
    # Y_test = Y_test[test_start:test_end]
    # X_attack = X_attack[attack_start:attack_end]
    # Y_attack = Y_attack[attack_start:attack_end]

    Y_train = to_categorical(Y_train, 2)
    Y_test = to_categorical(Y_test, 2)
    Y_attack = to_categorical(Y_attack, 2)

    print('PDF X_train shape:', X_train.shape)
    print('PDF X_test shape:', X_test.shape)
    print('PDF X_attack shape:', X_attack.shape)

    return X_train, Y_train, X_test, Y_test, X_attack, Y_attack
