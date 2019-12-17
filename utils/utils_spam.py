from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import warnings

import os, socket

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def data_spam(datadir='/home/yz/code/trees/twitter_spam/', train_start=0, train_end=295870, test_start=0,
               test_end=126082):
    """
    Load and preprocess MNIST dataset
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

    if socket.gethostname() == 'deep':
        datadir = '/home/mcz/Desktop/courses_2019F/DavidBlei/Project/'
    
    train = np.loadtxt(datadir+"twitter_spam_reduced.train.csv", delimiter=",")
    test = np.loadtxt(datadir+"twitter_spam_reduced.test.csv", delimiter=",")
    X_train = train[:, 1:]
    Y_train = train[:, :1].flatten()
    X_test = test[:, 1:]
    Y_test = test[:, :1].flatten()

    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]
    
    Y_train = to_categorical(Y_train, 2)
    Y_test = to_categorical(Y_test, 2)
 
    print('Spam X_train shape:', X_train.shape)
    print('Spam X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test

