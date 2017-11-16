"""
Code for loading in data
"""

import os
import numpy as np
import scipy.misc
import h5py

from DataLoader import *
import config

# Dataset Parameters
load_size = config.size
fine_size = config.size-31
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

classes = 100

# Construct dataloader
opt_data_train = {
    'data_h5': config.h5_train,
    'data_root': config.data_root,
    'data_list': config.data_train_list,
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': config.h5_val,
    'data_root': config.data_root,
    'data_list': config.data_train_list,
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

def load_data():

    loader_train = DataLoaderH5(**opt_data_train)
    # loader_val = DataLoaderH5(**opt_data_val)

    X_train, y_train = loader_train.get_data()

    # normalize X
    X_train = X_train.astype(np.float32) / 255.

    # change y to one-hot
    n = y_train.shape[0]
    y_onehot = np.zeros((n, classes))
    y_onehot[np.arange(n), y_train] = 1

    return X_train, y_onehot

