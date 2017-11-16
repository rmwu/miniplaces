"""
Code for loading in data
"""

import os
import numpy as np
import scipy.misc
import h5py

from DataLoader import *

# Dataset Parameters
load_size = 256
fine_size = 224
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

def load_data():

    loader_train = DataLoaderH5(**opt_data_train)
    # loader_val = DataLoaderH5(**opt_data_val)

    X_train, y_train = loader_train.get_data()

    X_train.astype(np.float32) /= 255.

    return X_train, y_train

