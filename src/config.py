"""
Variables for configuration
"""
from model import *
from model_resnet import *
from model_densenet import *

### data sources
data_root = '../data/images'
data_train_list = '../data/train.txt'
data_val_list = '../data/val.txt'
data_test_list = '../data/test.txt'

size = 128 # image side length
h5_train = 'miniplaces_{}_train.h5'.format(size)
h5_val = 'miniplaces_{}_val.h5'.format(size)
h5_test = 'miniplaces_{}_test.h5'.format(size)

### cnn model parameters
filters = 32 # base number of filters (scaled up or down)

reg = 0.01 # regularization weight
p_dropout = 0.8 # probability of keeping units

dense_units = 1000

optimizer = 'adam'

### training parameters
batch_size = 64
epochs = 30

rotation = 30 # rotate up to 30 either way

val_split = 0.1 # unused atm
patience = 2 # epochs before early stopping

# reusable model
classes = 100
img_dim = (size, size, 3)
model = lambda: create_dense_net(classes, img_dim,dropout_rate=p_dropout)

# model = ResNet50



