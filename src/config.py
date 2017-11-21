"""
Variables for configuration
"""
from model import *
from model_resnet import *
from model_resnet_inception import *
from model_densenet import *
from model_resnet152 import *

from keras.optimizers import SGD

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
bn_axis = 3

filters = 32 # base number of filters (scaled up or down)

reg = 0.01 # regularization weight
p_dropout = 0.5 # probability of keeping units

dense_units = 1000

optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)# 'sgd'

### training parameters
batch_size = 64
epochs = 20

rotation = 50 # rotate up to 30 either way

val_split = 0.1 # unused atm
patience = 2 # epochs before early stopping

# reusable model
classes = 100
img_dim = (size, size, 3)
# model = lambda: create_dense_net(classes, img_dim, dropout_rate=p_dropout)

# model = lambda: ResNet50(reg=False)

model = ResNet152 # InceptionResNetV2

