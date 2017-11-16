"""
Variables for configuration
"""

### data sources
data_root = '../../data/images'
data_train_list = '../../data/train_b.txt'
data_val_list = '../../data/val.txt'

size = 128 # image side length
h5_train = 'miniplaces_{}_train.h5'.format(size)
h5_val = 'miniplaces_{}_val.h5'.format(size)

### cnn model parameters
filters = 32 # base number of filters (scaled up or down)

reg = 0.01 # regularization weight
p_dropout = 0.5 # probability of keeping units

optimizer = 'rmsprop'

### training parameters
batch_size = 8
epochs = 5
val_split = 0.2
patience = 2 # epochs before early stopping
