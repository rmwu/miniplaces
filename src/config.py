"""
Variables for configuration
"""

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
p_dropout = 0.5 # probability of keeping units

optimizer = 'adam'

### training parameters
batch_size = 32
epochs = 30
val_split = 0.1 # unused atm
patience = 2 # epochs before early stopping
