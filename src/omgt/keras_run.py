"""
Loads data and runs code
"""
import numpy as np

from keras_train import train, evaluate, predict
from load_data import load_data


train(batch_size=8, epochs=2, split=0.2)
