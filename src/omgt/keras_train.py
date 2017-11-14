
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras import regularizers

def vgg19_modify_model():
    """
    Modification of vgg19 with regularization, adjustable number of filters
    """
    model = Sequential()

    # images come in 128 by 128 with 3 channels

    filters = 32 # num filters for scaling up and down
    classes = 100 # number of scene classes

    reg = 0.01 # regularization constant
    p_dropout = 0.5 # probability of dropout

    # padding 'same' automatically zero-pads
    # TODO I want to use functional API to merge models
    # a la ensemble NN
    model.add(Convolution2D(filters, (3,3), padding="same", activation="relu", input_shape=(128,128,3)))
    model.add(Convolution2D(filters, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(2*filters, (3,3), padding="same", activation="relu"))
    model.add(Convolution2D(2*filters, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4*filters, (3,3), padding="same", activation="relu")),
    model.add(Convolution2D(4*filters, (3,3), padding="same", activation="relu"))
    model.add(Convolution2D(4*filters, (3,3), padding="same", activation="relu"))
    model.add(Convolution2D(4*filters, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4*filters, (3,3), padding="same", activation="relu"))
    model.add(Convolution2D(4*filters, (3,3), padding="same", activation="relu"))
    model.add(Convolution2D(4*filters, (3,3), padding="same", activation="relu"))
    model.add(Convolution2D(4*filters, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten()) # TODO figure out dimensions
    model.add(Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg)))
    model.add(Dropout(p_dropout))
    model.add(Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg)))
    model.add(Dropout(p_dropout))
    model.add(Dense(classes, activation="softmax")) # output layer

    # optimizers: adam, rmsprop, sgd
    model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def train(batch=32):
    model = vgg19_modify_model()





