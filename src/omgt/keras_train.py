
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras import regularizers

def vgg19_cascade_model():
    """
    Modification of vgg19:
    - output a fully dense layer from each convolution, concatenated
      onto inputs of final softmax prediction
    - added regularization (dropout, weights)
    - adjustable number of filters for convenience
    """
    model = Sequential()

    # images come in 128 by 128 with 3 channels
    # TODO figure out dimensions for each

    filters = 32 # num filters for scaling up and down
    classes = 100 # number of scene classes

    reg = 0.01 # regularization constant
    p_dropout = 0.5 # probability of dropout

    inputs = Input(shape=(128,128,3))
    # padding 'same' automatically zero-pads
    x = Convolution2D(filters, (3,3), padding="same", activation="relu")(inputs)
    x = Convolution2D(filters, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(2*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(2*filters, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # first intermediate outputs
    m1 = Flatten()(x)
    m1 = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg))(m1)
    m1 = Dropout(p_dropout)(m1)

    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # second intermediate outputs
    m2 = Flatten()(x)
    m2 = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg))(m2)
    m2 = Dropout(p_dropout)(m2)

    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # final outputs
    x = Flatten()(x) # TODO figure out dimensions
    x = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg))(x)
    x = Dropout(p_dropout)(x)
    x = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg))(x)
    x = Dropout(p_dropout)(x)

    # concatenate the three outputs together
    merged = concatenate([m1, m2, x], axis=-1)
    prediction = Dense(classes, activation="softmax")(merged)

    model = Model(inputs=inputs, outputs=prediction)

    # optimizers: adam, rmsprop, sgd
    model.compile(optimizer='rmsprop', # Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def train(X, y, batch_size=32, epochs=10, split=0.2):
    """
    Trains our CNN

    :param X: dataset
    :param y: labels
    :param batch_size: size of minibatch
    :param epochs: number of epochs to train for
    :param split: fraction to use as validation data
    """
    model = vgg19_cascade_model()

    model.fit(
        x=X, y=y, batch_size=batch_size, epochs=epochs,
        verbose=0, validation_split=split,
        steps_per_epoch=None, validation_steps=None)
    # steps_per_epoch might be used for regularization

def evaluate(X, y, model):
    return model.evaluate(X, y)

def predict(X, model):
    return model.predict(X)






