"""
CNN models, code to run
"""

from keras.models import Model
from keras.optimizers import Adam

from keras.layers.merge import add
from keras.layers import Input, concatenate, BatchNormalization
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution2D,MaxPooling2D

from keras import regularizers

import config

def vgg19_resnet_model():
    """
    Modification of vgg19:
    - output a fully dense layer from each convolution, concatenated
      onto inputs of final softmax prediction
    - added regularization (dropout, weights)
    - adjustable number of filters for convenience
    """
    # images come in 128 by 128 with 3 channels
    # TODO figure out dimensions for each

    filters = config.filters # num filters for scaling up and down
    classes = 100 # number of scene classes

    reg = config.reg # regularization constant
    p_dropout = config.p_dropout # probability of keeping units

    inputs = Input(shape=(config.size,config.size,3))
    # padding 'same' automatically zero-pads
    x = Convolution2D(filters, (3,3), padding="same", activation="relu")(inputs)
    x = Convolution2D(filters, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    xc = Convolution2D(2*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(2*filters, (3,3), padding="same", activation="relu")(xc)

    x = add([xc, x])

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    xc = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(xc)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)

    x = add([xc, x])

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # second intermediate outputs
    m2 = BatchNormalization()(x)
    m2 = Flatten()(m2)
    m2 = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg), name="m2-dense")(m2)
    m2 = Dropout(p_dropout)(m2)

    xc = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(xc)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)

    x = add([xc, x])

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # final outputs
    x = BatchNormalization()(x)
    x = Flatten()(x) # TODO figure out dimensions
    x = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg), name="m3-dense1")(x)
    x = Dropout(p_dropout)(x)
    x = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg), name="m3-dense2")(x)
    x = Dropout(p_dropout)(x)

    # concatenate the three outputs together
    merged = concatenate([m2, x], axis=-1)
    prediction = Dense(classes, activation="softmax", name="softmax-output")(merged)

    model = Model(inputs=inputs, outputs=prediction)

    return model

def vgg19_cascade_model():
    """
    Modification of vgg19:
    - output a fully dense layer from each convolution, concatenated
      onto inputs of final softmax prediction
    - added regularization (dropout, weights)
    - adjustable number of filters for convenience
    """
    # images come in 128 by 128 with 3 channels
    # TODO figure out dimensions for each

    filters = config.filters # num filters for scaling up and down
    classes = 100 # number of scene classes

    reg = config.reg # regularization constant
    p_dropout = config.p_dropout # probability of keeping units

    inputs = Input(shape=(config.size,config.size,3))
    # padding 'same' automatically zero-pads
    x = Convolution2D(filters, (3,3), padding="same", activation="relu")(inputs)
    x = Convolution2D(filters, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Convolution2D(2*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(2*filters, (3,3), padding="same", activation="relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # first intermediate outputs
    # m1 = BatchNormalization()(x)
    # m1 = Flatten()(m1)
    # m1 = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg), name="m1-dense")(m1)
    # m1 = Dropout(p_dropout)(m1)

    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # second intermediate outputs
    m2 = BatchNormalization()(x)
    m2 = Flatten()(m2)
    m2 = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg), name="m2-dense")(m2)
    m2 = Dropout(p_dropout)(m2)

    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)
    x = Convolution2D(4*filters, (3,3), padding="same", activation="relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # final outputs
    x = BatchNormalization()(x)
    x = Flatten()(x) # TODO figure out dimensions
    x = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg), name="m3-dense1")(x)
    x = Dropout(p_dropout)(x)
    x = Dense(filters**2, activation="relu", kernel_regularizer=regularizers.l2(reg), name="m3-dense2")(x)
    x = Dropout(p_dropout)(x)

    # concatenate the three outputs together
    # merged = concatenate([m1, m2, x], axis=-1)
    merged = concatenate([m2, x], axis=-1)
    prediction = Dense(classes, activation="softmax", name="softmax-output")(merged)

    model = Model(inputs=inputs, outputs=prediction)

    return model