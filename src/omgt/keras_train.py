"""
Code for running and training model
"""

from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from model import vgg19_cascade_model
from load_data import load_data

def preprocess_data(X, y, batch_size):
    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # actual transformation step
    datagen.fit(X)

    # input into model fit
    return datagen.flow(X, y, batch_size)

def train(batch_size=32, epochs=10, split=0.2):
    """
    Trains our CNN

    :param batch_size: size of minibatch
    :param epochs: number of epochs to train for
    :param split: fraction to use as validation data
    """
    # create base model
    model = vgg19_cascade_model()

    # callbacks for training
    cb_early_stop = EarlyStopping(monitor="val_loss", patience=2)
    cb_checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    cb_csv = CSVLogger("training.log")

    # preprocessed data niceness
    X, y = load_data()
    inputs = preprocess_data(X, y, batch_size)

    # fit using specified batches with data augmentation
    model.fit_generator(
        inputs, epochs=epochs,
        verbose=0, validation_split=split,
        callbacks=[cb_early_stop, cb_checkpoint, cb_csv],
        steps_per_epoch=None, validation_steps=None)
    # steps_per_epoch might be used for regularization

def evaluate(X, y, model):
    return model.evaluate(X, y)

def predict(X, model):
    return model.predict(X)

