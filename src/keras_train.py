"""
Code for running and training model
"""

from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from model import vgg19_cascade_model
from load_data import load_data
import config

#### UNUSED AS OF NOW BECAUSE TOO MUCH MEMORY CONSUMPTION
def preprocess_data(X, y):
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
    return datagen.flow(X, y, config.batch_size)

def train():
    """
    Trains our CNN
    """
    # create base model
    model = vgg19_cascade_model()

    # callbacks for training
    cb_early_stop = EarlyStopping(monitor="val_loss", patience=config.patience)
    cb_checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    cb_csv = CSVLogger("training.log")

    # preprocessed data niceness
    X, y = load_data()
    inputs = preprocess_data(X, y)

    print("# Data loaded. Beginning training.")

    # fit using specified batches with data augmentation
    model.fit(X, y, batch_size=config.batch_size, epochs=config.epochs,
        verbose=1, validation_split=config.split,
        callbacks=[cb_early_stop, cb_checkpoint, cb_csv])

def evaluate(X, y, model):
    return model.evaluate(X, y)

def predict(X, model):
    return model.predict(X)

