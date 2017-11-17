"""
Loads data and runs code
"""
import numpy as np

from keras_train import train, evaluate, predict
from load_data import load_data

if __name__=='__main__':
    X_train, y_train, X_val, y_val, X_test = load_data()

    # fit the model
    model, history = train(X_train, y_train, X_val, y_val)

    # save results from training
    loss = np.array(history.history['loss'])
    acc = np.array(history.history['acc'])
    val_loss = np.array(history.history['val_loss'])
    val_acc = np.array(history.history['val_acc'])
    val_top_k = np.array(history.history['val_top_k_categorical_accuracy'])

    np.savetxt("loss.csv", loss, delimiter=",")
    np.savetxt("acc.csv", acc, delimiter=",")
    np.savetxt("val_loss.csv", val_loss, delimiter=",")
    np.savetxt("val_acc.csv", val_acc, delimiter=",")
    np.savetxt("val_top_k.csv", val_top_k, delimiter=",")

    # make predictions on test data
    results = predict(X_test, model)

    # save results from prediction
    np.savetxt("results.csv", results, delimiter=",")