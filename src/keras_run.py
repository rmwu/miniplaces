"""
Loads data and runs code
"""
import numpy as np

from keras_train import train, evaluate, predict
from load_data import load_data
from model import *
from model_resnet import *

def test_model(X_test, model, weights_path = None):
    # make predictions on test data
    results = predict(X_test, model)

    # save results from prediction
    np.savetxt("results.csv", results, delimiter=",")

    # fetch top 5 indices
    ind=np.argsort(results,axis=1)[:,-5:][:,::-1]

    # now write the submission file
    if weights_path:
        filename = 'submit-{}.txt'.format(weights_path)
    else:
        filename = 'submit.txt'
    with open(filename, 'w+') as f:
        f.write('')

    with open(filename, 'a') as f:
        for x in range(10000):
            path = 'test/' + str(x+1).zfill(8)[-8:] + '.jpg'
            labels = str(ind[x])[1:-1] # cut off [] lol
            f.write(path + ' ' + labels + '\n')

def run_past_model(weights_path):
    _, _, _, _, X_test = load_data()

    # recreate the model
    model = ResNet50()
    model.load_weights(weights_path)

    test_model(X_test, model, weights_path)

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
    test_model(X_test, model)

