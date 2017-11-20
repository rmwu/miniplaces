"""
Loads data and runs code
"""
import sys

import numpy as np

from keras_train import train, evaluate
from load_data import load_data

from model_resnet import ResNet50
import config

def predict(X, model):
    results = model.predict(X)
    return results

def test_model(X_test, model, weights_path=None):
    # make predictions on test data
    results = predict(X_test, model)

    # save results from prediction
    np.savetxt("results.csv", results, delimiter=",")

    # submission friendly format
    prepare_submission(results)

def prepare_submission(results, weights_path=None):
    """
    :param results: softmax output on test set
    """
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

def run_past_model(weights_path, model=None):
    X_test = load_data(test_only=True)

    # recreate the model
    if model is None:
        m = config.model()
    else:
        m = model()
    m.load_weights(weights_path)

    test_model(X_test, m, weights_path)

def ensemble_models(weights, models, contributions=None):
    """
    :param weights: list of hdf5 neural net weights
    :param models: list of functions that return corresponding models
    :param contributions: relative weight of models
    """
    X_test = load_data(test_only=True)

    cumulative = None # stores results from all models

    if contributions is None: # if none, consider all equal
        contributions = [1] * len(weights)

    for w, m, c in zip(weights, models, contributions):
        model = m()
        model.load_weights(w)
        results = predict(X_test, model)

        if cumulative is None:
            cumulative = results
        else:
            assert cumulative.shape == results.shape # same dataset
            cumulative *= results * c

    prepare_submission(cumulative, '20171120-ensemble')

# 0.289 top 5
def current_ensemble():
    # filenames
    weights = [
        '20171120-RN50-weights.09-2.81.hdf5',
        '20171120-RN50Real-weights.15-2.42.hdf5',
        '20171120-RN50Reg-weights.05-3.26.hdf5']

    # must be functions
    models = [
      lambda: ResNet50(reg=False, deeper=False),
      lambda: ResNet50(reg=False),
      ResNet50]

    # accuracies were 0.63 0.69 0.5?
    contributions = [2,4,1]

    ensemble_models(weights, models, contributions)

if __name__=='__main__':
    X_train, y_train, X_val, y_val, X_test = load_data()

    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    else:
        weights_path = None

    # fit the model
    model, history = train(X_train, y_train, X_val, y_val, weights_path)

    # save results from training
    loss = np.array(history.history['loss'])
    acc = np.array(history.history['acc'])
    top_k = np.array(history.history['top_k_categorical_accuracy'])
    val_loss = np.array(history.history['val_loss'])
    val_acc = np.array(history.history['val_acc'])
    val_top_k = np.array(history.history['val_top_k_categorical_accuracy'])

    np.savetxt("loss.csv", loss, delimiter=",")
    np.savetxt("acc.csv", acc, delimiter=",")
    np.savetxt("top_k.csv", top_k, delimiter=",")
    np.savetxt("val_loss.csv", val_loss, delimiter=",")
    np.savetxt("val_acc.csv", val_acc, delimiter=",")
    np.savetxt("val_top_k.csv", val_top_k, delimiter=",")

    # make predictions on test data
    test_model(X_test, model)

