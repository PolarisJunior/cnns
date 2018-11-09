
import numpy as np
from mnist import MNIST

MNIST_SCALE = 255.0
MNIST_NUM_FEATURES = 784

def vectorize_label(j):
    res = np.zeros((10, 1))
    res[j] = 1.0
    return res

def load_mnist():
    return MNIST('./data/')

def shape_mnist_data(data):
    return [(np.reshape(x, (MNIST_NUM_FEATURES, 1)), y) for x, y in data]

def mnist_training(mndata):
    X_train, labels_train = map(np.array, mndata.load_training())
    X_train = X_train / MNIST_SCALE
    return shape_mnist_data(list(zip(X_train, labels_train)))

def mnist_testing(mndata):
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_test = X_test / MNIST_SCALE
    return shape_mnist_data(list(zip(X_test, labels_test)))

def mnist_training_vectorized(mndata):
    training_data = mnist_training(mndata)
    return [(x, vectorize_label(y)) for x, y in training_data]