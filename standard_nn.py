
import numpy as np

# sigmoid := 1/(1 + e^-x)
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


class CNN():
    # num_neurons := list containing # of neurons in each layer
    def __init__(self, num_neurons):
        self.num_layers = len(num_neurons)
        self.num_neurons = num_neurons

    def predict(self, input):
        pass


