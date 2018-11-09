
from data_loader import *
from standard_nn import NeuralNetwork

network = NeuralNetwork([784, 30, 10])

mndata = load_mnist()

training_dat = mnist_training(mndata)
testing_dat = mnist_training(mndata)
training_vectorized = mnist_training_vectorized(mndata)

network.SGD(training_vectorized, 1, 16, 3.0)
print(network.evaluate(testing_dat))

