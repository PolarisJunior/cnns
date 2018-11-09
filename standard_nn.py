import numpy as np

# sigmoid := 1/(1 + e^-x)
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

class NeuralNetwork():
    # num_neurons := list containing # of neurons in each layer
    def __init__(self, num_neurons):
        self.num_layers = len(num_neurons)
        self.num_neurons = num_neurons
        # 1 for each neuron in each layer except input layer
        self.biases = [np.random.randn(n, 1) for n in num_neurons[1:]]
        """ each neuron in each layer is connected to each neuron in the next layer
            so the number of weights in layer_i = neurons_i * neurons_(i+1)
            also note that one row corresponds to """
        self.weights = [np.random.randn(y, x) 
            for x, y in zip(num_neurons[:-1], num_neurons[1:])]

    # feedforward the input
    def predict(self, input):
        for b, w in zip(self.biases, self.weights):
            input = sigmoid(w @ input + b) 
        return input
    
    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        # update epoch every len(training_data)/mini_batch_size iterations
        for i in range(epochs):
            np.random.shuffle(training_data)
            # separate into batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("epoch {0}".format(i))

    """ Updates the weights and biases from the mini_batch """    
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
             delta_nabla_b, delta_nabla_w = self.backprop(x, y)
             nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
             nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        pass
       
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward
        delta = self.cost_derivative(activations[-1], y) * deriv_sigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = deriv_sigmoid(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        return nabla_b, nabla_w
        pass

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    """ return # of correct predictions. Prediction is output neuron with
        highest activation """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y)
                        for (x, y) in test_data]
        return np.sum(int(x == y) for (x, y) in test_results)
