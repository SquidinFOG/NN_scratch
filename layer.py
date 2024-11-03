import numpy as np
# Base Layer class for all layers in the network

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad, learning_rate):
        raise NotImplementedError
    

class Dense(Layer):
    def __init__(self,input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        self.input = np.array(input_data)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output 
    def backward(self, output_grad, learning_rate):
        input_grad = np.dot(output_grad, self.weights.T)
        weights_grad = np.dot(self.input.T, output_grad)
        bias_grad = output_grad

        # Update weights and bias (maybe in an other method)
        self.weights -= learning_rate * weights_grad
        self.bias -= learning_rate * bias_grad
        return input_grad
    
class Activation(Layer):
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    def forward(self, input_data):
        self.input = np.array(input_data)
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_grad, learning_rate):
        return self.d_activation(self.input) * output_grad
    

    