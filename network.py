import numpy as np
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.d_loss = None
    
    def add(self, layer):
        self.layers.append(layer)
    
    def set_loss(self, loss, d_loss):
        self.loss = loss
        self.d_loss = d_loss

    def forward(self, input_data):
                
        sample_size = len(input_data)
        result = []
        for i in range(sample_size):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        
        return np.array(result)

    def backward(self, output_grad, learning_rate):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad, learning_rate)
            
            

    def train(self, x_train, y_train, epochs, learning_rate):
        sample_size = len(x_train)
        for i in range(epochs):
            epoch_loss = 0
            for j in range(sample_size):
                # Forward pass
                output = self.forward([x_train[j]])
                epoch_loss += self.loss(output, y_train[j])
                # Backward pass
                output_grad = self.d_loss(output, y_train[j])
                self.backward(output_grad, learning_rate)
            epoch_loss /= sample_size
            print(f'Epoch {i+1}/{epochs} Loss: {epoch_loss}')


