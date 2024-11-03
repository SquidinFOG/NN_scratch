import numpy as np

from network import Network
from layer import Dense, Activation
from activation import tanh, d_tanh
from loss_function import mse_loss, d_mse_loss

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(Dense(2, 3))
net.add(Activation(tanh, d_tanh))
net.add(Dense(3, 1))
net.add(Activation(tanh, d_tanh))

# train
net.set_loss(mse_loss, d_mse_loss)
net.train(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.forward(x_train)
print(out)