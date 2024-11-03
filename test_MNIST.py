import numpy as np

from network import Network
from layer import Dense, Activation
from activation import relu, d_relu, softmax, d_softmax, sigmoid, d_sigmoid, tanh, d_tanh
from loss_function import mse_loss, d_mse_loss
from torchvision.datasets import MNIST
from utils import one_hot_encode
# Charger le dataset MNIST
train_set = MNIST('./data', train=True, download=False)
test_set = MNIST('./data', train=False, download=False)

x_train, y_train = train_set.data.numpy(), train_set.targets.numpy()
x_test, y_test = test_set.data.numpy(), test_set.targets.numpy()

# # Afficher l'image et le label
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], cmap='gray')
# plt.show()
# print(f'Label: {y_train[0]}')

x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1] * x_train.shape[2])
x_train = x_train.astype('float32')
x_train /= 255
y_train = one_hot_encode(y_train, 10)

x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1] * x_test.shape[2])
x_test = x_test.astype('float32')
x_test /= 255
y_test = one_hot_encode(y_test, 10)

# network
net = Network()
net.add(Dense(784, 128))
net.add(Activation(relu, d_relu))
net.add(Dense(128, 50))
net.add(Activation(relu, d_relu))
net.add(Dense(50, 10))
net.add(Activation(tanh, d_tanh))

# train
net.set_loss(mse_loss, d_mse_loss)
net.train(x_train, y_train, epochs=30, learning_rate=0.1)

# test
output = net.forward(x_test[0:3])
print("\n")
print("predicted values : ")
print(output, end="\n")
print("true values : ")
print(y_test[0:3], end="\n")
