import numpy as np

def mse_loss(output, target):
    return np.mean(np.power(target-output, 2))

def d_mse_loss(output, target):
    return np.squeeze(2*(output - target)/output.size)

def cross_entropy_loss(output, target):
    return -np.sum(target*np.log(output))

def d_cross_entropy_loss(output, target):
    return -target/output