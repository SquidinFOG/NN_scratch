import numpy as np

def one_hot_encode(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])