import numpy as np

def sigmoid (a) :
    # Sigmoid function:
    # Normalizes elements between 0 and 1
    # Maintains order, so if x > y then
    # sigmoid(x) > sigmoid(y)
    return np.reciprocal(np.exp(-1 * a) + 1)

def normalized(a,scale = 1) :
    # Normalization function
    # Normalizes elements between 0 and 1
    # Maintains order, so if x > y then
    # normalized(x) > normalized(y)
    return scale * (a - np.min(a))/(np.max(a) - np.min(a))