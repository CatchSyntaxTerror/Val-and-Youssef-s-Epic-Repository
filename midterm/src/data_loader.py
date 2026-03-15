import os
import idx2numpy
import numpy as np

"""
This file loads the MNIST and Fashion-MNIST datasets using the idx2numpy library
I also flattened the data here before returning so X_train and X_test are 1d numpy arrays
"""

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data")

def load_idx_data(images_path, labels_path):
    """
    helper to load images and labels in lists
    """
    train_images = idx2numpy.convert_from_file(images_path)
    train_labels = idx2numpy.convert_from_file(labels_path)
    return train_images, train_labels
    
def load_mnist():
    """
    Loads the MNIST images and labeles
    Stores them in training sets and test sets 
    """
    train_images_path = os.path.join(DATA_DIR, "mnist", "train-images-idx3-ubyte")
    train_labels_path = os.path.join(DATA_DIR, "mnist", "train-labels-idx1-ubyte")
    test_images_path  = os.path.join(DATA_DIR, "mnist", "t10k-images-idx3-ubyte")
    test_labels_path  = os.path.join(DATA_DIR, "mnist", "t10k-labels-idx1-ubyte")

    X_train, y_train = load_idx_data(train_images_path, train_labels_path)
    X_test, y_test = load_idx_data(test_images_path, test_labels_path)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test


def load_fashion_mnist():
    """
    Loads the Fashion-MNIST images and labeles
    Stores them in training sets and test sets
    """
    train_images_path = os.path.join(DATA_DIR, "fashion_mnist", "train-images-idx3-ubyte")
    train_labels_path = os.path.join(DATA_DIR, "fashion_mnist", "train-labels-idx1-ubyte")
    test_images_path  = os.path.join(DATA_DIR, "fashion_mnist", "t10k-images-idx3-ubyte")
    test_labels_path  = os.path.join(DATA_DIR, "fashion_mnist", "t10k-labels-idx1-ubyte")


    X_train, y_train = load_idx_data(train_images_path, train_labels_path)
    X_test, y_test = load_idx_data(test_images_path, test_labels_path)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test

def partintion_prac_tst(X_train, chunk_number):
    """
    extract the specified chunk
    returns the chunk and X_train with the chunk removed
    """
    n = len(X_train) // 5
    start =  n * chunk_number
    end = start + n
    prac_test = X_train[start:end]
    new_trian = np.delete(X_train, list(range(start, end)))
    return prac_test, new_trian
