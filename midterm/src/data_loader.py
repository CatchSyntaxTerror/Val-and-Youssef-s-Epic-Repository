import idx2numpy

"""
This file loads the data 
"""

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
    train_images_path = "../data/mnist/train-images.idx3-ubyte"
    train_labels_path = "../data/mnist/train-labels.idx1-ubyte"
    test_images_path = "../data/mnist/t10k-images.idx3-ubyte"
    test_labels_path = "../data/mnist/t10k-labels.idx1-ubyte"

    X_train, y_train = load_idx_data(train_images_path, train_labels_path)
    X_test, y_test = load_idx_data(test_images_path, test_labels_path)

    return X_train, y_train, X_test, y_test


def load_fashion_mnist():
    """
    Loads the Fashion-MNIST images and labeles
    Stores them in training sets and test sets
    """
    train_images_path = "../data/fashion_mnist/train-images.idx3-ubyte"
    train_labels_path = "../data/fashion_mnist/train-labels.idx1-ubyte"
    test_images_path = "../data/fashion_mnist/t10k-images.idx3-ubyte"
    test_labels_path = "../data/fashion_mnist/t10k-labels.idx1-ubyte"

    X_train, y_train = load_idx_data(train_images_path, train_labels_path)
    X_test, y_test = load_idx_data(test_images_path, test_labels_path)

    return X_train, y_train, X_test, y_test