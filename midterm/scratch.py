import src.data_loader as dl
import src.graphing as gr

"""
I'm just using this to test scripts so we can delete the file instead of having to clean up all out code
"""

X_mnist_train, y_mnist_train, X_mnist_test, y_mnist_test = dl.load_mnist()
gr.record_raw_data(y_mnist_train, y_mnist_test, True)

X_fashion_train, y_fashion_train, X_fashion_test, y_fashion_test = dl.load_fashion_mnist()
gr.record_raw_data(y_fashion_train, y_fashion_test, False)