import src


def main():
    X_train, y_train, y_test, num_bags = src.load_mnist()
    str = input()
    src.run_bagging(X_train, y_train, y_test, num_bags)

#Todo: finish setting up main file
