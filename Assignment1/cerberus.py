import numpy as np
import pandas as pd
import normalization_utils as norm

"""
The beast of legend! The three neuron beast!!!
"""


def load_iris_all():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    df.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    df = df.dropna(subset=["class"])

    X = df.iloc[:, 0:4].values
    y = df["class"].values
    X = norm.normalized(X)

    return X, y


class Perceptron:
    """
    Based off AdalineGD except it applies a hard cutoff and
    only updates when its wrong
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=X.shape[1] + 1)

        for i in range(self.n_iter):
            for xi, target in zip(X, y):  # lol haskell
                xi_aug = np.append(xi, 1.0)
                pred = 1 if np.dot(xi_aug, self.w_) >= 0.0 else 0
                update = self.eta * (target - pred)
                self.w_ += update * xi_aug
        return self

    def net_input(self, X):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.dot(X_aug, self.w_)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# train the perceptrons
X, y = load_iris_all()

p_setosa = Perceptron()
p_versicolor = Perceptron()
p_virginica = Perceptron()

p_setosa.fit(X, (y == "Iris-setosa").astype(int))
p_versicolor.fit(X, (y == "Iris-versicolor").astype(int))
p_virginica.fit(X, (y == "Iris-virginica").astype(int))


def predict_multiclass(X, models, class_names):
    scores = np.vstack([m.net_input(X) for m in models])
    winner = np.argmax(scores, axis=0)
    return class_names[winner]


models = [p_setosa, p_versicolor, p_virginica]
class_names = np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

preds = predict_multiclass(X, models, class_names)

classes, counts = np.unique(preds, return_counts=True)
for i in range(len(classes)):
    print(classes[i], ":", counts[i])
