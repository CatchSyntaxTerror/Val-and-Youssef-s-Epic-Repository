import numpy as np
import pandas as pd
import normalization_utils as norm
import matplotlib.pyplot as plt

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
    The Books perceptron object modified
    removed bias by appending a 1 to each row.  
    """

    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=X.shape[1] + 1)        
        self.errors_ = []
        
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * np.append(xi, 1.0)
                errors += int(update != 0)
            self.errors_.append(errors)
        return self
         
    def net_input(self, X):
        if X.ndim == 1:
            X_aug = np.append(X, 1.0)
        else:
            X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.dot(X_aug, self.w_)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# Instantiate three perceptrons. One for each type of flower. 
# Each perceptron is responsible for identifying its given flower
X, y = load_iris_all()

p_setosa = Perceptron()
p_versicolor = Perceptron()
p_virginica = Perceptron()

# train the data
p_setosa.fit(X, (y == "Iris-setosa").astype(int))
p_versicolor.fit(X, (y == "Iris-versicolor").astype(int))
p_virginica.fit(X, (y == "Iris-virginica").astype(int))

# compare confidence of each model. 
# pick the model that was most confident. 
# returns an array of indexes that map to the model of most confdence.
def predict_multiclass(X, models, class_names):
    scores = np.vstack([m.net_input(X) for m in models])
    winner = np.argmax(scores, axis=0)
    return class_names[winner]

# print the number of times each flower appears in the dataset.
models = [p_setosa, p_versicolor, p_virginica]
class_names = np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
preds = predict_multiclass(X, models, class_names)
classes, counts = np.unique(preds, return_counts=True)
for i in range(len(classes)):
    print(classes[i], ":", counts[i])

# print the errors and graph them. 
print("Setosa errors per epoch:", p_setosa.errors_)
print("Versicolor errors per epoch:", p_versicolor.errors_)
print("Virginica errors per epoch:", p_virginica.errors_)

plt.xlabel("Epoch")
plt.ylabel("Number of Misclassifications")
plt.title("Perceptron Training Erros")
plt.plot(p_setosa.errors_, label="Setosa")
plt.plot(p_versicolor.errors_, label="Versicolor")
plt.plot(p_virginica.errors_, label="Virginica")
plt.legend()
plt.savefig("images/perceptron_erros.png")
plt.show()