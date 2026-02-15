
import matplotlib.pyplot as plt
import cerberus as c
import numpy as np

from ModifiedAdalineGD import ModifiedAdalineGD
from ModifiedLogisticRegressionGD import ModifiedLogisticRegressionGD
from datasetImport import load_wine_data, load_iris_data

"""
You need matplotlib to run.
use: python -m pip install matplotlib

This plots the adaline and logistic regression
"""

# wine data sets 
X_wine, y_wine = load_wine_data()
ada_wine = ModifiedAdalineGD(eta=0.01, n_iter=10000, random_state=1)
ada_wine.fit(X_wine, y_wine)
log_wine = ModifiedLogisticRegressionGD(eta=0.01, n_iter=10000, random_state=1)
log_wine.fit(X_wine, y_wine)

X_iris, y_iris = load_iris_data()
ada_iris = ModifiedAdalineGD(eta=0.01, n_iter=10000, random_state=1)
ada_iris.fit(X_iris, y_iris)
log_iris = ModifiedLogisticRegressionGD(eta=0.01, n_iter=10000, random_state=1)
log_iris.fit(X_iris, y_iris)

X_cerberus, y_cerberus = c.load_iris_all()

p_setosa = c.Perceptron()
p_versicolor = c.Perceptron()
p_virginica = c.Perceptron()
p_setosa.fit(X_cerberus, (y_cerberus == "Iris-setosa").astype(int))
p_versicolor.fit(X_cerberus, (y_cerberus == "Iris-versicolor").astype(int))
p_virginica.fit(X_cerberus, (y_cerberus == "Iris-virginica").astype(int))
models = [p_setosa, p_versicolor, p_virginica]
class_names = np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
preds = c.predict_multiclass(X_cerberus, models, class_names)


# wine
plt.figure()
plt.plot(ada_wine.losses_)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Wine — Adaline Loss")
plt.savefig("images/wine_adaline_loss.png")
plt.close()

plt.figure()
plt.plot(log_wine.losses_)
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Wine — Logistic Regression Loss")
plt.savefig("images/wine_logistic_loss.png")
plt.close()


# iris
plt.figure()
plt.plot(ada_iris.losses_)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Iris — Adaline Loss")
plt.savefig("images/iris_adaline_loss.png")
plt.close()

plt.figure()
plt.plot(log_iris.losses_)
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Iris — Logistic Regression Loss")
plt.savefig("images/iris_logistic_loss.png")
plt.close()

# errors from 3 label classification. [Task 3]
plt.xlabel("Epoch")
plt.ylabel("Number of Misclassifications")
plt.title("Perceptron Training Erros")
plt.plot(p_setosa.errors_, label="Setosa")
plt.plot(p_versicolor.errors_, label="Versicolor")
plt.plot(p_virginica.errors_, label="Virginica")
plt.legend()
plt.savefig("images/perceptron_erros.png")
plt.close()