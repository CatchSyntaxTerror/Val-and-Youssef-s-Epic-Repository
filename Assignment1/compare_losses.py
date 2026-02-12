
import matplotlib.pyplot as plt

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
ada_wine = ModifiedAdalineGD(eta=0.01, n_iter=50, random_state=1)
ada_wine.fit(X_wine, y_wine)
log_wine = ModifiedLogisticRegressionGD(eta=0.01, n_iter=50, random_state=1)
log_wine.fit(X_wine, y_wine)

print("Adaline final loss:", ada_wine.losses_[-1])
print("Adaline epochs:", len(ada_wine.losses_))
print("Logistic final loss:", log_wine.losses_[-1])
print("Logistic epochs:", len(log_wine.losses_))

X_iris, y_iris = load_iris_data()
ada_iris = ModifiedAdalineGD(eta=0.01, n_iter=50, random_state=1)
ada_iris.fit(X_iris, y_iris)
log_iris = ModifiedLogisticRegressionGD(eta=0.01, n_iter=50, random_state=1)
log_iris.fit(X_iris, y_iris)

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

