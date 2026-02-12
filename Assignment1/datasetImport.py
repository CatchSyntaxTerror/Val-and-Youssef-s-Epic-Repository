import pandas as pd
import numpy as np
import normalization_utils as norm

# loads wine data set and returns a normalized version


def load_wine_data():
    df_wine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/" "machine-learning-databases/wine/wine.data",
        header=None,
    )

    df_wine.columns = [
        "Class label",
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline",
    ]

    df_wine = df_wine[df_wine["Class label"].isin([1, 2])]

    X_wine = df_wine.iloc[:, 1:].values
    y_wine = np.where(df_wine["Class label"].values == 2, 1, 0)
    X_wine = norm.normalized(X_wine)
    return X_wine, y_wine

# loads iris data set and returns normailzed data
def load_iris_data():
    df_iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/" "machine-learning-databases/iris/iris.data",
        header=None,
    )
    df_iris.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]

    df_iris = df_iris.dropna(subset=["class"])

    df_iris = df_iris[df_iris["class"].isin(["Iris-setosa", "Iris-versicolor"])]

    X_iris = df_iris.iloc[:, 0:4].values
    y_iris = np.where(df_iris["class"].values == "Iris-versicolor", 1, 0)
    X_iris = norm.normalized(X_iris)
    return X_iris, y_iris


# X_wine, y_wine = load_wine_data()
# print("Wine data (wines, features)(labels): ", X_wine.shape, y_wine.shape)

# X_iris, y_iris = load_iris_data()
# print("Iris data (flowers, features)(labels): ",X_iris.shape, y_iris.shape)
