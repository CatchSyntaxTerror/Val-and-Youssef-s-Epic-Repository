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
    y = df["class"].values  # strings
    X = norm.normalized(X)

    return X, y

