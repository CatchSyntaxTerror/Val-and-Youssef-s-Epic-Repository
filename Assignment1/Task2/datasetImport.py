import pandas as pd
import numpy as np

"""
Todo: Turn to objects or some shit. 
"""

# wine
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data',
    header=None
)

df_wine.columns = ['Class label', 'Alcohol',
    'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
]

df_wine = df_wine[df_wine['Class label'].isin([1, 2])]

X_wine = df_wine.iloc[:, 1:].values
y_wine = np.where(df_wine['Class label'].values == 2, 1, 0)

print("Wine labels:", np.unique(y_wine, return_counts=True))
print("Wine X shape:", X_wine.shape)


# iris
df_iris = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data',
    header=None
)

df_iris.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

df_iris = df_iris.dropna(subset=['class'])

df_iris = df_iris[df_iris['class'].isin(['Iris-setosa', 'Iris-versicolor'])]

X_iris = df_iris.iloc[:, 0:4].values
y_iris = np.where(df_iris['class'].values == 'Iris-versicolor', 1, 0)

print("Iris labels:", np.unique(y_iris, return_counts=True))
print("Iris X shape:", X_iris.shape)
