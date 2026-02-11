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

x_wine = df_wine.iloc[:, 1:].values
y_wine = np.where(df_wine['Class label'].values == 2, 1, 0)

print("Wine labels:", np.unique(y_wine, return_counts=True))
print("Wine X shape:", x_wine.shape)


# iris
df_iris = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data',
    header=None
)

df_iris.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

df_iris = df_iris.dropna(subset=['class'])

df_iris = df_iris[df_iris['class'].isin(['Iris-setosa', 'Iris-versicolor'])]

x_iris = df_iris.iloc[:, 0:4].values
y_iris = np.where(df_iris['class'].values == 'Iris-versicolor', 1, 0)

print("Iris labels:", np.unique(y_iris, return_counts=True))
print("Iris X shape:", x_iris.shape)

def sigmoid (a) :
    # Sigmoid function:
    # Normalizes elements between 0 and 1
    # Maintains order, so if x > y then
    # sigmoid(x) > sigmoid(y)
    return np.reciprocal(np.exp(-1 * a) + 1)

def normalized(a) :
    # Normalization function
    # Normalizes elements between 0 and 1
    # Maintains order, so if x > y then
    # normalized(x) > normalized(y)
    return (a - np.min(a))/(np.max(a) - np.min(a))

def normalized(a,scale) :
    # Scales the regular normalization function by specified value
    return scale * (a - np.min(a))/(np.max(a) - np.min(a))


print("wine x:",x_wine)
print("sigmoid x:",sigmoid(x_wine))
print("normalized x:",normalized(x_wine))