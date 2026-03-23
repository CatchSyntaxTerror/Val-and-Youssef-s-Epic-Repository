from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

import numpy as np
import time as time
import warnings

"""
This file handles standardization, dimensionality reduction, SVC training, tuning, and evaluation.
Here are some helpful links:
    https://scikit-learn.org/stable/modules/compose.html#pipeline
    https://scikit-learn.org/stable/modules/decomposition.html#pca
    https://scikit-learn.org/stable/modules/lda_qda.html
"""

def build_pca_model(num_comps, kernel, C, gamma=0, degree=0):
    """
    makes a pca pipeline
    """
    return Pipeline([("scaler", StandardScaler()),
                    ("pca", PCA(n_components=num_comps)),
                    ("svc", build_svc(kernel, C, gamma, degree))])


def build_lda_model(n, kernel, C, gamma=0, degree=0):
    """
    makes an lda pipeline, n is ignored
    """
    return Pipeline([("scaler", StandardScaler()),
                    ("lda", LinearDiscriminantAnalysis()),
                    ("svc", build_svc(kernel, C, gamma, degree))])

def build_svc(kernel, C, gamma=0, degree=0):
    """
    initializes svc for pipelines
    """
    match kernel: 
        case "linear": return SVC(max_iter=5000, kernel="linear", C=C)
        case "rbf": return SVC(max_iter=1500, kernel="rbf", C=C, gamma=gamma)
        case "poly":return SVC(max_iter=500, kernel="poly", C=C, gamma=gamma, degree=degree)
    

def time_fit(model, X_train, y_train):
    """
    call fit, time the function
    """
    with warnings.catch_warnings(record=True):
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
    warnings.simplefilter("always") 
    return train_time
    
def run_model(model, X_train, y_train, X_valid, y_valid):
    """
    calls fit, returns validation error, test error and train time 
    """
    train_time = time_fit(model, X_train, y_train)
    y_pred_t = model.predict(X_train)
    y_pred_v = model.predict(X_valid)

    error_v = np.count_nonzero(y_pred_v != y_valid) / len(y_valid)
    error_t = np.count_nonzero(y_pred_t != y_train) / len(y_train)

    return error_v, error_t, train_time

def get_folds(X_train, y_train):
    """
    splits training set into training set and valid set
    returns a list of quatuples
    """
    n = int(input("Enter number of folds: "))

    x_chunks = np.split(X_train, n)
    y_chunks = np.split(y_train, n)
    folds = []
    
    for i in range(n):
        x_valid = x_chunks[i]
        y_valid = y_chunks[i]
        x_tr = np.concatenate(x_chunks[:i] + x_chunks[i+1:])
        y_tr = np.concatenate(y_chunks[:i] + y_chunks[i+1:])
        folds.append((x_tr, y_tr, x_valid, y_valid))
    return folds