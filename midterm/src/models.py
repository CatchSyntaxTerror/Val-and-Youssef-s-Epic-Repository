from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

import numpy as np
import time as time
import src.data_loader as dl

"""
This file handles standardization, dimensionality reduction, SVC training, tuning, and evaluation.
Here are some helpful links:
    https://scikit-learn.org/stable/modules/compose.html#pipeline
    https://scikit-learn.org/stable/modules/decomposition.html#pca
    https://scikit-learn.org/stable/modules/lda_qda.html
"""

def pca_pipeline(X_train, y_train, X_valid, y_valid, num_comps,  ker, C, gamma, degree):
    """
    This function creates a PCA pipeline, calls fit and computes the prediction error
    """
    match ker:
        case "linear":  model = Pipeline([("scaler", StandardScaler()),
                                           ("pca", PCA(n_components=num_comps)),
                                           ("svc", SVC(kernel=ker, C=C))])
        case "rbf": model = Pipeline([("scaler", StandardScaler()),
                                        ("pca", PCA(n_components=num_comps)),
                                        ("svc", SVC(kernel=ker, C=C, gamma=gamma))])
        case "poly": model = Pipeline([("scaler", StandardScaler()),
                                        ("pca", PCA(n_components=num_comps)),
                                        ("svc", SVC(kernel=ker, C=C, gamma=gamma, degree=degree))])
        
    train_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_time

    y_pred = model.predict(X_valid)
    error = np.count_nonzero(y_pred != y_valid) / len(y_valid)
    return error, train_time

def lda_pipeline(X_train, y_train, X_valid, y_valid, ker, C, gamma, degree):
    """
    This function creates a LDA pipeline, calls fit and computes the prediction error
    """
    match ker:
        case "linear": model  = Pipeline([("scaler", StandardScaler()),
                                            ("lda", LinearDiscriminantAnalysis()),
                                            ("svc", SVC(kernel="linear", C=C))])
        
        case "rbf": model  = Pipeline([("scaler", StandardScaler()),
                                            ("lda", LinearDiscriminantAnalysis()),
                                            ("svc", SVC(kernel="linear", C=C, gamma=gamma))])
    
        case "poly": model  = Pipeline([("scaler", StandardScaler()),
                                            ("lda", LinearDiscriminantAnalysis()),
                                            ("svc", SVC(kernel="linear", C=C, gamma=gamma, degree=degree))])
    train_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_time

    y_pred = model.predict(X_valid)
    error = np.count_nonzero(y_pred != y_valid) / len(y_valid)
    return error, train_time

def get_folds(X_train, y_train):
    """
    splits training set into training set and valid set
    returns a list of quatuples
    """

    x_chunks = np.split(X_train, 5)
    y_chunks = np.split(y_train, 5)
    folds = []
    
    for i in range(5):
        x_valid = x_chunks[i]
        y_valid = y_chunks[i]
        x_tr = np.concatenate(x_chunks[:i] + x_chunks[i+1:])
        y_tr = np.concatenate(y_chunks[:i] + y_chunks[i+1:])
        
        folds.append((x_tr, y_tr, x_valid, y_valid))
    return folds

def test_pca(X_train, y_train, X_test, y_test):
    """
    tune PCA hyper parameters and record results
    """
    folds = get_folds(X_train, y_train)

    for x_re, y_tr, x_valid, y_valid in folds:
        """do shit"""

def test_lda(X_train, y_train, X_test, y_test):
    """
    tune PCA hyper parameters and record results
    """
    folds = get_folds(X_train, y_train)

    for x_re, y_tr, x_valid, y_valid in folds:
        """do shit"""
