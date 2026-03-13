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

def pca_pipeline(num_comps,  ker, C, gamma, degree):
    """
    This function creates a PCA pipeline, calls fit and computes the prediction error
    """
    X_train, y_train, X_test, y_test = dl.load_mnist()
    
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

    y_pred = model.predict(X_test)
    error = np.count_nonzero(y_pred != y_test) / y_test.size
    return error, train_time

def lda_pipeline(ker, C, gamma, degree):
    """
    This function creates a LDA pipeline, calls fit and computes the prediction error
    """
    X_train, y_train, X_test, y_test = dl.load_mnist()

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

    y_pred = model.predict(X_test)
    error = np.count_nonzero(y_pred != y_test) / y_test.size
    return error, train_time

def test_pca():
    """
    used to tune PCA hyper parameters and graph
    """
def test_lda():
    """
    used to tune PCA hyper parameters and graph
    """
