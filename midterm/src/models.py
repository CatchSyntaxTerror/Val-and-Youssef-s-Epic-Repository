from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

import numpy as np
import time as time
import warnings
import src.data_loader as dl
import src.graphing as gr

"""
This file handles standardization, dimensionality reduction, SVC training, tuning, and evaluation.
Here are some helpful links:
    https://scikit-learn.org/stable/modules/compose.html#pipeline
    https://scikit-learn.org/stable/modules/decomposition.html#pca
    https://scikit-learn.org/stable/modules/lda_qda.html
"""

def pca_model(X_train, y_train, X_valid, y_valid, num_comps,  ker, C, gamma, degree):
    match ker:
        case "linear":  model = Pipeline([("scaler", StandardScaler()),
                                           ("pca", PCA(n_components=num_comps)),
                                           ("svc", SVC(max_iter=100, kernel=ker, C=C))])
        case "rbf": model = Pipeline([("scaler", StandardScaler()),
                                        ("pca", PCA(n_components=num_comps)),
                                        ("svc", SVC(max_iter=100, kernel=ker, C=C, gamma=gamma))])
        case "poly": model = Pipeline([("scaler", StandardScaler()),
                                        ("pca", PCA(n_components=num_comps)),
                                        ("svc", SVC(max_iter=100, kernel=ker, C=C, gamma=gamma, degree=degree))])

    with warnings.catch_warnings(record=True) as recorded_warnings:
        train_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_time
    warnings.simplefilter("always") 

    y_pred_v = model.predict(X_valid)
    y_pred_t = model.predict(X_train)

    return y_pred_v, y_pred_t, train_time
    
def lda_model(X_train, y_train, X_valid, y_valid, ker, C, gamma, degree):
    match ker:
        case "linear": model  = Pipeline([("scaler", StandardScaler()),
                                            ("lda", LinearDiscriminantAnalysis()),
                                            ("svc", SVC(max_iter=100, kernel="linear", C=C))])
        
        case "rbf": model  = Pipeline([("scaler", StandardScaler()),
                                            ("lda", LinearDiscriminantAnalysis()),
                                            ("svc", SVC(max_iter=100, kernel="rbf", C=C, gamma=gamma))])
    
        case "poly": model  = Pipeline([("scaler", StandardScaler()),
                                            ("lda", LinearDiscriminantAnalysis()),
                                            ("svc", SVC(max_iter=100, kernel="poly", C=C, gamma=gamma, degree=degree))])
    
    with warnings.catch_warnings(record=True) as recorded_warnings:
        train_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_time
    warnings.simplefilter("always") 

    y_pred_v = model.predict(X_valid)
    y_pred_t = model.predict(X_train)
    
    return y_pred_v, y_pred_t, train_time

def pca_pipeline(X_train, y_train, X_valid, y_valid, num_comps,  ker, C, gamma, degree):
    """
    This function creates a PCA pipeline, calls fit and computes the prediction error
    """
    y_pred_v, y_pred_t, train_time = pca_model(X_train, y_train, X_valid, y_valid, num_comps, ker, C, gamma, degree)
    error_v = np.count_nonzero(y_pred_v != y_valid) / len(y_valid)
    error_t = np.count_nonzero(y_pred_t != y_train) / len(y_train)
    return error_v, error_t, train_time

def lda_pipeline(X_train, y_train, X_valid, y_valid, ker, C, gamma, degree):
    """
    This function creates a LDA pipeline, calls fit and computes the prediction error
    """
    y_pred_v, y_pred_t, train_time = lda_model(X_train, y_train, X_valid, y_valid, ker, C, gamma, degree)
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

def test_pca(X_train, y_train, X_test, y_test, kernal):
    """
    tune PCA hyper parameters and record results
    """
    folds = get_folds(X_train, y_train)
    num_comps = get_comps()
    err_v = 0
    err_t = 0
    time = 0

    for x_re, y_tr, x_valid, y_valid in folds:
        C, gamma, degree = get_params(kernal,err_v,err_t,time)
        err_v, err_t, time = pca_pipeline(x_re, y_tr, x_valid, y_valid, num_comps, kernal, C, gamma, degree)
        gr.record_result(err_v, err_t, 0, time, C, gamma, degree, kernal, True, num_comps, True)
    C, gamma, degree = get_params(kernal,err_v,err_t,time, test=True)
    err_v, err_t, time = pca_pipeline(X_train, y_train, X_test, y_test, num_comps, kernal, C, gamma, degree)
    gr.record_result(0, err_t, err_v, time, C, gamma, degree, kernal, False, num_comps, True)
    print(f"Final: error_test: {err_v}, time: {time}")

def test_lda(X_train, y_train, X_test, y_test, kernal):
    """
    tune PCA hyper parameters and record results
    """
    folds = get_folds(X_train, y_train)

    err_v = 0
    err_t = 0
    time = 0

    for x_re, y_tr, x_valid, y_valid in folds:
        C, gamma, degree = get_params(kernal,err_v,err_t,time)
        err_v, err_t, time = lda_pipeline(x_re, y_tr, x_valid, y_valid, kernal, C, gamma, degree)
        gr.record_result(err_v, err_t, 0, time, C, gamma, degree, kernal, True, 0, False)
    C, gamma, degree = get_params(kernal,err_v,err_t,time, test=True)
    err_v, err_t, time = lda_pipeline(X_train, y_train, X_test, y_test, kernal, C, gamma, degree)
    gr.record_result(0, err_t, err_v, time, C, gamma, degree, kernal, False, 0, False)
    print(f"Final: error_test: {err_v}, time: {time}")

    

def get_params(ker, err_v=0.0, err_t=0.0, time=0.0, test = False):
    """
    Gets C, gamma and degree from user.
    """
    if test: str = "Input Final "
    else: str = "Input "
    print(f"error_v: {err_v}, error_t: {err_t}, time: {time}")
    match ker:
        case "linear": 
            C = float(input(f"{str}C for linear: "))
            gamma = -1.0
            degree = -1
        case "rbf":
            C = float(input(f"{str}C: "))
            gamma = float(input(f"{str}gamma: "))
            degree = -1
        case "poly":
            C = float(input(f"{str}C: "))
            gamma = float(input(f"{str}gamma: "))
            degree = int(input(f"{str}degree: "))
    return C, gamma, degree

def get_comps():
    n = int(input("enter number of comps for PCA: "))
    return n


def tewst():
    ker = input("Enter Kernel: ")
    X_train, y_train, X_test, y_test = dl.load_mnist();
    # test_pca(X_train, y_train, X_test, y_test, ker)
    test_lda(X_train, y_train, X_test, y_test, ker)

