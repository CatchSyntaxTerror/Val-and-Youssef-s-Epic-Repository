import src.models as mods
import numpy as np
import src.analysis as ana
import time as time

"""
functions for bootstrap aggregating with multiple SVC models.
"""
def model_prediction(tech, x_bag, y_bag, X_test, ker, C, gamma, degree, num_comps):
    """
    train a model on data block
    """
    if tech == "pca": model = mods.build_pca_model(num_comps, ker, C, gamma, degree)
    else: model = mods.build_lda_model(num_comps, ker, C, gamma, degree)

    mods.time_fit(model, x_bag, y_bag)
    return model.predict(X_test)

def predictions(tech, x_bags, y_bags, X_test):
    """
    train one model for each bag and return a list predictions
    """
    ker, C, gamma, degree, num_comps = ana.get_input(tech)
    y_preds = []
    for i in range(len(x_bags)):
        y_preds.append(model_prediction(tech, x_bags[i], y_bags[i], X_test, ker, C, gamma, degree, num_comps))
    return y_preds, ker, num_comps

def calc_votes(y_preds):
    """
    calculate votes and for the final prediction
    """
    y_preds = np.array(y_preds).T

    vote_pred = []
    for col in y_preds:
        preds, votes = np.unique(col, return_counts=True)
        vote_pred.append(preds[np.argmax(votes)])
    return vote_pred

def run_bagging(tech, X_train, y_train, X_test, y_test, num_bags):
    """
    split data into bags, train models, get votes, calculate error. 
    """
    np.random.seed(42)

    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]

    x_bags = np.array_split(X_train, num_bags)
    y_bags = np.array_split(y_train, num_bags)
    
    start = time.time()
    y_preds, ker, num_comps = predictions(tech, x_bags, y_bags, X_test)
    voted = calc_votes(y_preds)

    error = np.count_nonzero(voted != y_test) / len(voted)
    return time.time() - start, error, ker, num_comps