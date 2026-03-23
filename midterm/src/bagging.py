import src.models as mods
import numpy as np
import src.analysis as ana
import time as time

"""
functions for bootstrap aggregating with multiple SVC models.
"""
def model_prediction(tech, x_bag, y_bag, X_test):
    """
    train a model on data block
    """
    ker, C, gamma, degree, num_comps = ana.get_input(tech)
    if tech == "pca": model = mods.build_pca_model(num_comps, ker, C, gamma, degree)
    else: model = mods.build_lda_model(num_comps, ker, C, gamma, degree)

    mods.time_fit(model, x_bag, y_bag)
    return model.predict(X_test)

def predictions(tech, x_bags, y_bags, X_test):
    """
    train one model for each bag and return a list predictions
    """
    y_preds = []
    for i in range(len(x_bags)):
        y_preds.append(model_prediction(tech, x_bags[i], y_bags[i], X_test))
    return y_preds

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
    start = time.time()
    ana.get_input(tech)
    x_bags = np.split(X_train, num_bags)
    y_bags = np.split(y_train, num_bags)
    

    y_preds = predictions(tech, x_bags, y_bags, X_test)
    voted = calc_votes(y_preds)
    return time.time() - start, np.count_nonzero(voted != y_test) / len(voted)