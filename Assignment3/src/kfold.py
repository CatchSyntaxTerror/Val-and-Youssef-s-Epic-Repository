import torch
import torch.nn as nn
import torch.optim as optimizer
from sklearn.model_selection import KFold
import numpy as np

from src.model import Net
from src.timer import Timer
import src.training_loop as tl


def decision(y_preds):
    np_preds = np.array(y_preds)
    avg = np.mean(np_preds)
    #TODO: sigmoid on out
    out = avg
    return out


def split_training_data(data,config):
    """Creates folds from data
    
    folds :: [data]
    A list of data (same format as data in) with folds instead of raw data
    """

    #Relevant data
    xtr = data["xtr"]
    ytr = data["ytr"]
    k = config["k"]
    n = xtr.shape[0]

    folds = []

    for i in range(k):
        # Determine indices
        split_start_index = i*(n/k)
        split_end_index = (i+1)*(n/k)

        #Training split
        fold_xtr = np.concatenate((xtr[:split_start_index],xtr[split_end_index:]))
        fold_ytr = np.concatenate((ytr[:split_start_index],ytr[split_end_index:]))

        #Valid split
        fold_xvld = xtr[split_start_index:split_end_index]
        fold_yvld = ytr[split_start_index:split_end_index]

        #Package out
        (folds[i])["xtr"] = fold_xtr
        (folds[i])["xtst"] = fold_xvld
        (folds[i])["ytr"] = fold_ytr
        (folds[i])["ytst"] = fold_yvld

    return folds

    

def run_kfold(data,config):
    """Runs k fold on data given config"""
    folds = split_training_data(data,config)
    valid_results = []
    test_pre_activation = []
    test_avg = 0.0
    x_tst = data["xtst"]
    y_tst = data["ytst"]

    kfold = KFold(n_splits=config["k"], shuffle=True, random_state=42)
    for fold in folds:
        print(f'FOLD {fold}')
        print('--------------------------------')
        valid_results.append(tl.my_train(folds[fold],config,False))
        #TODO: extract model from this for final step
        print(valid_results[fold])
        #TODO: run models prediction pre activation and append to list
    
    avg_pred = decision(test_pre_activation)
    #TODO: evaluate avg pred on accuracy, runtime
    
    
    



  
        