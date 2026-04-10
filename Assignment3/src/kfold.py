import torch
import torch.nn as nn
import torch.optim as optimizer
from sklearn.model_selection import KFold
import numpy as np

from src.model import Net
from src.timer import Timer
import src.training_loop as tl


def decision(y_preds):
    avg = np.mean(y_preds, axis=0)
    eval_probs = torch.sigmoid(avg).squeeze()
    out = (eval_probs >= 0.5).float()
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
    models = []
    train_pre_activation = []
    test_pre_activation = []
    train_avg = 0.0
    test_avg = 0.0

    x_tr = torch.tensor(data["xtst"].toarray(), dtype=torch.float32)
    y_tr = torch.tensor(data["ytst"], dtype=torch.float32)
    x_tst = torch.tensor(data["xtst"].toarray(), dtype=torch.float32)
    y_tst = torch.tensor(data["ytst"], dtype=torch.float32)

    timing_kfold = Timer()
    kfold = KFold(n_splits=config["k"], shuffle=True, random_state=42)
    for fold in folds:
        print(f'FOLD {fold}')
        print('--------------------------------')
        results = tl.my_train(folds[fold],config,False,True)
        valid_results.append(results)
        models.append(results["model"])
        #print(f"Results for fold {fold} validation: ")
        
        train_pre_activation.append(results["model"](x_tr))
        test_pre_activation.append(results["model"](x_tst))
    
    train_pred = decision(train_pre_activation)
    test_pred = decision(test_pre_activation)
    
    time = timing_kfold.stop()
    train_acc = (train_pred == y_tr).float().mean().item()
    test_acc = (test_pred == y_tst).float().mean().item()
    
    print(f"K-fold accuracy: train: {train_acc}, test: {test_acc}")
    print(f"time: {time}")
    
    
    



  
        