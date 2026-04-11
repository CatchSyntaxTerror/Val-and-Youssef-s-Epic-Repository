import torch
import torch.nn as nn
import torch.optim as optimizer
from sklearn.model_selection import KFold
import numpy as np

from src.model import Net
from src.timer import Timer
import src.training_loop as tl


def split_training_data(data,config,i):
    """Creates folds from data
    
    folds :: [data]
    A list of data (same format as data in) with folds instead of raw data
    """

    #Relevant data
    xtr = data["xtr"]
    ytr = data["ytr"]
    k = config["k"]
    n = xtr.shape[0]

    indices = np.array_split(np.arange(n), k)

    val_idx = indices[i]
    train_idx = np.concatenate([indices[j] for j in range(k) if j != i])

    fold_xtr = xtr[train_idx]
    fold_ytr = ytr[train_idx]

    fold_xvld = xtr[val_idx]
    fold_yvld = ytr[val_idx]

    return {"xtr": fold_xtr, "xtst": fold_xvld, "ytr": fold_ytr, "ytst": fold_yvld}
    
def batched_predict(model, x_data, batch_size=512):
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, x_data.shape[0], batch_size):
            xb = x_data[start:start + batch_size]

            # if sparse matrix
            if hasattr(xb, "toarray"):
                xb = xb.toarray()

            xb = torch.tensor(xb, dtype=torch.float32)
            out = model(xb)
            outputs.append(out)

    return torch.cat(outputs, dim=0)

def run_kfold(data,config):
    """Runs k fold on data given config"""
    y_tr = torch.tensor(data["ytr"], dtype=torch.float32)
    y_tst = torch.tensor(data["ytst"], dtype=torch.float32)

    train_sum = None
    test_sum = None

    timing_kfold = Timer()

    for i in range(config["k"]):
        print(f"FOLD {i}")
        print("--------------------------------")

        fold = split_training_data(data, config, i)
        results = tl.my_train(fold, config, False, True)

        train_out = batched_predict(results["model"], data["xtr"], batch_size=512)
        test_out = batched_predict(results["model"], data["xtst"], batch_size=512)

        if train_sum is None:
            train_sum = train_out
            test_sum = test_out
        else:
            train_sum += train_out
            test_sum += test_out

    train_avg = train_sum / config["k"]
    test_avg = test_sum / config["k"]

    train_pred = (torch.sigmoid(train_avg).squeeze() >= 0.5).float()
    test_pred = (torch.sigmoid(test_avg).squeeze() >= 0.5).float()

    time = timing_kfold.stop()
    train_acc = (train_pred == y_tr).float().mean().item()
    test_acc = (test_pred == y_tst).float().mean().item()

    print(f"K-fold accuracy: train: {train_acc}, test: {test_acc}")
    print(f"time: {time}")
    
    
    



  
        