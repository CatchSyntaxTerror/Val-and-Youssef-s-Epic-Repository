import torch
import torch.nn as nn
import torch.optim as optimizer
from src.model import Net
from src.timer import Timer

def split_train_val(xtr, ytr):
    """
    get a validation set
    """
    shuff = torch.randperm(xtr.size(0))
    split_idx = int(0.8 * xtr.size(0))
    xtr = xtr[shuff]
    ytr = ytr[shuff]
    return xtr[:split_idx], ytr[:split_idx], xtr[split_idx:], ytr[split_idx:]

def to_tensors(data):
    """
    convert data to tensors
    """
    X_train = torch.tensor(data["xtr"].toarray(), dtype=torch.float32)
    y_train = torch.tensor(data["ytr"], dtype=torch.float32)
    X_test = torch.tensor(data["xtst"].toarray(), dtype=torch.float32)
    y_test = torch.tensor(data["ytst"], dtype=torch.float32)

    return X_train, y_train, X_test, y_test

def initialize(data, config):
    """
    initialize everything for loop
    """
    xtr, ytr, xtst, ytst = to_tensors(data)
    model = Net(xtr.shape[1], config["hls"], config.get("dropout", 0.0))
    loss_func = nn.BCEWithLogitsLoss()
    optim = optimizer.Adam(model.parameters(), lr=config["lr"], weight_decay=config["w_decay"])
    return xtr, ytr, xtst, ytst, model, loss_func, optim

def my_train(data, config, use_test = False, ret_model=False):
    """
    The main training loop
    """
    xtr, ytr, xtst, ytst, model, loss_func, optim = initialize(data, config)
    if not use_test: xtr, ytr, xtst, ytst = split_train_val(xtr, ytr)
    
    timer = Timer()
    for e in range(config["epochs"]):
        model.train()

        optim.zero_grad()
        outs = model(xtr).squeeze()
        loss = loss_func(outs, ytr)
        loss.backward()
        optim.step()
    tr_time = timer.stop()

    model.eval()
    with torch.no_grad():
        tr_probs = torch.sigmoid(model(xtr)).squeeze()
        tr_preds = (tr_probs >= 0.5).float()
        tr_acc = (tr_preds == ytr).float().mean().item()

        if use_test: acc_name = "tst_acc"
        else: acc_name = "val_acc"

        eval_probs = torch.sigmoid(model(xtst)).squeeze()
        eval_preds = (eval_probs >= 0.5).float()
        eval_acc = (eval_preds == ytst).float().mean().item()

    if (ret_model) :
        return {acc_name: eval_acc, "tr_acc": tr_acc, "time": tr_time, "model": model}
    else :
        return {acc_name: eval_acc, "tr_acc": tr_acc, "time": tr_time}











