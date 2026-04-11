import numpy as np
import torch
import src.training_loop as tl
from src.kfold import decision
from src.timer import Timer
"""
Functions for Task 5 dropout comparason
"""

def split_data(data, config):
    """
    split test data into num_bags sets 
    """
    return np.array_split(data["xtr"], config["num_models"]), np.array_split(data["ytr"], config["num_models"])

def run_dropout(data, config):
    """
    1. split data
    2. make models
    3. train models
    4. collect preactiviation
    5. select most confident
    6. return {results}
    """
    og_xtr, og_ytr = data["xtr"], data["ytr"]
    xtrs, ytrs = split_data(data, config)
    results = []
    tr_pre_act = []
    tst_pre_act = []
    xtr_tens = torch.tensor(og_xtr.toarray(), dtype=torch.float32)
    xtst_tens = torch.tensor(data["xtst"].toarray(), dtype=torch.float32)

    t = Timer()
    for x, y in zip(xtrs, ytrs):
        data["xtr"], data["ytr"] = x, y
        results.append(tl.my_train(data, config, use_test=False, ret_model=True))
        with torch.no_grad():
            tr_pre_act.append(results[-1]["model"](xtr_tens))
            tst_pre_act.append(results[-1]["model"](xtst_tens))

    tr_preds = decision(tr_pre_act)
    tst_preds = decision(tst_pre_act)
    t_time = t.stop()

    tr_acc = (tr_preds == torch.tensor(og_ytr, dtype=torch.float32)).float().mean().item()
    tst_acc = (tst_preds == torch.tensor(data["ytst"], dtype=torch.float32)).float().mean().item()

    return {"tst_acc": tst_acc, "tr_acc": tr_acc, "time": t_time, "num_models": config["num_models"]}
