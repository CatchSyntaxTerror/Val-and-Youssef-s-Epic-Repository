import numpy as np
import torch
import src.training_loop as tl
from src.timer import Timer

"""
Functions for Task 5 dropout comparison
"""

def split_data(data, config):
    """
    split training data into num_models sets
    """

    indecies = np.array_split(np.arange(data["xtr"].shape[0]), config["num_models"])
    xs, ys = [], []

    for i in range(config["num_models"]):
        xs.append(data["xtr"][indecies[i]])
        ys.append(data["ytr"][indecies[i]])

    return xs, ys


def batched_predict(model, x_data, batch_size=512):
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, x_data.shape[0], batch_size):
            xb = x_data[start:start + batch_size]

            if hasattr(xb, "toarray"):
                xb = xb.toarray()

            xb = torch.tensor(xb, dtype=torch.float32)
            out = model(xb)
            outputs.append(out)

    return torch.cat(outputs, dim=0)


def run_dropout(data, config):
    """
    1. split data
    2. make models
    3. train models
    4. collect predictions with batching
    5. average outputs
    6. return results
    """
    og_xtr, og_ytr = data["xtr"], data["ytr"]
    xtrs, ytrs = split_data(data, config)

    train_sum = None
    test_sum = None

    t = Timer()

    i = 0
    for x, y in zip(xtrs, ytrs):
        print(f"ROUND {i}")
        i += 1
        data["xtr"], data["ytr"] = x, y
        result = tl.my_train(data, config, use_test=False, ret_model=True)

        tr_out = batched_predict(result["model"], og_xtr, batch_size=512)
        tst_out = batched_predict(result["model"], data["xtst"], batch_size=512)

        if train_sum is None:
            train_sum = tr_out
            test_sum = tst_out
        else:
            train_sum += tr_out
            test_sum += tst_out

    train_avg = train_sum / config["num_models"]
    test_avg = test_sum / config["num_models"]

    tr_preds = (torch.sigmoid(train_avg).squeeze() >= 0.5).float()
    tst_preds = (torch.sigmoid(test_avg).squeeze() >= 0.5).float()

    t_time = t.stop()

    tr_acc = (tr_preds == torch.tensor(og_ytr, dtype=torch.float32)).float().mean().item()
    tst_acc = (tst_preds == torch.tensor(data["ytst"], dtype=torch.float32)).float().mean().item()

    return {"tst_acc": tst_acc, "tr_acc": tr_acc, "time": t_time, "num_models": config["num_models"]}