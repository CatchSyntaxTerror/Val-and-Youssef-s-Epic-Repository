import src.training_loop as tl
import src.record as rec
import src.kfold as k
import src.dropout as drop
import src.logistic_regression as logi
"""
Exposed functions for main to call
"""

def run_baseline(data, config, use_test = False):
    """
    baseline
    """
    if use_test: results = tl.my_train(data, config, use_test=True)
    else: results = tl.my_train(data, config)
    rec.save_baseline(config, results, use_test)

def run_kfold(data, config):
    """
    k-fold
    """
    k.run_kfold(data, config)

def run_dropout(data, config):
    """
    dropout
    """
    results = drop.run_dropout(data, config)
    rec.save_others(config, results)

def run_LR(data, config):
    """
    logistic regression
    """
    results = logi.run_logistic_regression(data, config)
    rec.save_others(config, results)

def get_baseline_config():
    """
    Get info for training baseline
    """
    hls = list(map(int, input("Enter Hidden Layers (ex: 128 64 ...): ").split())) 
    lr = float(input("Enter learning rate: "))
    w_decay = float(input("Enter weight decay: "))
    epochs = int(input("Enter epochs: "))

    return {"hls": hls, "lr": lr, "w_decay": w_decay, "epochs": epochs}

def get_kfold_config():
    """
    get info for kfold config
    """
    config = {'hls': [128, 64, 32, 32], 'lr': 0.00140001, 'w_decay': 6.1e-06, 'epochs': 51}
    config["k"] = int(input("Number of Folds: "))
    return config

def get_dropout_config():
    """
    get info for droupout config
    """
    config = {'hls': [128, 64, 32, 32], 'lr': 0.00140001, 'w_decay': 6.1e-06, 'epochs': 51}
    config["dropout"] = float(input("Dropout rate: "))
    config["num_models"] = int(input("Number of models: "))
    return config

def get_LR_config():
    """
    Get info for training LR
    """
    return {'lr': 0.00140001, 'w_decay': 6.1e-06, 'epochs': 51}