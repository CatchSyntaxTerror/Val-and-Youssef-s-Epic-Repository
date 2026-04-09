import src.training_loop as tl
"""
Exposed functions for main to call
"""

def run_baseline(data, config, use_test = False):
    """
    Task 3 tune the hyper perameters
    """
    if use_test: tl.train(data, config, use_test=True)
    else: tl.train(data, config)

def run_kfold(data, config):
    """"""

def run_dropout(data, config):
    """"""

def get_baseline_config():
    """
    Get info for training baseline
    """
    hls = list(map(int, input("Enter Hidden Layers (ex: 128 64 ...): ").split()))
    lr = float(input("Enter learning rate: "))
    w_decay = float(input("Enter weight decay: "))
    epochs = int(input("enter epochs: "))

    return {"hls": hls, "lr": lr, "w_decay": w_decay, "epochs": epochs}

def get_kfold_config():
    """
    get info for kfold config
    """
    config = get_baseline_config()
    config["k"] = int(input("Number of Folds: "))
    return config
def get_dropout_config():
    """
    get info for droupout config
    """
    config = get_baseline_config()
    config["dropout"] = float(input("Dropout rate: "))
    config["num_models"] = int(input("Number of models: "))
    return config

