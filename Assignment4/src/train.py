import torch
import torch.nn as nn
import src.utils as utils
import src.visuals as vis
from src.models import RNN
from tqdm import tqdm
"""
Training functions
"""

def run_RNN(data, config=None):
    """ final run of RNN """
    # Todo: replace values with final Hyper parameters
    if config is None: config = {"hidden_size": 32, "num_layers": 2, "lr": 0.001, "epochs": 50} 

    data = utils.to_tensors(data)
    model = RNN(config["hidden_size"], config["num_layers"])
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    train_losses = []
    test_losses = []

    for e in tqdm(range(config["epochs"]), desc="Training RNN", colour="cyan", ncols=80, bar_format="{l_bar}{bar}| [{elapsed}]", ascii=False):
        model.train()
        optim.zero_grad()

        loss = loss_fn(model(data["xtr"]), data["ytr"])
        loss.backward()
        optim.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_pred = model(data["xtst"])
            test_loss = loss_fn(test_pred, data["ytst"])
            test_losses.append(test_loss.item())
            acc = (torch.abs(test_pred - data["ytst"]) < 0.05).float().mean()

        vis.log_epoch(f"Epoch {e+1}: train_loss={loss.item():.6f}, test_loss={test_loss.item():.6f}, acc={acc.item():.4f}\n", "RNN", e+1)
    
    results = {"tr_losses": train_losses, "tst_losses": test_losses, "acc": acc.item()}
    return config, results
    
def run_GRU(data):
    """ final run of GRU """

def run_LSTM(data):
    """ final run of LSTM """


def tune_RNN(data):
    """ the tuning pipeline for RNN """
    for i in range(5):
        print(f"\nRound {i+1}:")
        config = {
            "hidden_size": int(input("Hidden size: ")),
            "num_layers": int(input("Number of layers: ")),
            "lr": float(input("Learning rate: ")),
            "epochs": int(input("Epochs: "))
            }
        config, results = run_RNN(data, config)
        vis.log_run(config, results, "RNN", tuning=True)

def tune_GRU(data):
    """ the tuning pipeline for GRU """

def tune_LSTM(data):
    """ the tuning pipeline for LSTM """

def run_training_loop(model_str, data):
    """ training entry point """
    mode = int(input("Are you tuning:\n1) Yes \n2) No\n"))
    match model_str:
        case "RNN": 
            if mode == 1: 
                tune_RNN(data)
            else: 
                config, results = run_RNN(data)
                vis.log_run(config, results, "RNN")

        case "GRU":  
            if mode == 1: 
                tune_GRU(data)
            else: 
                config, results = run_GRU(data)
                vis.log_run(config, results, "GRU")
        case "LSTM": 
            if mode == 1: 
                tune_LSTM(data)
            else: 
                config, results = run_LSTM(data)
                vis.log_run(config, results, "LSTM")
    