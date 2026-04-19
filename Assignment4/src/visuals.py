import os
import datetime
import matplotlib.pyplot as plt

"""
Functions for making tables and graphs
"""
BASE_DIR = "output"
RESULTS_DIR = f"{BASE_DIR}/results"
x = 1

def plot_stocks(stock_data):
    """ plots the raw stock data """

    out = f"{BASE_DIR}/raw_stocks"
    os.makedirs(out, exist_ok=True)

    for stock, data in stock_data.items():
        plt.figure()
        plt.plot(data["Close"])
        plt.title(stock)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.savefig(f"{out}/{stock}_raw.png")
        plt.close()

def log_stocks(stock_data):
    """ print data to log for easy read """

    out = f"{BASE_DIR}/raw_stocks"
    os.makedirs(out, exist_ok=True)
    s = "".join(k[0] for k in stock_data.keys())
    
    with open(f"{out}/{s}.log", "w") as f:
        f.write(f"Log time: {datetime.datetime.now()}\n\n")

        for stock, data in stock_data.items():
            f.write(f"{stock}\n")
            f.write(str(data.head()) + "\n")
            f.write(str(data.columns) + "\n\n")

def log_run(config, results, str, tuning=False):
    """ Log the output of RNN runs """

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{str}_results.log", "a") as f:
        f.write("Tuning Run:\n") if tuning else f.write("Final Run:\n")
        f.write(f"\tHyper Parameters: {config}\n\tAccuracy: {results["acc"]}\n\n")
        f.close()

def log_epoch(str, model:str, epoch):
    """log loss per epoch"""
    global x
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{model}_losses.log", "a") as f:
        if x == 1: f.write("\nNew Run\n\n")
        if epoch == 1:
            f.write(f"\nRound {x}:\n\n")
            x += 1
        f.write(str)
        f.close()