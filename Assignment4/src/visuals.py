import os
import datetime
import matplotlib.pyplot as plt

"""
Functions for making tables and graphs
"""
BASE_DIR = "output"

def plot_stocks(stock_data):
    """
    plots the raw stock data
    """
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
    """
    print data to log for easy read
    """
    out = f"{BASE_DIR}/raw_stocks"
    os.makedirs(out, exist_ok=True)
    s = "".join(k[0] for k in stock_data.keys())
    
    with open(f"{out}/{s}.log", "w") as f:
        f.write(f"Log time: {datetime.datetime.now()}\n\n")

        for stock, data in stock_data.items():
            f.write(f"{stock}\n")
            f.write(str(data.head()) + "\n")
            f.write(str(data.columns) + "\n\n")