import yfinance as yf
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
"""
load data from yahoo, preprocess data
"""

def save_data(stocks):
    """
    load stock data from yahoo 
    1h interval and 2 year period <- I'm cool withy changing this, but hourly max is 730 days
    Save to CSV
    """
    os.makedirs("data", exist_ok=True)
    for stock in stocks:
        d = yf.download(stock, interval="1h", period="2y")
        d = d.reset_index()
        d.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        d.to_csv(f"data/{stock}.csv", index=False)
    
def load_save_data(stocks):
    """
    load stock data from CSVs
    """
    data = {}
    for s in stocks:
        data[s] = pd.read_csv(f"data/{s}.csv", index_col=0, parse_dates=["Date"])
    return data

def split_sets(prices):
    """
    split data into training and test sets
    """
    idx = int(len(prices) * 0.8)
    train, test = prices[:idx], prices[idx:]
    train, test = train.reshape(-1, 1), test.reshape(-1, 1)
    scaler = MinMaxScaler()
    
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled

def make_xy(set, size):
    """
    split the set into X and y 
    """
    X, y = [], []
    for i in range(len(set) - size):
        X.append(set[i:i+size])
        y.append(set[i+size])
    return X, y

def split_data(ps, n):
    """
    split data into train and test sets
    split sets into X, y
    """
    train, test = split_sets(ps)
    X_train, y_train = make_xy(train, n)
    X_test, y_test = make_xy(test, n)
    return X_train, y_train, X_test, y_test


def load_data(stock, split_size):
    """
    Load and preprocess data
    """
    data = load_save_data([stock])[stock]
    X_train, y_train, X_test, y_test = split_data(data["Close"].values)
    return {"xtr": X_train, "ytr": y_train, "xtst": X_test, "ytst": y_test}
    
