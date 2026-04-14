# Assignment 4 - Stock Market Prediction using Recurrent Neural Networks

Authors: Youssef Amin, Valerie Barker

## Overview  
This project builds and compares RNN, GRU, and LSTM models to predict stock prices.

## Directories

- `data\`  
  Contains CSV files for each stock

- `output\`  
  All outputs from the program

- `src\`  
  The module which contains all our code. Used by `main.py` and `get_stock_data.py`

## Files

- `main.py`  
  Entry point. Loads data, preprocesses, and runs training for each stock.

- `get_stock_data.py`  
  a script to load stock data from Yahoo Finance and saves it as CSV files.

- `src/data_loader.py`  
  Loads CSV data, scales values, splits into train/test sets, and creates input/output sequences.

- `src/models.py`  
  Defines the RNN, GRU, and LSTM models.

- `src/train.py`  
  Handles training and computes accuracy.

- `src/utils.py`  
  Helper functions.

- `src/visuals.py`  
  Generates plots and logs for stock data.

## How to Run
To populate CSVs:
```bash
python3 get_stock_data.py
```
To run main program:
```bash
python3 main.py