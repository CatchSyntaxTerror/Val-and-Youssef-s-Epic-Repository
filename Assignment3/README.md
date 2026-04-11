# Assignment 3 - Sentiment Classification

Authors: Youssef Amin, Valerie Barker

## Overview
This project builds a neural network to classify IMDb movie reviews.

## Directories

- `aclimbd\`  
  The Imbd movie data

- `outputs\`  
  All outputs from the program

- `src\`  
  The module which contains all our code. Used by `main.py`

## Files

- `main.py`  
  Entry point. Loads data, and runs tests.

- `src/API.py`  
  Contains wrappers for main to run tests

- `src/data_loader.py`    
  Loads, preprocesses, shuffles, splits, and vectorizes the dataset.

- `kfold.py`  
  Implements Kfold for task 4

- `droput.py`  
  Implements dropout for task 5

- `src/model.py`   
  Defines the neural network (fully connected with configurable layers and dropout).

- `src/record.py`  
  Writes resulyts and makes graphs.

- `src/timer.py`  
  A classic meme.

- `src/training_loop.py`  
  Convert data to tensors, trains model, computes loss, times training, returns results.

- `aclImdb/`  
  Dataset containing training and test reviews.

- `logistic_regression.py`  
  The textbooks Logistic regression code for analyzing the movie data

## How to Run

```bash
python3 main.py