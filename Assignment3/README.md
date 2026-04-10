# Assignment 3 - Sentiment Classification

Authors: Youssef Amin, Valerie Barker

## Overview
This project builds a neural network to classify IMDb movie reviews.

## Files

- `main.py`  
  Entry point. Loads data, and runs tests.

- `src/API.py`  
  Contains wrappers for main to run tests

- `src/data_loader.py`    
  Loads, preprocesses, shuffles, splits, and vectorizes the dataset.

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

## How to Run

```bash
python3 main.py